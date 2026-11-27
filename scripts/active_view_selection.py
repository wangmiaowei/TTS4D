#!/usr/bin/env python3
"""Active view selection with the predicted uncertainty (supplementary, Mip-NeRF 360 BONSAI, following FisherRF).

Candidates = the 3DGS training split of the scene (name-sorted images, index % 8 != 0; 255 views for bonsai), held-out
views = the 3DGS test views (index % 8 == 0; 37 for bonsai).  Every strategy starts from the same --n_start views with
maximal pairwise camera distance (greedy farthest-point) and adds one view per step until --n_final:
  unc       the candidate with the largest mean predicted standard deviation exp(U/2): the memory is solved from the
            views chosen so far and every remaining candidate is rendered with its uncertainty (no retraining)
  unc_epi   the same with the epistemic part only, exp(U_epi/2)
  farthest  the candidate farthest from the chosen cameras (geometry only)
  random    uniformly at random (one run per --seeds entry)
After every step the held-out views are rendered from the current inputs and scored (PSNR of the 8-bit rendering, 3DGS
SSIM, LPIPS-VGG).  Images at --short pixels short side (512 in the paper's figure); scene unit = span of all cameras of
the scene, floored at the 20th percentile of the spans of the Mip-NeRF 360 scenes found under --root.

Outputs in --out: results.json (per strategy: chosen views, scores, metrics per step), curves.png, and the held-out
renderings of the steps in --save_steps (renders/<strategy>/n<k>/).

Example:
  python scripts/active_view_selection.py --root data/raw/mipnerf360 --out results/avs_bonsai
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tts4d import metrics as M  # noqa: E402
from tts4d.data import static as S  # noqa: E402
from tts4d.data.common import read_rgb, resize_short, span, to_model_frame  # noqa: E402
from tts4d.runner import SceneRenderer, add_model_args, model_from_args  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Mip-NeRF 360 folder (<scene>/images*, <scene>/sparse/0)")
    ap.add_argument("--scene", default="bonsai")
    ap.add_argument("--out", required=True)
    ap.add_argument("--short", type=int, default=512, help="short side of the images (0 = 3DGS evaluation resolution)")
    ap.add_argument("--n_start", type=int, default=4)
    ap.add_argument("--n_final", type=int, default=20)
    ap.add_argument("--strategies", default="unc,random", help="comma list of unc, unc_epi, farthest, random")
    ap.add_argument("--seeds", default="0,1,2", help="seeds of the random strategy")
    ap.add_argument("--save_steps", default="4,8,12,20", help="numbers of views at which the held-out renderings are saved")
    add_model_args(ap)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device

    # ---- scene: every image once, at the working resolution
    ds = "tandt" if a.scene in S.TANDT else "mip360"
    names, c2w_w, K0, (h0, w0) = S.colmap_scene(a.root, a.scene)
    folder = S.image_folder(ds, a.scene)
    POOL, TEST = S.split_3dgs(len(names))
    floor = S.span_floor(a.root, ds, 0, "sequence")
    seq_scale = 1.0 / max(round(span(c2w_w), 4), floor)
    views = []
    for i, n in enumerate(names):
        im = read_rgb(f"{a.root}/{a.scene}/{folder}/{n}")
        h, w = im.shape[:2]
        views.append({"img": im, "K": K0[i] * np.array([w / w0, h / h0, w / w0, h / h0])})
    views = resize_short(views, a.short or None)
    IMG = np.stack([v["img"] for v in views]).transpose(0, 3, 1, 2)
    KK = np.stack([v["K"] for v in views]).astype(np.float32)
    C = c2w_w.astype(np.float64)[:, :3, 3]
    print(f"{a.scene}: {len(POOL)} candidates, {len(TEST)} held out, {IMG.shape[-1]}x{IMG.shape[-2]}, seq_scale {seq_scale:.4f}", flush=True)

    model = model_from_args(a)

    def solve(I):
        rel = to_model_frame(np.concatenate([c2w_w[I], c2w_w]), seq_scale)    # first input = identity
        sc = dict(inp_image=IMG[I], inp_c2w=rel[:len(I)], inp_K=KK[I], inp_time=np.zeros(len(I), np.float32))
        return SceneRenderer(model, sc, dev), rel[len(I):]

    def render(r, cam, idx, keys=("std",)):
        idx = list(idx)
        return r.render(cam[idx], KK[idx], np.zeros(len(idx), np.float32), keys=keys)

    def eval_holdout(r, cam, save=None):
        o = render(r, cam, TEST)
        u8 = (o["rgb"] * 255).round().astype(np.uint8)
        pred = torch.from_numpy(u8).permute(0, 3, 1, 2)
        gt = torch.from_numpy(IMG[TEST])
        ps = M.psnr(pred, gt)
        p, g = pred.float().div(255).to(dev), gt.float().div(255).to(dev)
        ss = [float(M.ssim_3dgs(p[i:i + 1], g[i:i + 1])) for i in range(len(p))]
        lp = M.lpips_vgg(p, g)
        if save:
            os.makedirs(save, exist_ok=True)
            np.savez_compressed(f"{save}/holdout.npz", pred=u8, std=o["std"].astype(np.float16), test_idx=np.array(TEST), psnr=np.array(ps))
        return dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), lpips=float(np.mean(lp)))

    # ---- start: greedy farthest point over the candidates
    i0, i1 = max(((i, j) for i in POOL for j in POOL if i < j), key=lambda ij: np.linalg.norm(C[ij[0]] - C[ij[1]]))
    START = [i0, i1]
    while len(START) < a.n_start:
        START.append(max((i for i in POOL if i not in START), key=lambda i: min(np.linalg.norm(C[i] - C[s]) for s in START)))
    SAVE = {int(x) for x in a.save_steps.split(",") if x}
    strategies = []
    for s in a.strategies.split(","):
        strategies += [f"random_{sd}" for sd in a.seeds.split(",")] if s == "random" else [s]
    rp = f"{a.out}/results.json"
    R = json.load(open(rp)) if os.path.exists(rp) else {}
    for st in strategies:
        if st in R and len(R[st]["steps"]) == a.n_final - a.n_start + 1:
            continue
        rng = random.Random(int(st.split("_")[1])) if st.startswith("random") else None
        I = list(START)
        rec = {"start": START, "chosen": [], "steps": []}
        t0 = time.time()
        score = None
        while True:
            r, cam = solve(I)
            e = eval_holdout(r, cam, f"{a.out}/renders/{st}/n{len(I):02d}" if len(I) in SAVE else None)
            rec["steps"].append(dict(n=len(I), score=score, **e))
            sc_txt = f" score {score:.4f}" if score is not None else ""
            print(f"{st:9s} n={len(I):2d}{(' +' + str(I[-1])) if len(I) > a.n_start else ''}{sc_txt}  PSNR {e['psnr']:.2f} SSIM {e['ssim']:.3f} "
                  f"LPIPS {e['lpips']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if len(I) >= a.n_final:
                break
            cands = [i for i in POOL if i not in I]
            score = None
            if st.startswith("random"):
                nxt = rng.choice(cands)
            elif st == "farthest":
                nxt = max(cands, key=lambda i: min(np.linalg.norm(C[i] - C[s]) for s in I))
            else:
                key = "std" if st == "unc" else "log_var_epi"
                o = render(r, cam, cands, keys=(key,))
                u = o[key] if st == "unc" else np.exp(o[key] / 2)
                m = u.reshape(len(cands), -1).mean(1)
                nxt, score = int(cands[int(np.argmax(m))]), float(m.max())
            I.append(int(nxt))
            rec["chosen"].append(int(nxt))
            del r
            torch.cuda.empty_cache()
        rec["secs"] = round(time.time() - t0, 1)
        R[st] = rec
        R["meta"] = dict(scene=a.scene, short=a.short, candidates=len(POOL), held_out=len(TEST), n_start=a.n_start, n_final=a.n_final)
        json.dump(R, open(rp, "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for st, rec in R.items():
        if st == "meta":
            continue
        n = [s["n"] for s in rec["steps"]]
        for k, key in enumerate(("psnr", "ssim", "lpips")):
            ax[k].plot(n, [s[key] for s in rec["steps"]], marker="o", ms=3, label=st, ls="--" if st.startswith("random") else "-")
    for k, key in enumerate(("PSNR (dB)", "SSIM", "LPIPS")):
        ax[k].set_xlabel("number of input views")
        ax[k].set_ylabel(key)
        ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{a.out}/curves.png", dpi=150)
    print("written", rp)


if __name__ == "__main__":
    main()
