#!/usr/bin/env python3
"""Score the model on a prepared benchmark (scripts/prepare_benchmark.py) and write the paper's metrics table.

For every scene .npz of <data>/N<n>/: solve the memory from the inputs, render every target, and score it with the
metric code of the benchmark's protocol (``meta.metric``: default / dynamicnerf / dycheck / 3dgs, see tts4d/metrics.py),
plus the uncertainty metrics (Pearson correlation and AUSE of the predicted std exp(U/2) against the L1 and DSSIM errors;
DyCheck novel views: inside the covisibility masks).  Aggregation: per target image -> mean over the targets of a scene ->
mean over scenes.

Outputs in --out: rows_N<n>.jsonl (one row per scene, with per-target values; the run resumes from it), table.md and
table.json (one line per N: PSNR / SSIM / LPIPS, Pearson L1 / DSSIM, AUSE L1 / DSSIM).

Example:
  python scripts/eval_benchmark.py --data data/bench/dycheck --out results/dycheck --n_in 16 32 64
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tts4d import metrics as M  # noqa: E402
from tts4d.data.common import scene_meta  # noqa: E402
from tts4d.inference import load_scene  # noqa: E402
from tts4d.runner import SceneRenderer, add_model_args, model_from_args  # noqa: E402

IMG_KEYS = ("psnr", "ssim", "lpips")
UNC_KEYS = ("pearson_L1", "pearson_DSSIM", "AUSE_L1", "AUSE_DSSIM")


def score_scene(model, path, device, args):
    sc = load_scene(path)
    meta = scene_meta(sc)
    t0 = time.time()
    r = SceneRenderer(model, sc, device, pad_multiple=args.pad)
    t_solve = time.time() - t0
    nq = len(sc["tgt_c2w"])
    mask = sc.get("tgt_mask")
    metric = args.metric or meta.get("metric", "default")
    per = {k: [] for k in IMG_KEYS + (() if args.no_uncertainty else UNC_KEYS)}
    for s in range(0, nq, args.chunk):
        sl = slice(s, s + args.chunk)
        o = r.render(sc["tgt_c2w"][sl], sc["tgt_K"][sl], sc["tgt_time"][sl], batch=args.query_batch, keys=("std",))
        pred = torch.from_numpy(o["rgb"]).permute(0, 3, 1, 2)
        gt = torch.from_numpy(sc["tgt_image"][sl])
        mk = mask[sl] if (mask is not None and metric == "dycheck") else None
        m = M.image_metrics(pred, gt, metric, mk)
        for k in IMG_KEYS:
            per[k] += m[k]
        if not args.no_uncertainty:
            u = M.uncertainty_metrics(o["std"], pred, gt, mk)
            for k in UNC_KEYS:
                per[k] += u[k]
    groups = [str(g) for g in sc["tgt_group"]] if "tgt_group" in sc else ["all"] * nq
    mean = {k: float(np.nanmean(v)) for k, v in per.items() if v}
    by_group = {}
    for g in sorted(set(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        # (a target with fewer than 2 covisible pixels has no uncertainty values: those lists are shorter and left out here)
        by_group[g] = {k: float(np.nanmean([v[i] for i in idx])) for k, v in per.items() if len(v) == nq} | {"n": len(idx)}
    return dict(scene=meta.get("scene", os.path.basename(path)[:-4]), dataset=meta.get("dataset"), protocol=meta.get("protocol"), metric=metric,
                n_in=int(len(sc["inp_c2w"])), n_in_requested=meta.get("n_in_requested"), n_tgt=nq, hw=list(sc["tgt_image"].shape[-2:]),
                mean=mean, by_group=by_group, per_target=per, t_solve=round(t_solve, 2), secs=round(time.time() - t0, 1))


def table(rows_by_n, title):
    L = [f"# {title}", "", "Mean over scenes of the per-scene mean over targets.  Uncertainty: predicted std exp(U/2) vs the L1 / DSSIM error.", "",
         "| Vin | scenes | PSNR | SSIM | LPIPS | Pearson L1 | Pearson DSSIM | AUSE L1 | AUSE DSSIM |", "|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for n, rows in sorted(rows_by_n.items()):
        if not rows:
            continue
        keys = rows[0]["mean"].keys()
        mean = {k: float(np.mean([r["mean"][k] for r in rows if k in r["mean"]])) for k in keys}
        out[n] = dict(mean, scenes=len(rows))
        f = lambda k, d: f"{mean[k]:.{d}f}" if k in mean else "-"   # noqa: E731
        L.append(f"| {n} | {len(rows)} | {f('psnr', 2)} | {f('ssim', 3)} | {f('lpips', 3)} | {f('pearson_L1', 3)} | {f('pearson_DSSIM', 3)} | "
                 f"{f('AUSE_L1', 3)} | {f('AUSE_DSSIM', 3)} |")
    return "\n".join(L) + "\n", out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="prepared benchmark folder, <out>/<benchmark> of prepare_benchmark.py (holds N<n>/*.npz)")
    ap.add_argument("--out", required=True, help="results folder")
    ap.add_argument("--n_in", type=int, nargs="+", default=None, help="numbers of input views to score (default: every N<n> folder)")
    ap.add_argument("--scenes", default="", help="comma-separated subset of scene names")
    ap.add_argument("--limit", type=int, default=0, help="only the first L scenes of each N")
    ap.add_argument("--metric", default="", choices=["", "default", "dynamicnerf", "dycheck", "3dgs"], help="override the scene's metric protocol")
    ap.add_argument("--no_uncertainty", action="store_true", help="skip the uncertainty metrics")
    ap.add_argument("--pad", type=int, default=32, help="pad the images to a multiple of this (32 in all our evaluations)")
    ap.add_argument("--query_batch", type=int, default=None, help="target views per forward (default: by image size)")
    ap.add_argument("--chunk", type=int, default=64, help="targets rendered and scored at a time (memory of the host)")
    add_model_args(ap)
    a = ap.parse_args()
    ns = a.n_in or sorted(int(os.path.basename(d)[1:]) for d in glob.glob(os.path.join(a.data, "N*")) if os.path.isdir(d))
    os.makedirs(a.out, exist_ok=True)
    model = None
    rows_by_n = {}
    for n in ns:
        files = sorted(glob.glob(os.path.join(a.data, f"N{n}", "*.npz")))
        if a.scenes:
            keep = set(a.scenes.split(","))
            files = [f for f in files if os.path.basename(f)[:-4] in keep]
        if a.limit:
            files = files[:a.limit]
        rp = os.path.join(a.out, f"rows_N{n}.jsonl")
        done = {}
        if os.path.exists(rp):
            for ln in open(rp):
                r = json.loads(ln)
                done[r["file"]] = r
        with open(rp, "a") as fout:
            for k, f in enumerate(files):
                key = os.path.basename(f)
                if key in done:
                    continue
                if model is None:
                    model = model_from_args(a)
                try:
                    row = score_scene(model, f, a.device, a)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  [N{n} {k + 1}/{len(files)}] {key}: out of GPU memory, skipped", flush=True)
                    continue
                row["file"] = key
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                done[key] = row
                m = row["mean"]
                print(f"  [N{n} {k + 1}/{len(files)}] {row['scene'][:28]:28s} n_in {row['n_in']:3d} targets {row['n_tgt']:3d}: "
                      + " ".join(f"{kk} {v:.4f}" for kk, v in m.items()) + f" ({row['secs']}s)", flush=True)
        rows_by_n[n] = [done[os.path.basename(f)] for f in files if os.path.basename(f) in done]
    md, js = table(rows_by_n, f"{os.path.basename(os.path.normpath(a.data))}")
    open(os.path.join(a.out, "table.md"), "w").write(md)
    json.dump(js, open(os.path.join(a.out, "table.json"), "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
