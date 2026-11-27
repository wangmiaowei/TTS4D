"""Whole-model check of the fast path: encode + render of held-out views with the fast kernels and with the PyTorch
reference, on one or more scenes.

For every scene: held-out PSNR (against the ground-truth targets) of
    fast        fast=True,  bf16 autocast   (the default inference path)
    ref_amp     fast=False, bf16 autocast   (reference implementation, same autocast)
    ref_fp32    fast=False, fp32            (reference implementation, no autocast)
the PSNR between the renderings (fast vs each reference), the agreement of the uncertainty maps, and the wall-clock
time of encode + render (after one warm-up run that compiles the Triton kernels).

    python tests/verify_fast_path.py --weights tts4d.safetensors --scene a.npz b.npz c.npz [--json out.json]

A scene is an .npz as read by ``tts4d.inference.load_scene`` with held-out targets (tgt_*).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tts4d.inference import DEFAULT_CONFIG_PATH, encode, inputs_of, load_model, load_scene, psnr, render_views  # noqa


def run(model, sc, dev, amp, batch):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mem = encode(model, inputs_of(sc, dev), amp=amp)
    torch.cuda.synchronize()
    t1 = time.time()
    r = render_views(model, mem, sc["tgt_c2w"], sc["tgt_K"], sc["tgt_time"], dev, batch=batch, amp=amp)
    torch.cuda.synchronize()
    t2 = time.time()
    return r, dict(encode_s=t1 - t0, render_s=t2 - t1, total_s=t2 - t0, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--scene", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=4, help="query views per render forward")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per variant (after one warm-up)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = "cuda"
    model = load_model(args.weights, args.config, dev)
    variants = (("fast", True, True), ("ref_amp", False, True), ("ref_fp32", False, False))
    rows = []
    for path in args.scene:
        sc = load_scene(path)
        gt = sc["tgt_image"].transpose(0, 2, 3, 1) / 255.0
        nq = len(gt)
        row = dict(scene=os.path.splitext(os.path.basename(path))[0], n_in=len(sc["inp_c2w"]), n_tgt=nq,
                   hw=list(sc["inp_image"].shape[-2:]))
        rend = {}
        for name, fast, amp in variants:
            model.set_fast(fast)
            run(model, sc, dev, amp, args.batch)                          # warm-up (Triton compilation, caches)
            times = []
            for _ in range(args.repeats):
                r, t = run(model, sc, dev, amp, args.batch)
                times.append(t)
            rend[name] = r
            row[f"psnr_{name}"] = float(np.mean([psnr(r["rgb"][i], gt[i]) for i in range(nq)]))
            row[f"time_{name}"] = {k: round(float(np.median([t[k] for t in times])), 3) for k in times[0]}
        for ref in ("ref_amp", "ref_fp32"):
            a, b = rend["fast"], rend[ref]
            row[f"psnr_fast_vs_{ref}"] = float(np.mean([psnr(a["rgb"][i], b["rgb"][i]) for i in range(nq)]))
            row[f"psnr_{ref}_vs_ref_fp32"] = (float(np.mean([psnr(b["rgb"][i], rend["ref_fp32"]["rgb"][i])
                                                             for i in range(nq)])) if ref != "ref_fp32" else None)
            for k in ("log_var", "log_var_epi"):
                row[f"{k}_fast_vs_{ref}"] = dict(
                    corr=float(np.corrcoef(a[k].ravel(), b[k].ravel())[0, 1]),
                    mean_abs_diff=float(np.abs(a[k] - b[k]).mean()))
        row = {k: v for k, v in row.items() if v is not None}
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    model.set_fast(None)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
