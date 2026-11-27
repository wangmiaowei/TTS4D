#!/usr/bin/env python3
"""Convert a benchmark's released data into scene .npz files with the input / target split of the paper's protocol.

Writes <out>/<benchmark>/N<n>/<scene>.npz for every requested number of input views (the inputs differ with N), each
holding the inputs, the held-out targets, the cameras in the model frame and a JSON ``meta`` (protocol, metric, ...).
Score them with scripts/eval_benchmark.py.

Benchmarks (``--root`` is the dataset's release folder; see the docstrings of tts4d/data/*.py for the exact rules):
  stereo4d          Stereo4D test clips, left-eye inputs, right eye scored (main paper)      --root has annotations_camera/, mp4/, mp4_right/
  stereo4d_interp   Stereo4D, left eye at held-out instants (supplementary)
  dycheck           DyCheck iPhone, time interpolation on the training video (main paper)   --root = .../iphone
  dycheck_novel     DyCheck, the two test cameras, covisibility-masked metrics (supplementary)
  nvidia            NVIDIA Dynamic Scenes, camera 1 at the 12 instants (main paper)         --root has Balloon1/, ...
  nvidia_bullet     NVIDIA, all 12 cameras at a held-out instant (supplementary)
  mip360, tandt     Mip-NeRF 360 / Tanks and Temples (3DGS), 3DGS hold-out (supplementary) --root has <scene>/sparse/0
  dl3dv             DL3DV-140 (supplementary); also needs --dl3dv_benchmark, --dl3dv_list, --dl3dv_kmeans

Examples:
  python scripts/prepare_benchmark.py --benchmark dycheck --root data/raw/DyCheck/iphone --out data/bench --n_in 16 32 64
  python scripts/prepare_benchmark.py --benchmark stereo4d --root data/raw/Stereo4D --out data/bench --stride 16
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tts4d.data import common  # noqa: E402

BENCHMARKS = ["stereo4d", "stereo4d_interp", "dycheck", "dycheck_novel", "nvidia", "nvidia_bullet", "mip360", "tandt", "dl3dv"]


def jobs(a):
    """[(scene name, builder(n) -> scene dict)]"""
    b = a.benchmark
    if b.startswith("stereo4d"):
        from tts4d.data import stereo4d as M
        names = M.clips(a.root)[::a.stride]
        proto = "interp" if b == "stereo4d_interp" else "novel"
        return [(c, (lambda n, c=c: M.build(a.root, c, n, proto))) for c in names]
    if b.startswith("dycheck"):
        from tts4d.data import dycheck as M
        proto = "novel" if b == "dycheck_novel" else "interp"
        return [(s, (lambda n, s=s: M.build(a.root, s, n, proto))) for s in M.sequences(a.root, proto)]
    if b.startswith("nvidia"):
        from tts4d.data import nvidia as M
        proto = "bullet" if b == "nvidia_bullet" else "ext"
        return [(s if h is None else f"{s}_t{h:02d}", (lambda n, s=s, h=h: M.build(a.root, s, n, proto, h))) for s, h in M.items(a.root, proto)]
    if b in ("mip360", "tandt"):
        from tts4d.data import static as M
        floors = {}

        def mk(s):
            def f(n):
                if n not in floors:
                    floors[n] = M.span_floor(a.root, b, n, a.unit)
                return M.build_3dgs(a.root, b, s, n, floors[n], a.unit, a.short or None)
            return f
        return [(s, mk(s)) for s in M.scenes_3dgs(a.root, b)]
    if b == "dl3dv":
        from tts4d.data import static as M
        assert a.dl3dv_benchmark and a.dl3dv_list and a.dl3dv_kmeans, "dl3dv needs --dl3dv_benchmark, --dl3dv_list and --dl3dv_kmeans"
        return [(h, (lambda n, h=h: M.build_dl3dv(a.dl3dv_benchmark, a.root, a.dl3dv_kmeans, h, n))) for h in M.dl3dv_scenes(a.dl3dv_list)]
    raise ValueError(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--root", required=True, help="the dataset's release folder (for dl3dv: the DL3DV-10K images, zips or <hash>/images_4 folders)")
    ap.add_argument("--out", required=True, help="output folder; scenes go to <out>/<benchmark>/N<n>/<scene>.npz")
    ap.add_argument("--n_in", type=int, nargs="+", default=[16, 32, 64], help="numbers of input views (default 16 32 64; 0 = all, static sets only)")
    ap.add_argument("--scenes", default="", help="comma-separated subset of scene / clip names (default: all)")
    ap.add_argument("--stride", type=int, default=1, help="stereo4d: every k-th clip of the sorted test list (16 -> the 511-clip subset)")
    ap.add_argument("--limit", type=int, default=0, help="only the first L scenes")
    ap.add_argument("--unit", default="input", choices=["input", "sequence"], help="mip360 / tandt scene unit (see tts4d/data/static.py)")
    ap.add_argument("--short", type=int, default=0, help="mip360 / tandt: resize to this short side (default: 3DGS evaluation resolution)")
    ap.add_argument("--dl3dv_benchmark", default="", help="DL3DV-Benchmark folder with <hash>/nerfstudio/transforms.json")
    ap.add_argument("--dl3dv_list", default="", help="MVP data/dl3dv_benchmark.txt")
    ap.add_argument("--dl3dv_kmeans", default="", help="MVP data/dl3dv_fold_8_kmeans_input_idx.json")
    ap.add_argument("--no_compress", action="store_true", help="write uncompressed .npz (faster, larger)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    js = jobs(a)
    if a.scenes:
        keep = set(a.scenes.split(","))
        js = [j for j in js if j[0] in keep]
    if a.limit:
        js = js[:a.limit]
    print(f"{a.benchmark}: {len(js)} scenes x N in {a.n_in}", flush=True)
    for n in a.n_in:
        od = os.path.join(a.out, a.benchmark, f"N{n}")
        for k, (name, build) in enumerate(js):
            path = os.path.join(od, f"{name}.npz")
            if os.path.exists(path) and not a.overwrite:
                continue
            t0 = time.time()
            try:
                sc = build(n)
            except (ValueError, FileNotFoundError) as e:
                print(f"  [N{n} {k + 1}/{len(js)}] {name}: skipped ({e})", flush=True)
                continue
            common.save_scene(path, sc, compress=not a.no_compress)
            m = common.scene_meta(sc)
            print(f"  [N{n} {k + 1}/{len(js)}] {name}: {m['n_in']} inputs, {m['n_tgt']} targets, {m['hw'][1]}x{m['hw'][0]} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
