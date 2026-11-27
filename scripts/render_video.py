#!/usr/bin/env python3
"""Render a novel space-time camera path of one scene to an mp4, with the predicted uncertainty next to the rendering.

The memory is solved once from the scene's input views; every frame of the path is then a query (camera + time).  Paths
(all in the scene's model frame; see tts4d/data/common.py for the .npz format):
  trajectory  the camera follows a smoothed version of the input trajectory, time follows the inputs (a replay)
  bullet      time frozen at --t (default: middle of the inputs) while the camera moves along the input trajectory
  orbit       the camera circles around the mean input camera (radius --radius x the spread of the input cameras), always
              looking at the focus point of the inputs; time sweeps the inputs' time span (or frozen with --t)
  orbit360    a full circle around the focus point at the mean camera distance (static / object-centric scenes)
  fixed       the camera stays at input view --view while time sweeps the inputs' time span
The right half of the video is the predicted standard deviation exp(U/2) on a log scale (turbo colour map, one fixed
range for the whole video: blue = low, red = high).

Example:
  python scripts/render_video.py --scene data/bench/dycheck/N32/apple.npz --path bullet --out apple_bullet.mp4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402

from tts4d.inference import degrid, load_scene, turbo  # noqa: E402
from tts4d.runner import SceneRenderer, add_model_args, model_from_args  # noqa: E402


# ------------------------------------------------------------------------------------------------ camera paths
def _slerp_path(c2w, key, s):
    from scipy.spatial.transform import Rotation, Slerp
    out = np.tile(np.eye(4), (len(s), 1, 1))
    out[:, :3, :3] = Slerp(key, Rotation.from_matrix(c2w[:, :3, :3]))(s).as_matrix()
    for k in range(3):
        out[:, k, 3] = np.interp(s, key, c2w[:, k, 3])
    return out


def _smooth(c2w, width):
    """Moving average of positions and (re-orthonormalised) rotations over `width` neighbouring cameras."""
    if width <= 1:
        return c2w
    out = c2w.copy()
    n = len(c2w)
    for i in range(n):
        sl = slice(max(0, i - width // 2), min(n, i + width // 2 + 1))
        out[i, :3, 3] = c2w[sl, :3, 3].mean(0)
        u, _, vt = np.linalg.svd(c2w[sl, :3, :3].mean(0))
        out[i, :3, :3] = u @ vt
    return out


def focus_point(c2w):
    """Point closest (least squares) to the optical axes of the cameras; in front of the mean camera if ill-posed."""
    o, d = c2w[:, :3, 3], c2w[:, :3, 2]
    A, b = np.zeros((3, 3)), np.zeros(3)
    for oi, di in zip(o, d):
        P = np.eye(3) - np.outer(di, di)
        A += P
        b += P @ oi
    p = np.linalg.lstsq(A, b, rcond=None)[0]
    mean_o, mean_d = o.mean(0), d.mean(0) / np.linalg.norm(d.mean(0))
    depth = float((p - mean_o) @ mean_d)
    if np.linalg.cond(A) > 1e4 or depth <= 0:
        p = mean_o + mean_d * max(1.0, 2.0 * float(np.linalg.norm(o - mean_o, axis=1).max()))
    return p


def look_at(pos, target, up):
    """OpenCV camera-to-world (x right, y down, z forward) at `pos` looking at `target`, `up` = world up direction."""
    z = target - pos
    z = z / np.linalg.norm(z)
    x = np.cross(-up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, pos
    return m


def make_path(sc, kind, n, t=None, view=None, radius=0.5, smooth=5):
    c2w, times = sc["inp_c2w"].astype(np.float64), sc["inp_time"].astype(np.float64)
    order = np.argsort(times, kind="stable")
    c2w_t, times_t = c2w[order], times[order]
    tmin, tmax = float(times.min()), float(times.max())
    s = np.linspace(0, 1, n)
    key = np.linspace(0, 1, len(c2w_t))
    sweep = tmin + s * (tmax - tmin)
    t_fix = float(np.median(times)) if t is None else float(t)
    up = -c2w[:, :3, 1].mean(0)
    up = up / np.linalg.norm(up)
    if kind == "trajectory":
        cams = _slerp_path(_smooth(c2w_t, smooth), key, s)
        tt = np.interp(s, key, times_t)
    elif kind == "bullet":
        cams = _slerp_path(_smooth(c2w_t, smooth), key, s)
        tt = np.full(n, t_fix)
    elif kind in ("orbit", "orbit360"):
        f = focus_point(c2w)
        o = c2w[:, :3, 3]
        if kind == "orbit":
            centre = o.mean(0)
            fwd = f - centre
            fwd = fwd / np.linalg.norm(fwd)
            a = np.cross(fwd, up)
            a = a / np.linalg.norm(a)
            b = np.cross(a, fwd)
            r = radius * max(float(np.linalg.norm(o - centre, axis=1).mean()), 1e-3)
            ang = 2 * np.pi * s
            pos = centre[None] + r * (np.cos(ang)[:, None] * a[None] + np.sin(ang)[:, None] * b[None])
        else:
            rel = o - f
            h = float((rel @ up).mean())
            rad = float(np.linalg.norm(rel - np.outer(rel @ up, up), axis=1).mean())
            a = rel[0] - (rel[0] @ up) * up
            a = a / np.linalg.norm(a)
            b = np.cross(up, a)
            ang = 2 * np.pi * s
            pos = f[None] + h * up[None] + rad * (np.cos(ang)[:, None] * a[None] + np.sin(ang)[:, None] * b[None])
        cams = np.stack([look_at(p, f, up) for p in pos])
        tt = sweep if t is None else np.full(n, t_fix)
    elif kind == "fixed":
        v = int(np.argsort(times)[len(times) // 2]) if view is None else int(view)
        cams = np.repeat(c2w[v][None], n, 0)
        tt = sweep
    else:
        raise ValueError(kind)
    return cams.astype(np.float32), tt.astype(np.float32)


# ------------------------------------------------------------------------------------------------ video
def write_mp4(path, frames, fps):
    H, W = frames[0].shape[:2]
    frames = [np.pad(f, ((0, H % 2), (0, W % 2), (0, 0)), mode="edge") for f in frames]       # even size for H.264
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1) as w:
            for f in frames:
                w.append_data(f)
        return "imageio/ffmpeg (H.264)"
    except Exception:
        import cv2
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
        for f in frames:
            w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        w.release()
        return "OpenCV (mp4v)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="scene .npz (tts4d/data format; targets are not needed)")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--path", default="bullet", choices=["trajectory", "bullet", "orbit", "orbit360", "fixed"])
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--t", type=float, default=None, help="frozen time in seconds after the first input (bullet / orbit)")
    ap.add_argument("--view", type=int, default=None, help="input view of the fixed camera (default: the middle one in time)")
    ap.add_argument("--radius", type=float, default=0.5, help="orbit radius, in units of the mean distance of the input cameras from their centre")
    ap.add_argument("--max_inputs", type=int, default=0, help="use at most this many inputs, uniform in time (0 = all)")
    ap.add_argument("--degrid", action="store_true", help="apply the training-free grid removal to the renderings")
    ap.add_argument("--no_uncertainty", action="store_true", help="rendering only")
    add_model_args(ap)
    a = ap.parse_args()
    sc = load_scene(a.scene)
    idx = None
    if a.max_inputs and a.max_inputs < len(sc["inp_c2w"]):
        order = np.argsort(sc["inp_time"], kind="stable")
        idx = order[np.round(np.linspace(0, len(order) - 1, a.max_inputs)).astype(int)]
        for k in [k for k in sc if k.startswith("inp_")]:
            sc[k] = sc[k][idx]
    cams, tt = make_path(sc, a.path, a.frames, a.t, a.view, a.radius)
    K = np.repeat(sc["inp_K"][:1], len(cams), 0)
    model = model_from_args(a)
    r = SceneRenderer(model, sc, a.device)
    o = r.render(cams, K, tt, keys=("std",))
    rgb = degrid(o["rgb"]) if a.degrid else o["rgb"]
    frames = (rgb * 255).round().astype(np.uint8)
    if not a.no_uncertainty:
        lu = np.log(np.maximum(o["std"], 1e-6))
        lo, hi = np.percentile(lu, 1), np.percentile(lu, 99.5)
        unc = turbo((lu - lo) / max(hi - lo, 1e-6))
        sep = np.full((frames.shape[1], 4, 3), 255, np.uint8)
        frames = [np.concatenate([f, sep, u], 1) for f, u in zip(frames, unc)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    how = write_mp4(a.out, list(frames), a.fps)
    print(f"{a.out}: {len(cams)} frames ({a.path}, times {tt.min():.2f}..{tt.max():.2f} s), {len(sc['inp_c2w'])} inputs, written with {how}")


if __name__ == "__main__":
    main()
