"""DyCheck iPhone (Gao et al., NeurIPS 2022), released layout ``<root>/<sequence>/{camera, rgb/{1x,2x}, splits, covisible, ...}``.

Cameras: ``camera/<id>.json`` (nerfies format): orientation = world-to-camera rotation R (OpenCV axes), position = camera
centre C, so c2w = [R^T | C]; fx = focal_length, fy = focal_length * pixel_aspect_ratio, (cx, cy) = principal_point, in
pixels of ``rgb/1x``.  Images: the released ``rgb/2x`` frames (``extra.json`` factor 2: 360 x 480, the resolution of the
official covisibility masks), intrinsics divided by 2.  Time = time id / fps (``extra.json``).

Protocols (inputs and targets of one sequence):
  ``interp`` (main paper, all 14 sequences): time interpolation on the training video (camera 0 = ``splits/train.json``):
             ``common.interp_split`` over its frames, 8 targets between inputs; metric protocol ``default``.
  ``novel``  (supplementary, the 7 sequences with test cameras): inputs uniform over the training video, targets = every
             frame of ``splits/val.json`` (cameras 1 / 2), scored inside the released covisibility masks (metric ``dycheck``).
Scene unit: seq_scale = 1 / (largest distance of a camera-0 centre from the first camera-0 frame, whole video).
"""
import json
import os

import numpy as np

from .common import assemble, interp_split, uniform

N_INTERP_TARGETS = 8


def sequences(root, protocol="interp"):
    seqs = sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "dataset.json")))
    if protocol == "novel":
        seqs = [s for s in seqs if os.path.exists(f"{root}/{s}/splits/val.json") and json.load(open(f"{root}/{s}/splits/val.json"))["frame_names"]]
    return seqs


def _camera(path):
    cj = json.load(open(path))
    c2w = np.eye(4)
    c2w[:3, :3] = np.array(cj["orientation"], np.float64).T
    c2w[:3, 3] = np.array(cj["position"], np.float64)
    f, ar = float(cj["focal_length"]), float(cj["pixel_aspect_ratio"])
    cx, cy = [float(v) for v in cj["principal_point"]]
    return c2w.astype(np.float32), np.array([f, f * ar, cx, cy], np.float32)      # stored as float32, as in our evaluation


def build(root, seq, n_in, protocol="interp"):
    import cv2
    d = f"{root}/{seq}"
    ex = json.load(open(f"{d}/extra.json"))
    fac, fps = int(ex["factor"]), float(ex["fps"])
    tr = json.load(open(f"{d}/splits/train.json"))
    trt = sorted(zip(tr["time_ids"], tr["camera_ids"], tr["frame_names"]))
    assert all(c == 0 for _, c, _ in trt) and [t for t, _, _ in trt] == list(range(len(trt))), "unexpected train split"
    T = len(trt)
    cam0 = [_camera(f"{d}/camera/{name}.json")[0] for _, _, name in trt]
    C = np.stack(cam0).astype(np.float64)[:, :3, 3]
    seq_scale = 1.0 / max(float(np.linalg.norm(C - C[0], axis=1).max()), 1e-6)

    def view(t, c, name, group=None, split=None):
        c2w, K = _camera(f"{d}/camera/{name}.json")
        im = cv2.imread(f"{d}/rgb/{fac}x/{name}.png", cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(f"{d}/rgb/{fac}x/{name}.png")
        v = {"img": cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB), "c2w_world": c2w, "K": K.astype(np.float64) / fac, "t": float(np.float32(t / fps)),
             "frame": int(t), "cam": int(c)}
        if group:
            v["group"] = group
        if split is not None:
            cv = cv2.imread(f"{d}/covisible/{fac}x/{split}/{name}.png", cv2.IMREAD_UNCHANGED)
            cv = cv[..., 0] if cv.ndim == 3 else cv
            v["mask"] = (cv == 255).astype(np.uint8)
        return v

    if protocol == "interp":
        I, Q, rule = interp_split(T, n_in, N_INTERP_TARGETS)
        views = [view(trt[t][0], 0, trt[t][2]) for t in I] + [view(trt[t][0], 0, trt[t][2], "interp") for t in Q]
        meta = dict(rule=f"time interpolation, rule {rule}", metric="default", T=T)
    elif protocol == "novel":
        va = json.load(open(f"{d}/splits/val.json"))
        I = uniform(range(T), n_in)
        vat = sorted(zip(va["time_ids"], va["camera_ids"], va["frame_names"]))
        views = [view(trt[t][0], 0, trt[t][2]) for t in I] + [view(int(t), int(c), n, f"cam{int(c)}", "val") for t, c, n in vat]
        meta = dict(rule="inputs uniform over the training video; targets = every frame of splits/val.json", metric="dycheck", T=T)
    else:
        raise ValueError(protocol)
    meta.update(dataset="dycheck", scene=seq, protocol=protocol, n_in_requested=int(n_in))
    return assemble(views, len(I), seq_scale, False, meta)
