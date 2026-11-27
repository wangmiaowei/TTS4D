"""Stereo4D (Jin et al., CVPR 2025) test clips: left-eye rectified perspective videos, official camera annotations, and the
right-eye perspective videos.

Expected layout (``--root``):
  annotations_camera/test/<clip>.npz                  official annotation (camera2world [T, 3, 4], timestamps [T] in us)
  mp4/test/<clip>-left_rectified.mp4                  512 x 512 left-eye perspective video (one frame per annotation)
  mp4_right/test/<clip>-right_rectified.mp4           512 x 512 right-eye perspective video (a third-party conversion of
                                                      the release; used for the novel-view targets)
Cameras: camera2world used directly (OpenCV); 60 degree horizontal field of view, principal point at the image centre.
Right eye: the left camera translated by a fixed stereo baseline of 0.063 along its own x axis.
Time = timestamp / 1e6 s.

Protocol (main paper, native 512 x 512, metric protocol ``default``), per clip with T frames:
  inputs   = left eye, ``common.interp_split(T, N)`` (rule A: uniform inputs, 8 targets at input-gap midpoints; rule B for
             short clips: 8 uniform targets, inputs uniform over the rest, so ``n_in`` may be below N);
  targets  ``novel``  (main table): the right eye at the 8 held-out instants (group ``right_heldout_time``) and at 8 input
                      instants ``uniform(inputs, 8)`` (group ``right_input_time``);
           ``interp`` (supplementary): the left eye at the 8 held-out instants.
Clip list: the sorted names of the test annotations (8,162 clips); a uniform subset is every 16th clip of that list.
Scene unit: seq_scale = 1 / max(largest distance of a left camera from the first one (rounded to 1e-4), baseline 0.063).
"""
import os

import numpy as np

from .common import assemble, interp_split, uniform

FOV_DEG = 60.0
BASELINE = 0.063
N_TARGETS = 8


def clips(root, split="test"):
    d = f"{root}/annotations_camera/{split}"
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npz"))


def _read_frames(path, wanted):
    """Decode the wanted frame indices of a video (sequential decode, exact frames)."""
    import cv2
    wanted = sorted(set(int(i) for i in wanted))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    out, cur = {}, 0
    for w in wanted:
        while cur < w:
            cap.grab()
            cur += 1
        ok, fr = cap.read()
        cur += 1
        if not ok:
            raise ValueError(f"cannot decode frame {w} of {path}")
        out[w] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def build(root, clip, n_in, protocol="novel", split="test", baseline=BASELINE):
    import cv2
    z = np.load(f"{root}/annotations_camera/{split}/{clip}.npz")
    T = len(z["camera2world"])
    c2w = np.tile(np.eye(4), (T, 1, 1))
    c2w[:, :3, :4] = z["camera2world"]
    c2w = c2w.astype(np.float32)
    ts = np.asarray(z["timestamps"], np.float64) / 1e6
    ts = (ts - ts.min()).astype(np.float32)
    left = f"{root}/mp4/{split}/{clip}-left_rectified.mp4"
    right = f"{root}/mp4_right/{split}/{clip}-right_rectified.mp4"
    cap = cv2.VideoCapture(left)
    W, H, NF = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if NF != T:
        raise ValueError(f"{clip}: {NF} video frames vs {T} poses")
    f = (W / 2.0) / np.tan(np.radians(FOV_DEG / 2.0))
    K = np.array([f, f, W / 2.0, H / 2.0], np.float32).astype(np.float64)
    off = np.eye(4)
    off[0, 3] = baseline
    c2w_right = (c2w.astype(np.float64) @ off).astype(np.float32)
    C = c2w.astype(np.float64)[:, :3, 3]
    seq_scale = 1.0 / max(round(float(np.linalg.norm(C - C[0], axis=1).max()), 4), baseline)    # span rounded to 1e-4, as in our evaluation

    I, Q, rule = interp_split(T, n_in, N_TARGETS)
    has_right = os.path.exists(right)
    if protocol == "novel":
        if not has_right:
            raise FileNotFoundError(f"{clip}: no right-eye video {right}")
        Iv = uniform(I, N_TARGETS)
        tl = [(t, 0, None) for t in I]
        tr = [(t, 1, "right_heldout_time") for t in Q] + [(t, 1, "right_input_time") for t in Iv]
    elif protocol == "interp":
        tl = [(t, 0, None) for t in I] + [(t, 0, "interp") for t in Q]
        tr = []
    else:
        raise ValueError(protocol)
    fl = _read_frames(left, [t for t, _, _ in tl])
    fr = _read_frames(right, [t for t, _, _ in tr]) if tr else {}
    views = []
    for t, c, g in tl + tr:
        v = {"img": (fl if c == 0 else fr)[t], "c2w_world": c2w[t] if c == 0 else c2w_right[t], "K": K, "t": float(ts[t]), "frame": t, "cam": c}
        if g:
            v["group"] = g
        views.append(v)
    meta = dict(dataset="stereo4d", scene=clip, protocol=protocol, metric="default", n_in_requested=int(n_in), T=T,
                rule=f"left-eye inputs, time-interpolation rule {rule}" + ("; right eye at the 8 held-out and 8 input instants" if protocol == "novel" else ""))
    return assemble(views, len(I), seq_scale, False, meta)
