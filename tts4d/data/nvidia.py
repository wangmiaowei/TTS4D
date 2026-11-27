"""NVIDIA Dynamic Scenes (Yoon et al., CVPR 2020), released layout ``<root>/<scene>/{multiview_GT/<instant>/camXX.jpg,
calibration/camXX/{intrinsic,extrinsic}.txt}``.

Only the 9 scenes that ship ``multiview_GT`` (12 instants x 12 cameras, 960 x 540) can be scored: Balloon1, Balloon2,
DynamicFace, Jumping, Playground, Skating, Teadybear, Truck, Umbrella.
Cameras: ``extrinsic.txt`` = [C (camera centre); R (3 x 3, world-to-camera, OpenCV axes)], so c2w = [R^T | C];
``intrinsic.txt`` = K of the 960 x 540 images.  Time = instant / 30 s (the capture rate is not documented; 30 fps assumed).
Cameras are numbered 0..11 in the order cam01..cam12; camera 0 (cam01) is the test camera of DynamicNeRF.

Protocols (native 960 x 540, metric protocol ``dynamicnerf``):
  ``ext``    (main paper): targets = camera 0 at the 12 instants (DynamicNeRF's test views, never an input); inputs =
             ``uniform`` over the 132 images of cameras 1..11, ordered (instant, camera).
  ``bullet`` (supplementary): one item per (scene, held-out instant h): targets = the 12 cameras at instant h; inputs =
             ``uniform`` over the 132 images of the other 11 instants (all cameras), ordered (instant, camera).
Scene unit: seq_scale = 1 / (largest distance between two rig cameras).
"""
import glob
import os

import numpy as np

from .common import assemble, read_rgb, uniform

FPS = 30.0


def scenes(root):
    return sorted(s for s in os.listdir(root) if not s.startswith("_") and glob.glob(f"{root}/{s}/multiview_GT/*/*.jpg"))


def _index(root, scene):
    d = f"{root}/{scene}"
    gt = sorted(glob.glob(f"{d}/multiview_GT/*/*.jpg"))
    inst = sorted({os.path.basename(os.path.dirname(p)) for p in gt})
    cams = sorted({os.path.basename(p)[:-4] for p in gt})
    assert len(inst) == 12 and len(cams) == 12 and len(gt) == 144, (scene, len(inst), len(cams), len(gt))
    c2w, K = [], []
    for c in cams:
        k = np.loadtxt(f"{d}/calibration/{c}/intrinsic.txt").reshape(3, 3)
        rows = np.loadtxt(f"{d}/calibration/{c}/extrinsic.txt").reshape(4, 3)
        m = np.eye(4)
        m[:3, :3] = rows[1:].T
        m[:3, 3] = rows[0]
        c2w.append(m)
        K.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    paths = {(t, c): f"{d}/multiview_GT/{inst[t]}/{cams[c]}.jpg" for t in range(12) for c in range(12)}
    return np.stack(c2w).astype(np.float32), np.array(K, np.float32), paths     # float32, as in our evaluation


def items(root, protocol="ext"):
    """(scene, held-out instant or None) of every item of the protocol."""
    sc = scenes(root)
    return [(s, None) for s in sc] if protocol == "ext" else [(s, h) for s in sc for h in range(12)]


def build(root, scene, n_in, protocol="ext", held_instant=None):
    c2w, K, paths = _index(root, scene)
    C = c2w.astype(np.float64)[:, :3, 3]
    seq_scale = 1.0 / max(float(np.linalg.norm(C[:, None] - C[None], axis=-1).max()), 1e-6)

    def view(t, c, group=None):
        im = read_rgb(paths[(t, c)])
        assert im.shape[:2] == (540, 960), im.shape
        v = {"img": im, "c2w_world": c2w[c], "K": K[c].astype(np.float64), "t": t / FPS, "frame": t, "cam": c}
        if group:
            v["group"] = group
        return v

    if protocol == "ext":
        pool = [(t, c) for t in range(12) for c in range(1, 12)]
        I = uniform(range(len(pool)), n_in)
        views = [view(*pool[i]) for i in I] + [view(t, 0, "test_cam") for t in range(12)]
        meta = dict(rule="inputs uniform over the 132 images of cameras 1-11 ordered (instant, camera); targets camera 0 at the 12 instants")
        name = scene
    elif protocol == "bullet":
        h = int(held_instant)
        pool = [(t, c) for t in range(12) if t != h for c in range(12)]
        I = uniform(range(len(pool)), n_in)
        views = [view(*pool[i]) for i in I] + [view(h, c, "bullet") for c in range(12)]
        meta = dict(rule=f"bullet time: targets = the 12 cameras at instant {h}; inputs uniform over the other 11 instants", held_instant=h)
        name = f"{scene}_t{h:02d}"
    else:
        raise ValueError(protocol)
    meta.update(dataset="nvidia", scene=name, protocol=protocol, metric="dynamicnerf", n_in_requested=int(n_in))
    return assemble(views, len(I), seq_scale, False, meta)
