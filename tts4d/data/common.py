"""Shared rules of the evaluation protocols and the scene .npz format.

Scene .npz (read by ``tts4d.inference.load_scene``):
    inp_image   uint8 [V, 3, H, W]   input views (RGB)
    inp_c2w     float32 [V, 4, 4]    OpenCV camera-to-world in the MODEL FRAME: first input camera = identity,
                                     translations multiplied by ``seq_scale``
    inp_K       float32 [V, 4]       fx, fy, cx, cy in pixels of the stored image (pixel-corner convention: an image resized
                                     by s has K scaled by s)
    inp_time    float32 [V]          seconds (``load_scene`` subtracts the earliest input time; 0 for static scenes)
    tgt_*       the same for the held-out target views, plus optional tgt_mask uint8 [Q, 1, H, W] (DyCheck covisibility)
    inp_c2w_world / tgt_c2w_world    the dataset's own cameras (OpenCV, dataset units), for other normalisations
    inp_frame / tgt_frame, inp_cam / tgt_cam    dataset frame (or instant) ids and camera ids
    tgt_group   str [Q]              target groups (e.g. Stereo4D right eye at held-out / input instants)
    meta        str                  JSON: dataset, scene, protocol, metric protocol, n_in (requested / real), seq_scale, rule
H and W are those of the protocol; they need not be multiples of 8 (NVIDIA is 960 x 540): ``tts4d.runner`` pads them.
"""
import json
import os

import numpy as np


def uniform(S, n):
    """The input rule of every protocol: ``S[round(linspace(0, len(S) - 1, n))]`` (numpy rounding, half to even); all of S
    when it has at most n elements."""
    S = list(S)
    if n >= len(S):
        return S
    return [S[int(i)] for i in np.round(np.linspace(0, len(S) - 1, n)).astype(np.int64)]


def interp_split(T, N, nq=8):
    """Time interpolation on one camera with frames 0..T-1 -> (inputs, targets, rule).
    Rule A (T >= N + nq and at least nq input gaps of >= 2 frames): inputs = round(linspace(0, T - 1, N)); among the gaps g
      (inputs I[g] -> I[g + 1]) with I[g + 1] - I[g] >= 2 take G_k = eligible[int((k + .5) * len(eligible) / nq)];
      target_k = (I[G_k] + I[G_k + 1]) // 2, the gap midpoint (k = 0..nq-1).
    Rule B (otherwise): targets Q_k = round((k + .5) * (T - 1) / nq), inputs = uniform(frames not in Q, N)."""
    if T >= N + nq:
        I = uniform(range(T), N)
        el = [g for g in range(len(I) - 1) if I[g + 1] - I[g] >= 2]
        if len(el) >= nq:
            G = [el[int((k + 0.5) * len(el) / nq)] for k in range(nq)]
            Q = [(I[g] + I[g + 1]) // 2 for g in G]
            assert len(set(Q)) == nq and not set(Q) & set(I)
            return I, Q, "A"
    Q = sorted({int(round((k + 0.5) * (T - 1) / nq)) for k in range(nq)})
    rest = [t for t in range(T) if t not in set(Q)]
    return uniform(rest, N), Q, "B"


def span(c2w, ref=0):
    """Largest distance of a camera centre from camera ``ref`` (dataset units)."""
    C = np.asarray(c2w, np.float64)[:, :3, 3]
    return float(np.linalg.norm(C - C[ref], axis=1).max())


def to_model_frame(c2w_world, seq_scale):
    """inv(c2w[0]) @ c2w (the first camera of the list = identity), translations x seq_scale.  float64 maths, float32 out."""
    c = np.asarray(c2w_world, np.float64)
    c = np.linalg.inv(c[0])[None] @ c
    c[:, :3, 3] *= float(seq_scale)
    return c.astype(np.float32)


def read_rgb(path):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def resize_short(views, short):
    """Every view to short side ``short`` (aspect kept, long side rounded), INTER_AREA when shrinking, K scaled per axis by
    the real size ratio.  views: list of dicts with 'img' (H, W, 3) and 'K' (4,)."""
    import cv2
    if not short:
        return views
    for v in views:
        h, w = v["img"].shape[:2]
        nh, nw = (short, int(round(w * short / h))) if h <= w else (int(round(h * short / w)), short)
        if (nh, nw) != (h, w):
            v["img"] = cv2.resize(v["img"], (nw, nh), interpolation=cv2.INTER_AREA if nh < h else cv2.INTER_LINEAR)
            v["K"] = np.asarray(v["K"], np.float64) * np.array([nw / w, nh / h, nw / w, nh / h])
    return views


def assemble(views, n_in, seq_scale, static, meta):
    """views: list of dicts {img (H, W, 3) uint8, c2w_world (4, 4), K (4,), t (s), frame, cam, [group], [mask]}, inputs
    first.  -> scene dict in the .npz format above (first input = identity, translations x seq_scale, times absolute)."""
    c2w_w = np.stack([np.asarray(v["c2w_world"]) for v in views])
    c2w = to_model_frame(c2w_w, seq_scale)
    t = np.array([v["t"] for v in views], np.float64)
    if static:
        t = np.zeros_like(t)
    inp, tgt = views[:n_in], views[n_in:]
    img = lambda vv: np.stack([v["img"] for v in vv]).transpose(0, 3, 1, 2).copy() if vv else np.zeros((0, 3, 1, 1), np.uint8)   # noqa: E731
    sc = dict(inp_image=img(inp), inp_c2w=c2w[:n_in], inp_K=np.stack([v["K"] for v in inp]).astype(np.float32),
              inp_time=t[:n_in].astype(np.float32), inp_c2w_world=c2w_w[:n_in].astype(np.float32),
              inp_frame=np.array([v["frame"] for v in inp], np.int64), inp_cam=np.array([v["cam"] for v in inp], np.int64))
    if tgt:
        sc.update(tgt_image=img(tgt), tgt_c2w=c2w[n_in:], tgt_K=np.stack([v["K"] for v in tgt]).astype(np.float32),
                  tgt_time=t[n_in:].astype(np.float32), tgt_c2w_world=c2w_w[n_in:].astype(np.float32),
                  tgt_frame=np.array([v["frame"] for v in tgt], np.int64), tgt_cam=np.array([v["cam"] for v in tgt], np.int64),
                  tgt_group=np.array([v.get("group", "all") for v in tgt]))
        if any("mask" in v for v in tgt):
            sc["tgt_mask"] = np.stack([v["mask"] for v in tgt])[:, None].astype(np.uint8)
    meta = dict(meta, n_in=int(n_in), n_tgt=len(tgt), hw=list(views[0]["img"].shape[:2]), seq_scale=float(seq_scale), static_time=bool(static))
    sc["meta"] = np.array(json.dumps(meta))
    return sc


def save_scene(path, sc, compress=True):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    (np.savez_compressed if compress else np.savez)(tmp, **sc)
    os.replace(tmp, path)


def scene_meta(sc):
    m = sc.get("meta")
    return json.loads(str(m)) if m is not None else {}
