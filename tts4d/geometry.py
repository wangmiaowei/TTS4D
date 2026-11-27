"""Cameras: Plucker rays and the normalisation of a scene into the model frame."""
import numpy as np
import torch


def plucker_rays(K, c2w, H, W):
    """Per-pixel Plucker rays (o, d, o x d), [B, V, 9, H, W].  K [B, V, 4] = (fx, fy, cx, cy) in pixels, c2w OpenCV."""
    B, V = K.shape[:2]
    dev = c2w.device
    xs = torch.arange(W, device=dev, dtype=torch.float32).view(1, W).expand(H, W).reshape(-1)
    ys = torch.arange(H, device=dev, dtype=torch.float32).view(H, 1).expand(H, W).reshape(-1)
    K = K.reshape(B * V, 4).float()
    c2w = c2w.reshape(B * V, 4, 4).float()
    x = (xs[None] + 0.5 - K[:, 2:3]) / K[:, 0:1]
    y = (ys[None] + 0.5 - K[:, 3:4]) / K[:, 1:2]
    d = torch.stack([x, y, torch.ones_like(x)], 1)                     # [BV, 3, HW]
    d = torch.bmm(c2w[:, :3, :3], d)
    d = d / d.norm(dim=1, keepdim=True)
    o = c2w[:, :3, 3:4].expand(-1, -1, H * W)
    m = torch.cross(o, d, dim=1)
    return torch.cat([o, d, m], 1).reshape(B, V, 9, H, W)


def normalize_scene(c2w_inputs, c2w_queries=None, span_floor=0.0):
    """Model frame used in training and evaluation: the first input camera is the identity, and translations are
    divided by the largest distance of an INPUT camera from the first one (at least ``span_floor``), so the targets
    never influence the normalisation.  c2w: [V, 4, 4] numpy (OpenCV).  Returns (inputs, queries, scale)."""
    ref = np.linalg.inv(c2w_inputs[0])
    ci = ref[None] @ c2w_inputs
    span = float(np.linalg.norm(ci[:, :3, 3], axis=-1).max())
    scale = 1.0 / max(span, span_floor, 1e-8)
    ci[:, :3, 3] *= scale
    cq = None
    if c2w_queries is not None:
        cq = ref[None] @ c2w_queries
        cq[:, :3, 3] *= scale
    return ci, cq, scale
