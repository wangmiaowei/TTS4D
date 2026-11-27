"""Solve one scene and render query views at any image size (used by the evaluation and application scripts).

The model works on images whose sides are multiples of the patch size.  As in all evaluations of the paper, the input
images are replicate-padded symmetrically to a multiple of 32 (the principal point is shifted by the padding), the query
cameras get the same shifted intrinsics, and the renderings and uncertainty maps are cropped back to the original
pixel grid, so nothing is resampled.  Inference runs under bfloat16 autocast; queries only read the solved memory, so
they are rendered in batches.
"""
import numpy as np
import torch
import torch.nn.functional as F

UNC_KEYS = ("log_var", "log_var_epi", "log_var_alea", "std")


def pad_images(images, K, multiple=32):
    """images [V, 3, H, W] (uint8 or float in [0, 1], numpy or torch), K [V, 4] -> (float images [V, 3, Hp, Wp],
    shifted K [V, 4], crop (top, left, H, W))."""
    x = torch.as_tensor(np.asarray(images) if not torch.is_tensor(images) else images)
    x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
    H, W = x.shape[-2:]
    Hp, Wp = -(-H // multiple) * multiple, -(-W // multiple) * multiple
    t, l = (Hp - H) // 2, (Wp - W) // 2
    if (Hp, Wp) != (H, W):
        x = F.pad(x, (l, Wp - W - l, t, Hp - H - t), mode="replicate")
    K = torch.as_tensor(np.asarray(K, np.float32)).clone()
    K[:, 2] += l
    K[:, 3] += t
    return x, K, (t, l, H, W)


def query_batch_for(Hp, Wp):
    """Queries per forward (the evaluation default; any value gives the same result up to bf16 rounding)."""
    return max(1, min(32, int(24 * 153600 / (Hp * Wp))))


class SceneRenderer:
    """Solve the memory once from the input views of a scene, then render any query cameras / times.

        r = SceneRenderer(model, sc, device)              # sc from tts4d.inference.load_scene, or the same keys
        out = r.render(c2w [Q, 4, 4], K [Q, 4], time [Q])  # rgb [Q, H, W, 3] in [0, 1], std / log_var* [Q, H, W]
    """

    def __init__(self, model, sc, device="cuda", inp_idx=None, pad_multiple=32, amp=True):
        self.model, self.device, self.amp = model, torch.device(device), amp
        idx = np.arange(len(sc["inp_c2w"])) if inp_idx is None else np.asarray(inp_idx)
        x, K, self.crop = pad_images(sc["inp_image"][idx], sc["inp_K"][idx], pad_multiple)
        self.Hp, self.Wp = x.shape[-2:]
        self.pad_tl = (self.crop[0], self.crop[1])
        f = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=self.device)[None]   # noqa: E731
        inputs = dict(image=x[None].to(self.device), c2w=f(sc["inp_c2w"][idx]), K=K[None].to(self.device), time=f(sc["inp_time"][idx]))
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=amp):
            self.memory = model.encode(inputs)
        del inputs

    @torch.no_grad()
    def render(self, c2w, K, time, batch=None, keys=UNC_KEYS):
        """Render query views.  Returns numpy: rgb [Q, H, W, 3] in [0, 1] and the requested uncertainty maps [Q, H, W]
        (``std`` = exp(U / 2), ``log_var`` = U, ``log_var_epi`` = U_epi, ``log_var_alea`` = r)."""
        t0, l0, H, W = self.crop
        K = np.asarray(K, np.float32).copy()
        K[:, 2] += l0
        K[:, 3] += t0
        batch = batch or query_batch_for(self.Hp, self.Wp)
        f = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=self.device)[None]   # noqa: E731
        rgb, maps = [], {k: [] for k in keys}
        for s in range(0, len(c2w), batch):
            q = dict(c2w=f(c2w[s:s + batch]), K=f(K[s:s + batch]), time=f(time[s:s + batch]))
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                o = self.model.render(self.memory, q)
            rgb.append(o["rgb"][0].float().clamp(0, 1)[:, :, t0:t0 + H, l0:l0 + W].permute(0, 2, 3, 1).cpu())
            for k in keys:
                maps[k].append(o[k][0, :, 0].float()[:, t0:t0 + H, l0:l0 + W].cpu())
        out = {"rgb": torch.cat(rgb).numpy()}
        out.update({k: torch.cat(v).numpy() for k, v in maps.items()})
        return out


def load_weights(path=None, device="cuda", fast=None):
    """The released model: from a local weights file (.pth / .safetensors) or, if ``path`` is None, downloaded from the
    Hugging Face Hub (cached).  ``fast``: None = automatic kernel choice, False = PyTorch reference path."""
    from .inference import from_pretrained, load_model
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return load_model(path, device=device, fast=fast) if path else from_pretrained(device=device, fast=fast)


def add_model_args(ap):
    """--weights / --reference / --device, shared by the scripts."""
    ap.add_argument("--weights", default=None, help="weights file (.pth / .safetensors); default: download the released weights")
    ap.add_argument("--reference", action="store_true", help="use the PyTorch reference implementation instead of the fast kernels")
    ap.add_argument("--device", default="cuda")


def model_from_args(a):
    return load_weights(a.weights, a.device, fast=False if a.reference else None)
