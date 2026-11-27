"""Loading the model and a scene, rendering, metrics, grid removal and uncertainty colouring."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .model import TTS4D

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(HERE, "..", "configs", "tts4d.json")


def load_model(ckpt_path, config_path=DEFAULT_CONFIG_PATH, device="cuda", fast=None):
    """TTS4D with the released weights (a plain state_dict, EMA weights).

    fast: None (default) = the fast kernels under bf16 autocast on a GPU with Triton, True / False = force them on /
    off (False = the PyTorch reference implementation); see ``TTS4D.set_fast``."""
    cfg = json.load(open(config_path))
    cfg["fast"] = fast
    model = TTS4D(cfg)
    if str(ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(str(ckpt_path))
    else:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}"
    return model.to(device).eval().requires_grad_(False)


HF_REPO = "miaoweiwang/TTS4D"


def from_pretrained(repo_id=HF_REPO, filename="tts4d.safetensors", device="cuda", fast=None):
    """Download the released weights from the Hugging Face Hub (cached) and build the model."""
    from huggingface_hub import hf_hub_download
    return load_model(hf_hub_download(repo_id, filename), device=device, fast=fast)


def load_scene(path):
    """A scene .npz in the model frame (see geometry.normalize_scene):
    inp_image uint8 [V, 3, H, W], inp_c2w [V, 4, 4], inp_K [V, 4] (fx, fy, cx, cy), inp_time [V] (seconds),
    and optionally held-out targets tgt_image / tgt_c2w / tgt_K / tgt_time."""
    z = np.load(path)
    sc = {k: z[k] for k in z.files}
    t0 = float(sc["inp_time"].min())                    # times relative to the first input
    sc["inp_time"] = sc["inp_time"] - t0
    if "tgt_time" in sc:
        sc["tgt_time"] = sc["tgt_time"] - t0
    return sc


def inputs_of(sc, device, idx=None):
    idx = np.arange(len(sc["inp_c2w"])) if idx is None else np.asarray(idx)
    t = lambda a, dt=torch.float32: torch.as_tensor(a[idx], dtype=dt, device=device)[None]   # noqa: E731
    return dict(image=t(sc["inp_image"]).float() / 255.0, c2w=t(sc["inp_c2w"]), K=t(sc["inp_K"]), time=t(sc["inp_time"]))


def queries_of(c2w, K, time, device):
    f = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)[None]   # noqa: E731
    return dict(c2w=f(c2w), K=f(K), time=f(time))


@torch.no_grad()
def render_views(model, memory, c2w, K, time, device, batch=4, amp=True):
    """Render any number of query views from one solved memory, `batch` views per forward.
    Returns numpy dict: rgb [V, H, W, 3] in [0, 1] and the uncertainty maps [V, H, W]."""
    outs = []
    for s in range(0, len(c2w), batch):
        q = queries_of(c2w[s:s + batch], K[s:s + batch], time[s:s + batch], device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
            o = model.render(memory, q)
        outs.append({k: v[0].float().cpu() for k, v in o.items()})
    cat = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    res = {"rgb": cat["rgb"].clamp(0, 1).permute(0, 2, 3, 1).numpy()}
    for k in ("log_var", "log_var_epi", "log_var_alea", "std", "conf", "conf_epi", "conf_alea"):
        res[k] = cat[k][:, 0].numpy()
    return res


@torch.no_grad()
def encode(model, inputs, amp=True):
    with torch.autocast(device_type=inputs["image"].device.type, dtype=torch.bfloat16, enabled=amp):
        return model.encode(inputs)


def psnr(a, b):
    return float(-10 * np.log10(max(float(((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2).mean()), 1e-12)))


@torch.no_grad()
def degrid(img, p=8, R=5, tau=6.0):
    """Training-free grid removal (Suppl., 'Grid removal'): removes the p-periodic pattern that per-token pixel
    decoding can leave, without changing the brightness of any patch.  img: [V, H, W, 3] in [0, 1] (numpy)."""
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(0, 3, 1, 2).float() * 255.0
    V, C, H, W = x.shape
    h8, w8 = H // p, W // p
    xp = F.pad(x, (p // 2 - 1, p // 2, p // 2 - 1, p // 2), mode="reflect")
    high = x - F.avg_pool2d(xp, p, stride=1)                       # the periodic part has zero mean over any p x p window
    pt = high.reshape(V, C, h8, p, w8, p).permute(0, 1, 3, 5, 2, 4).reshape(V, C * p * p, h8, w8)
    w = 1.0 / (1.0 + pt.pow(2).mean(1, keepdim=True) / (tau * tau))   # flat patches show the grid without content
    pool = lambda t: F.avg_pool2d(t, R, stride=1, padding=R // 2, count_include_pad=False)   # noqa: E731
    pat = pool(pt * w) / pool(w).clamp_min(1e-6)
    pat = pat.reshape(V, C, p * p, h8, w8)
    pat = (pat - pat.mean(2, keepdim=True)).reshape(V, C, p, p, h8, w8).permute(0, 1, 4, 2, 5, 3).reshape(V, C, H, W)
    return ((x - pat) / 255.0).clamp(0, 1).permute(0, 2, 3, 1).numpy()


def turbo(x):
    """Turbo colour map of values in [0, 1] -> uint8 RGB."""
    import matplotlib
    return (matplotlib.colormaps["turbo"](np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)


def rank_match(u, ref):
    """Map the values of u onto the distribution of ref (same ranks): the uncertainty is shown on the scale of the
    error it predicts, which compares the two maps by their ranking only (as Pearson / AUSE do)."""
    order = np.argsort(u.ravel())
    out = np.empty(u.size, np.float64)
    out[order] = np.sort(ref.ravel())
    return out.reshape(u.shape)


def log_turbo(e, lo=None, hi=None):
    """log-scaled turbo colouring shared by error and uncertainty maps."""
    le = np.log(np.maximum(e, 1e-4))
    lo = np.percentile(le, 1) if lo is None else lo
    hi = np.percentile(le, 99.5) if hi is None else hi
    return turbo((le - lo) / max(hi - lo, 1e-6)), (lo, hi)
