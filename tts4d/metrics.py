"""Image-quality and uncertainty metrics of the paper's evaluation.

Every benchmark is scored with the metric code of the protocol it comes from (per target image, then mean over the
targets of a scene, then mean over scenes):

  ``"default"``      Stereo4D, DyCheck (time interpolation), DL3DV-140.  PSNR of the float prediction (clamped to [0, 1]),
                     SSIM = scikit-image ``structural_similarity`` (11 x 11 Gaussian window, data range 1), LPIPS (VGG,
                     inputs in [0, 1] with ``normalize=True``).  The metric code of the Long-LRM / MVP DL3DV benchmark.
  ``"dynamicnerf"``  NVIDIA Dynamic Scenes, as DynamicNeRF ``utils/evaluation.py``: renders truncated to uint8 and read as
                     BGR, PSNR of the uint8 images, scikit-image SSIM on uint8 (default 7 x 7 window), LPIPS (AlexNet) on
                     the BGR images scaled to [-1, 1].
  ``"dycheck"``      DyCheck novel views (supplementary), as ``dycheck/core/metrics/image.py``: renders truncated to uint8,
                     PSNR / SSIM / LPIPS (AlexNet, spatial) restricted to the released covisibility mask (mPSNR, mSSIM,
                     mLPIPS).
  ``"3dgs"``         Tanks and Temples / Mip-NeRF 360, as 3D Gaussian Splatting ``metrics.py``: renders rounded to uint8,
                     PSNR, 3DGS SSIM (11 x 11 Gaussian, zero padding), LPIPS (VGG) on the [0, 1] images.

Uncertainty (``uncertainty_metrics``), protocol of GS-U (and FisherRF): per target image, the predicted standard deviation
``exp(U / 2)`` is compared with the per-pixel L1 error (mean over RGB) and the DSSIM error (1 - 3DGS SSIM map, mean over
RGB) by the Pearson correlation and by the area under the sparsification error curve (AUSE, 100 removal fractions in
[0, 0.999], curves normalised by the mean error of the image).

Optional dependencies: scikit-image (SSIM), lpips (LPIPS), scipy (DyCheck SSIM, CPU fallback only).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

_CACHE = {}


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_float(x):
    """numpy / torch, uint8 or float, [N, 3, H, W] -> float32 torch tensor in [0, 1] (uint8 / 255)."""
    x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x)
    return x.float() / 255.0 if x.dtype == torch.uint8 else x.float()


def to_uint8(x, mode="round"):
    """[0, 1] float -> uint8 as the official pipelines save their renders: 'round' (3DGS, torchvision save_image) or
    'trunc' (NeRF to8b / DyCheck to_uint8: (x * 255).clip(0, 255).astype(uint8))."""
    x = torch.as_tensor(x)
    if x.dtype == torch.uint8:
        return x
    y = x.float() * 255.0
    if mode == "round":
        return y.clamp(0, 255).round().to(torch.uint8)
    return y.clamp(0, 255).floor().to(torch.uint8)


def lpips_net(net="vgg", spatial=False, device=None):
    key = ("lpips", net, spatial, str(device or _dev()))
    if key not in _CACHE:
        import lpips
        _CACHE[key] = lpips.LPIPS(net=net, spatial=spatial, verbose=False).eval().to(device or _dev())
    return _CACHE[key]


# ----------------------------------------------------------------------------------------------- "default"
@torch.no_grad()
def psnr(pred, gt):
    """Per-image PSNR of [N, 3, H, W] images in [0, 1] (both clamped)."""
    p, g = _as_float(pred).clamp(0, 1), _as_float(gt).clamp(0, 1)
    mse = ((g - p) ** 2).mean(dim=(1, 2, 3))
    return (-10.0 * torch.log10(mse)).tolist()


def ssim_skimage(pred, gt, win_size=11, gaussian_weights=True):
    """Per-image scikit-image SSIM (channel axis 0, data range 1) of [N, 3, H, W] images in [0, 1]."""
    from skimage.metrics import structural_similarity
    p, g = _as_float(pred).cpu().numpy(), _as_float(gt).cpu().numpy()
    return [float(structural_similarity(gi, pi, win_size=win_size, gaussian_weights=gaussian_weights, channel_axis=0, data_range=1.0))
            for gi, pi in zip(g, p)]


@torch.no_grad()
def lpips_vgg(pred, gt, batch=10):
    """Per-image LPIPS (VGG, lpips 0.1) of [N, 3, H, W] images in [0, 1] (``normalize=True``)."""
    dev = _dev()
    net = lpips_net("vgg", device=dev)
    p, g = _as_float(pred).to(dev), _as_float(gt).to(dev)
    out = [net(g[i:i + batch], p[i:i + batch], normalize=True).reshape(-1) for i in range(0, len(p), batch)]
    return torch.cat(out).tolist()


def default_metrics(pred, gt):
    return {"psnr": psnr(pred, gt), "ssim": ssim_skimage(pred, gt), "lpips": lpips_vgg(pred, gt)}


# ----------------------------------------------------------------------------------------------- "dynamicnerf"
@torch.no_grad()
def dynamicnerf_metrics(pred, gt):
    """NVIDIA Dynamic Scenes, DynamicNeRF utils/evaluation.py (renders saved with to8b and read back with cv2)."""
    from skimage.metrics import structural_similarity
    dev = _dev()
    net = lpips_net("alex", device=dev)
    P = to_uint8(_as_float(pred), "trunc").permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]     # RGB -> BGR (cv2.imread)
    G = to_uint8(_as_float(gt), "trunc").permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]
    im2t = lambda im: torch.from_numpy(np.ascontiguousarray(im).transpose(2, 0, 1) / 127.5 - 1.0).float()[None].to(dev)   # noqa: E731
    out = {"psnr": [], "ssim": [], "lpips": []}
    for g, p in zip(G, P):
        g, p = np.ascontiguousarray(g), np.ascontiguousarray(p)
        mse = float(((g.astype(np.float64) - p.astype(np.float64)) ** 2).mean())       # = cv2.PSNR(g, p)
        out["psnr"].append(10.0 * math.log10(255.0 ** 2 / mse) if mse > 0 else 361.20199909921956)
        out["ssim"].append(float(structural_similarity(g, p, channel_axis=-1)))
        out["lpips"].append(float(net(im2t(g), im2t(p)).item()))
    return out


# ----------------------------------------------------------------------------------------------- "3dgs"
def _gauss_window(size=11, sigma=1.5, channels=3, device="cpu"):
    g = torch.tensor([math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
    g = (g / g.sum()).unsqueeze(1)
    w = g.mm(g.t()).float().unsqueeze(0).unsqueeze(0)
    return w.expand(channels, 1, size, size).contiguous().to(device)


def ssim_3dgs(img1, img2, window_size=11, map=False):
    """SSIM of 3D Gaussian Splatting (utils/loss_utils.ssim): Gaussian window 11 (sigma 1.5), zero padding.
    img: [B, C, H, W] in [0, 1].  map=True returns the per-pixel SSIM map [B, C, H, W] (GS-U's DSSIM error)."""
    C = img1.shape[-3]
    w = _gauss_window(window_size, 1.5, C, img1.device).type_as(img1)
    pad = window_size // 2
    mu1, mu2 = F.conv2d(img1, w, padding=pad, groups=C), F.conv2d(img2, w, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(img1 * img1, w, padding=pad, groups=C) - mu1_sq
    s2 = F.conv2d(img2 * img2, w, padding=pad, groups=C) - mu2_sq
    s12 = F.conv2d(img1 * img2, w, padding=pad, groups=C) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return m if map else m.mean()


@torch.no_grad()
def gs_metrics(pred, gt):
    """Tanks and Temples / Mip-NeRF 360, 3DGS metrics.py on the 8-bit (rounded) renders.  LPIPS = lpipsPyTorch VGG on the
    [0, 1] images, which equals the lpips package's VGG LPIPS called without input normalisation."""
    dev = _dev()
    net = lpips_net("vgg", device=dev)
    P = to_uint8(_as_float(pred), "round").float() / 255.0
    G = to_uint8(_as_float(gt), "round").float() / 255.0
    out = {"psnr": [], "ssim": [], "lpips": []}
    for p, g in zip(P, G):
        r, t = p[None].to(dev), g[None].to(dev)
        mse = ((r - t) ** 2).view(1, -1).mean(1, keepdim=True)
        out["psnr"].append(float(20 * torch.log10(1.0 / torch.sqrt(mse))))
        out["ssim"].append(float(ssim_3dgs(r, t)))
        out["lpips"].append(float(net(r, t)))
    return out


# ----------------------------------------------------------------------------------------------- "dycheck"
def _masked_mean(x, m):
    m = torch.broadcast_to(m, x.shape)
    return (x * m).sum() / m.sum().clamp_min(1e-6)


def _dycheck_ssim(a, b, m, max_val=1.0, size=11, sigma=1.5, k1=0.01, k2=0.03):
    """dycheck compute_ssim: separable Gaussian as a partial ('valid') convolution, rows first.  a, b [3, H, W], m [1, H, W]
    (float64)."""
    hw = size // 2
    shift = (2 * hw - size + 1) / 2
    f = torch.exp(-0.5 * ((torch.arange(size, dtype=a.dtype, device=a.device) - hw + shift) / sigma) ** 2)
    f = f / f.sum()
    fr, fc = f.flip(0).view(1, 1, 1, -1), f.flip(0).view(1, 1, -1, 1)
    orr, occ = torch.ones_like(fr), torch.ones_like(fc)
    M = m[None]

    def conv(z, mm, k, ones):
        z_ = F.conv2d(z * mm, k)
        m_ = F.conv2d(mm, ones)
        nz = m_ != 0
        return torch.where(nz, z_ * ones.sum() / torch.where(nz, m_, torch.ones_like(m_)), torch.zeros_like(z_)), nz.to(z.dtype)

    def filt(z):
        z2, m2 = conv(z[:, None], M, fr, orr)
        return conv(z2, m2, fc, occ)[0]
    mu0, mu1 = filt(a), filt(b)
    mu00, mu11, mu01 = mu0 * mu0, mu1 * mu1, mu0 * mu1
    s00 = (filt(a * a) - mu00).clamp_min(0)
    s11 = (filt(b * b) - mu11).clamp_min(0)
    s01 = filt(a * b) - mu01
    s01 = torch.sign(s01) * torch.minimum(torch.sqrt(s00 * s11), s01.abs())
    c1, c2 = (k1 * max_val) ** 2, (k2 * max_val) ** 2
    return float((((2 * mu01 + c1) * (2 * s01 + c2)) / ((mu00 + mu11 + c1) * (s00 + s11 + c2))).mean())


@torch.no_grad()
def dycheck_metrics(pred, gt, mask):
    """DyCheck novel views (core/tasks/novel_view.py): masked PSNR / SSIM / LPIPS with the covisibility mask [N, 1, H, W]."""
    dev = _dev()
    net = lpips_net("alex", spatial=True, device=dev)
    P = to_uint8(_as_float(pred), "trunc").double() / 255.0
    G = to_uint8(_as_float(gt), "trunc").double() / 255.0
    Mk = torch.as_tensor(np.asarray(mask)).double()
    out = {"psnr": [], "ssim": [], "lpips": []}
    for p, g, m in zip(P, G, Mk):
        p, g, m = p.to(dev), g.to(dev), m.to(dev)
        out["psnr"].append(float(-10.0 / math.log(10.0) * torch.log(_masked_mean((g - p) ** 2, m))))
        out["ssim"].append(_dycheck_ssim(g, p, m))
        t = lambda x: ((x * m) * 2.0 - 1.0)[None].float()                       # noqa: E731  lpips.im2tensor(img * mask)
        d = net(t(g), t(p))[0].double()
        out["lpips"].append(float(_masked_mean(d, m)))
    return out


PROTOCOLS = ("default", "dynamicnerf", "dycheck", "3dgs")


def image_metrics(pred, gt, protocol="default", mask=None):
    """{'psnr': [...], 'ssim': [...], 'lpips': [...]} per target.  pred, gt: [N, 3, H, W] (float in [0, 1] or uint8)."""
    if protocol == "default":
        return default_metrics(pred, gt)
    if protocol == "dynamicnerf":
        return dynamicnerf_metrics(pred, gt)
    if protocol == "3dgs":
        return gs_metrics(pred, gt)
    if protocol == "dycheck":
        assert mask is not None, "the DyCheck protocol needs the covisibility masks"
        return dycheck_metrics(pred, gt, mask)
    raise ValueError(f"unknown metric protocol {protocol!r}, expected one of {PROTOCOLS}")


# ----------------------------------------------------------------------------------------------- uncertainty
def pearson(u, e):
    """Pearson correlation of two maps (float64), over the pixels where the error is not NaN."""
    u, e = torch.as_tensor(u).reshape(-1), torch.as_tensor(e).reshape(-1)
    keep = e == e
    x, y = u[keep].double(), e[keep].double()
    xm, ym = x - x.mean(), y - y.mean()
    return float((xm * ym).sum() / torch.sqrt((xm * xm).sum() * (ym * ym).sum()))


def ause(error, uncertainty, n_steps=100):
    """Area under the sparsification error curve (GS-U / FisherRF protocol).  Pixels are removed in order of decreasing
    uncertainty (curve) and of decreasing error (oracle) for 100 fractions in [0, 0.999]; both curves are the mean error of the
    kept pixels divided by the mean error of all pixels; AUSE is the trapezoidal area between them.  Pixels with a zero error
    or a zero uncertainty are ignored.  Returns (ause, oracle curve, uncertainty curve)."""
    e = torch.as_tensor(error).reshape(-1)
    u = torch.as_tensor(uncertainty).reshape(-1)
    keep = (e == e)
    e, u = e[keep], u[keep]
    valid = torch.logical_and(e != 0.0, u != 0.0)
    if not torch.any(valid):
        return 0.0, None, None
    e, u = e[valid], u[valid]
    ratio = torch.linspace(0, 0.999, n_steps, device=e.device)
    n = len(e)
    idx = ((1 - ratio) * n).to(int)[:-1]
    e_sorted, _ = torch.sort(e)
    oracle = torch.cumsum(e_sorted, dim=0)[idx - 1] / idx
    _, order = torch.sort(u)
    by_u = torch.cumsum(e[order], dim=0)[idx - 1] / idx
    start = oracle[0]
    oracle, by_u = oracle / start, by_u / start
    area = torch.trapezoid(by_u - oracle, ratio[:len(oracle)])
    return float(area), oracle, by_u


@torch.no_grad()
def uncertainty_metrics(std, pred, gt, mask=None):
    """Pearson / AUSE of the predicted standard deviation against the L1 and DSSIM errors, per target.
    std [N, H, W], pred / gt [N, 3, H, W] (float in [0, 1] or uint8), optional mask [N, 1, H, W] (pixels to keep).
    Returns {'pearson_L1': [...], 'pearson_DSSIM': [...], 'AUSE_L1': [...], 'AUSE_DSSIM': [...]}."""
    dev = _dev()
    P, G = _as_float(pred), _as_float(gt)
    S = torch.as_tensor(np.asarray(std) if not torch.is_tensor(std) else std).float()
    out = {k: [] for k in ("pearson_L1", "pearson_DSSIM", "AUSE_L1", "AUSE_DSSIM")}
    for i in range(len(P)):
        g, p, u = G[i].to(dev), P[i].to(dev), S[i].to(dev)
        err = {"L1": (g - p).abs().mean(0), "DSSIM": 1.0 - ssim_3dgs(g[None], p[None], map=True)[0].mean(0)}
        sel = None
        if mask is not None:
            sel = torch.as_tensor(np.asarray(mask[i])).to(dev)[0] > 0
            if int(sel.sum()) < 2:
                continue
        for name, e in err.items():
            ee, uu = (e[sel], u[sel]) if sel is not None else (e, u)
            out[f"pearson_{name}"].append(pearson(uu, ee))
            out[f"AUSE_{name}"].append(ause(ee, uu)[0])
    return out
