"""Per-head RMSNorm of q and k on the packed qkv tensor of the within-view self-attention (Triton).

``SelfAttention`` computes ``qkv = to_qkv(x)`` -> [b, l, 3, H, Dh] and then ``nn.RMSNorm(Dh)`` (eps=None) of the
q and k heads.  The composite PyTorch RMSNorm is several elementwise kernels forward and more backward; here the
forward is ONE kernel that reads qkv once and writes the packed [b, l, 3, H, Dh] tensor (q, k normalised, v copied),
and the backward one kernel that writes the packed dqkv.

    r = rsqrt(mean(x_f32^2) + eps)            (eps = finfo(fp32).eps if None, as PyTorch's rms_norm uses)
    y = (x_f32 * r * w_f32).to(x.dtype)       (one rounding)
Backward, exact in fp32 until the final store:
    dn = dy * w;  dx = r * (dn - n * mean(dn * n));  dw = sum(dy * n),  n = x * r
(dw is reduced per 64-row tile, then summed over tiles).

PyTorch fallback and test reference: ``qk_norm_packed_ref``.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    triton = tl = None
    HAS_TRITON = False

BLOCK_R = 64


def _eff_eps(eps, dtype):
    """The eps PyTorch's rms_norm adds (fp32 machine epsilon if None), as stored in ``dtype``."""
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    return float(torch.tensor(float(eps), dtype=dtype).float())


if HAS_TRITON:
    @triton.jit
    def _qknorm_fwd_kernel(X, Y, WQ, WK, RSTD, n_rows, eps, H: tl.constexpr, DH: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK + tl.arange(0, BLOCK)          # row = token * H + head
        cols = tl.arange(0, DH)
        rmask = rows < n_rows
        m2 = rmask[:, None]
        tok = (rows // H).to(tl.int64)
        head = rows % H
        offq = (tok * (3 * H * DH) + head * DH)[:, None] + cols[None, :]
        offk = offq + H * DH
        offv = offq + 2 * H * DH
        xq = tl.load(X + offq, mask=m2, other=0.0)
        xk = tl.load(X + offk, mask=m2, other=0.0)
        xv = tl.load(X + offv, mask=m2, other=0.0)
        wq = tl.load(WQ + cols).to(tl.float32)
        wk = tl.load(WK + cols).to(tl.float32)
        xqf = xq.to(tl.float32)
        xkf = xk.to(tl.float32)
        rq = 1.0 / tl.sqrt(tl.sum(xqf * xqf, axis=1) / DH + eps)
        rk = 1.0 / tl.sqrt(tl.sum(xkf * xkf, axis=1) / DH + eps)
        tl.store(Y + offq, (xqf * rq[:, None] * wq[None, :]).to(xq.dtype), mask=m2)
        tl.store(Y + offk, (xkf * rk[:, None] * wk[None, :]).to(xk.dtype), mask=m2)
        tl.store(Y + offv, xv, mask=m2)
        tl.store(RSTD + rows * 2, rq, mask=rmask)
        tl.store(RSTD + rows * 2 + 1, rk, mask=rmask)

    @triton.jit
    def _qknorm_bwd_kernel(X, DY, DX, WQ, WK, RSTD, DWQ, DWK, n_rows,
                           H: tl.constexpr, DH: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK + tl.arange(0, BLOCK)
        cols = tl.arange(0, DH)
        rmask = rows < n_rows
        m2 = rmask[:, None]
        tok = (rows // H).to(tl.int64)
        head = rows % H
        offq = (tok * (3 * H * DH) + head * DH)[:, None] + cols[None, :]
        offk = offq + H * DH
        offv = offq + 2 * H * DH
        dv = tl.load(DY + offv, mask=m2, other=0.0)        # v: identity
        tl.store(DX + offv, dv, mask=m2)
        xq = tl.load(X + offq, mask=m2, other=0.0)
        dyq = tl.load(DY + offq, mask=m2, other=0.0).to(tl.float32)
        rq = tl.load(RSTD + rows * 2, mask=rmask, other=0.0)
        wq = tl.load(WQ + cols).to(tl.float32)
        nq = xq.to(tl.float32) * rq[:, None]
        dnq = dyq * wq[None, :]
        cq = tl.sum(dnq * nq, axis=1) / DH
        tl.store(DX + offq, (rq[:, None] * (dnq - nq * cq[:, None])).to(xq.dtype), mask=m2)
        tl.store(DWQ + pid * DH + cols, tl.sum(dyq * nq, axis=0))
        xk = tl.load(X + offk, mask=m2, other=0.0)
        dyk = tl.load(DY + offk, mask=m2, other=0.0).to(tl.float32)
        rk = tl.load(RSTD + rows * 2 + 1, mask=rmask, other=0.0)
        wk = tl.load(WK + cols).to(tl.float32)
        nk = xk.to(tl.float32) * rk[:, None]
        dnk = dyk * wk[None, :]
        ck = tl.sum(dnk * nk, axis=1) / DH
        tl.store(DX + offk, (rk[:, None] * (dnk - nk * ck[:, None])).to(xk.dtype), mask=m2)
        tl.store(DWK + pid * DH + cols, tl.sum(dyk * nk, axis=0))


class _QKNormPacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, wq, wk, eps):
        shape = qkv.shape                       # [..., 3, H, DH]
        H, DH = shape[-2], shape[-1]
        x = qkv.reshape(-1, 3 * H * DH).contiguous()
        n_rows = x.shape[0] * H
        wq32, wk32 = wq.detach().float().contiguous(), wk.detach().float().contiguous()
        y = torch.empty_like(x)
        rstd = torch.empty(n_rows, 2, device=x.device, dtype=torch.float32)
        _qknorm_fwd_kernel[(triton.cdiv(n_rows, BLOCK_R),)](x, y, wq32, wk32, rstd, n_rows, eps,
                                                            H=H, DH=DH, BLOCK=BLOCK_R, num_warps=4)
        ctx.save_for_backward(x, wq32, wk32, rstd)
        ctx.shape, ctx.w_dtypes = shape, (wq.dtype, wk.dtype)
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x, wq32, wk32, rstd = ctx.saved_tensors
        H, DH = ctx.shape[-2], ctx.shape[-1]
        dy = dy.reshape(x.shape).contiguous()
        n_rows = x.shape[0] * H
        n_prog = triton.cdiv(n_rows, BLOCK_R)
        dx = torch.empty_like(x)
        dwq = torch.empty(n_prog, DH, device=x.device, dtype=torch.float32)
        dwk = torch.empty(n_prog, DH, device=x.device, dtype=torch.float32)
        _qknorm_bwd_kernel[(n_prog,)](x, dy, dx, wq32, wk32, rstd, dwq, dwk, n_rows,
                                      H=H, DH=DH, BLOCK=BLOCK_R, num_warps=4)
        gq = dwq.sum(0).to(ctx.w_dtypes[0]) if ctx.needs_input_grad[1] else None
        gk = dwk.sum(0).to(ctx.w_dtypes[1]) if ctx.needs_input_grad[2] else None
        return dx.view(ctx.shape), gq, gk, None


def qk_norm_packed_ref(qkv, wq, wk, eps=None):
    """PyTorch reference: the same formula (fp32 math, one rounding to the input dtype)."""
    q, k, v = qkv.unbind(-3)
    e = _eff_eps(eps, qkv.dtype)

    def n(t, w):
        tf = t.float()
        return (tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + e) * w.float()).to(t.dtype)
    return torch.stack([n(q, wq), n(k, wk), v], dim=-3)


def qk_norm_ok(qkv):
    return (HAS_TRITON and qkv.is_cuda and qkv.dtype in (torch.bfloat16, torch.float16) and qkv.dim() >= 3
            and qkv.shape[-3] == 3 and qkv.shape[-1] in (16, 32, 64, 128))


def qk_norm_packed(qkv, wq, wk, eps=None):
    """RMSNorm over the last axis of qkv[..., 0, :, :] (q) and qkv[..., 1, :, :] (k); v is copied.

    qkv: [..., 3, H, Dh] (bf16 / fp16 for the Triton path); wq, wk: [Dh].  Returns a new packed tensor."""
    if qk_norm_ok(qkv):
        return _QKNormPacked.apply(qkv, wq, wk, _eff_eps(eps, qkv.dtype))
    return qk_norm_packed_ref(qkv, wq, wk, eps)
