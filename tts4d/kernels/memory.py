"""bf16 tensor-core path of the TTS layer's token-sized GEMMs: feature map, the write (G, C) and the read.

What stays fp32: every accumulator (G, C, S, T, A are fp32 GEMM outputs or fp32 reductions), the centring, the
scaling and the solve.  What becomes bf16 is only the GEMM *operands*: q, k, v come out of the bf16 ``to_qkv``
projection anyway, and a product of two bf16 numbers is exact in fp32, so the statistics are accumulated exactly,
only of operands rounded to bf16.

    feature_map   y = x @ [W0 | W2] as ONE GEMM (bf16 operands, fp32 output), then a fused Triton tail
                  phi = SiLU(g) * l / (||SiLU(g) * l|| + eps) in fp32, with a fused backward.
    moments       (G, C, S, T, A) from phi, v, a, the GEMMs on mean-shifted bf16 operands, hand-written backward.
    read_mm       pc @ W with bf16 operands, fp32 output and fp32 gradients.
    mm_f32        bf16 x bf16 -> fp32 GEMM.

Without Triton (or on the CPU) the same math runs in PyTorch.
"""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    triton = tl = None
    HAS_TRITON = False

BF16 = torch.bfloat16
_OUT_DTYPE = {}          # device type -> whether torch.mm(..., out_dtype=float32) works for bf16 operands


def _out_dtype_ok(device):
    key = device.type
    if key not in _OUT_DTYPE:
        try:
            a = torch.ones(2, 2, device=device, dtype=BF16)
            torch.mm(a, a, out_dtype=torch.float32)
            _OUT_DTYPE[key] = True
        except (TypeError, RuntimeError, NotImplementedError):
            _OUT_DTYPE[key] = False
    return _OUT_DTYPE[key]


def mm_f32(a, b):
    """a @ b (broadcast over leading dims) with operands rounded to bf16 and an fp32 result (fp32 accumulation).

    Uses ``torch.mm(out_dtype=float32)`` where available; otherwise an fp32 GEMM of the bf16-rounded operands, which
    forms the same (exact) products."""
    a, b = a.to(BF16), b.to(BF16)
    if not _out_dtype_ok(a.device):
        return torch.matmul(a.float(), b.float())
    if a.dim() == 2 and b.dim() == 2:
        return torch.mm(a, b, out_dtype=torch.float32)
    lead = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a3 = a.expand(*lead, *a.shape[-2:]).reshape(-1, *a.shape[-2:])
    b3 = b.expand(*lead, *b.shape[-2:]).reshape(-1, *b.shape[-2:])
    return torch.bmm(a3, b3, out_dtype=torch.float32).view(*lead, a.shape[-2], b.shape[-1])


def _tr(x):
    return x.transpose(-1, -2)


def _sum_to(t, shape):
    """Undo broadcasting: reduce ``t`` to ``shape``."""
    while t.dim() > len(shape):
        t = t.sum(0)
    dims = [i for i, (x, s) in enumerate(zip(t.shape, shape)) if s == 1 and x != 1]
    return t.sum(dim=dims, keepdim=True) if dims else t


# =====================================================================================================================
#  feature map:  y = x @ [W0 | W2]  ->  phi = SiLU(g) * l / (||.|| + eps)
# =====================================================================================================================
if HAS_TRITON:
    @triton.jit
    def _phi_fwd(Y, O, NRM, F_: tl.constexpr, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        c = tl.arange(0, BLOCK)
        m = c < F_
        g = tl.load(Y + row * (2 * F_) + c, mask=m, other=0.0).to(tl.float32)
        lv = tl.load(Y + row * (2 * F_) + F_ + c, mask=m, other=0.0).to(tl.float32)
        p = g * tl.sigmoid(g) * lv
        nrm = tl.sqrt(tl.sum(p * p, axis=0))
        tl.store(O + row * F_ + c, p / (nrm + eps), mask=m)
        tl.store(NRM + row, nrm)

    @triton.jit
    def _phi_bwd(Y, DO, NRM, DY, F_: tl.constexpr, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        c = tl.arange(0, BLOCK)
        m = c < F_
        g = tl.load(Y + row * (2 * F_) + c, mask=m, other=0.0).to(tl.float32)
        lv = tl.load(Y + row * (2 * F_) + F_ + c, mask=m, other=0.0).to(tl.float32)
        do = tl.load(DO + row * F_ + c, mask=m, other=0.0).to(tl.float32)
        nrm = tl.load(NRM + row)
        sig = tl.sigmoid(g)
        s = g * sig
        p = s * lv
        den = nrm + eps
        # d [p / (|p| + eps)] = do / den - p (p . do) / (den^2 |p|)
        pd = tl.sum(p * do, axis=0)
        coef = tl.where(nrm > 0, pd / (den * den * tl.where(nrm > 0, nrm, 1.0)), 0.0)
        dp = do / den - p * coef
        tl.store(DY + row * (2 * F_) + c, (dp * lv * (sig * (1.0 + g * (1.0 - sig)))).to(tl.bfloat16), mask=m)
        tl.store(DY + row * (2 * F_) + F_ + c, (dp * s).to(tl.bfloat16), mask=m)


def _phi_tail_torch(y, eps):
    g, lv = y.float().chunk(2, dim=-1)
    p = F.silu(g) * lv
    return p / (p.norm(dim=-1, keepdim=True) + eps)


def _launch(BLOCK):
    return 8 if BLOCK >= 2048 else 4


class _FeatureMap(torch.autograd.Function):
    """phi = tail(x @ [w0 | w2]): bf16 operands, fp32 GEMM output y (gate and linear branch are not rounded before
    the tail), fused Triton tail, fp32 phi.  Backward: fused tail gradient (bf16 dy, a GEMM operand), then
    dx = dy [w0 | w2]^T and d[w0 | w2] = x^T dy with fp32 outputs."""

    @staticmethod
    def forward(ctx, x, w0, w2, eps):
        wcat = torch.cat([w0, w2], dim=-1).to(BF16)          # [H, d, 2 dF]
        xb = x.to(BF16)
        y = mm_f32(xb, wcat).contiguous()                    # [B, H, n, 2 dF] fp32
        Fd = y.shape[-1] // 2
        rows = y.numel() // y.shape[-1]
        phi = torch.empty(*y.shape[:-1], Fd, device=y.device, dtype=torch.float32)
        nrm = torch.empty(rows, device=y.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(Fd)
        _phi_fwd[(rows,)](y, phi, nrm, Fd, eps, BLOCK=BLOCK, num_warps=_launch(BLOCK))
        ctx.save_for_backward(xb, wcat, y, nrm)
        ctx.eps, ctx.xdtype, ctx.Fd = eps, x.dtype, Fd
        return phi

    @staticmethod
    def backward(ctx, gphi):
        xb, wcat, y, nrm = ctx.saved_tensors
        Fd = ctx.Fd
        rows = y.numel() // y.shape[-1]
        dy = torch.empty(y.shape, device=y.device, dtype=BF16)
        BLOCK = triton.next_power_of_2(Fd)
        _phi_bwd[(rows,)](y, gphi.contiguous(), nrm, dy, Fd, ctx.eps, BLOCK=BLOCK, num_warps=_launch(BLOCK))
        gx = gw0 = gw2 = None
        if ctx.needs_input_grad[0]:
            gx = mm_f32(dy, _tr(wcat)).to(ctx.xdtype)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            gw = mm_f32(_tr(xb), dy)                         # [B, H, d, 2 dF]
            while gw.dim() > wcat.dim():
                gw = gw.sum(0)
            gw0, gw2 = gw[..., :Fd], gw[..., Fd:]
        return gx, gw0, gw2, None


def feature_map(x, w0, w2, eps=1e-6):
    """phi(x) = SiLU(x w0) * (x w2) / (||.|| + eps) with bf16 GEMM operands; fp32 output.

    x: [B, H, n, d]; w0, w2: [H, d, dF].  One GEMM against the packed [w0 | w2] instead of two."""
    if HAS_TRITON and x.is_cuda and 2 * w0.shape[-1] <= 32768:
        return _FeatureMap.apply(x, w0, w2, float(eps))
    # PyTorch fallback, differentiable by autograd: fp32 GEMM of the bf16-rounded operands (the same exact products)
    y = torch.matmul(x.to(BF16).float(), torch.cat([w0, w2], dim=-1).to(BF16).float())
    return _phi_tail_torch(y, eps)


# =====================================================================================================================
#  the write: sufficient statistics from bf16 operands, shifted by their mean
# =====================================================================================================================
def _outer(u, v):
    """u^T v for row vectors [.., 1, p], [.., 1, q] -> [.., p, q] in exact fp32 (not a K = 1 GEMM, which would run
    in TF32 when that is enabled; these rank-one terms are the large, nearly cancelling mean parts)."""
    return _tr(u) * v


def _vecmat(u, M):
    """u @ M for a row vector u [.., 1, p] and M [.., p, q] in exact fp32."""
    return (_tr(u) * M).sum(dim=-2, keepdim=True)


class _Moments(torch.autograd.Function):
    """G = sum a phi phi^T, C = sum a phi v^T, S = sum a phi, T = sum a v, A = sum a.

    phi: [B, H, n, dF] fp32;  v: [B, H, n, dV] (bf16 or fp32);  a: [B, H, n, 1] fp32.

    The features share a large mean direction, so phi ~ phi_bar, and a bf16 GEMM on phi itself would round away the
    part that the centring later keeps.  The GEMMs therefore run on the CENTRED operands delta = phi - c,
    eps = v - d, with c, d the (constant) weighted means of this chunk, and the rank-one terms are added back in fp32:

        G = G_d + c^T S_d + S_d^T c + A c^T c,  S = S_d + A c
        C = C_d + S_d^T d + c^T T_e + A c^T d,  T = T_e + A d

    which is exact algebra for any constant shift.  Backward, with Gsym = dG + dG^T:
        u_phi = c Gsym + d dC^T + dS,   u_v = c dC + dT
        d phi = a (delta Gsym + eps dC^T + u_phi)
        d v   = a (delta dC + u_v)
        d a   = rowsum(delta (delta Gsym / 2 + eps dC^T + u_phi)) + eps . u_v + const
    """

    @staticmethod
    def forward(ctx, phi, v, a):
        A = a.sum(dim=-2, keepdim=True)                      # [B, H, 1, 1]
        Ac = A.clamp_min(1e-30)
        c = (phi * a).sum(dim=-2, keepdim=True) / Ac         # [B, H, 1, dF]
        d = (v * a).sum(dim=-2, keepdim=True) / Ac           # [B, H, 1, dV]
        dl32 = phi.float() - c
        ep32 = v.float() - d
        adl32 = dl32 * a
        S_d = adl32.sum(dim=-2, keepdim=True)
        T_e = (ep32 * a).sum(dim=-2, keepdim=True)
        dl, ep, adl = dl32.to(BF16), ep32.to(BF16), adl32.to(BF16)
        del dl32, ep32, adl32
        G = mm_f32(_tr(adl), dl)
        C = mm_f32(_tr(adl), ep)
        G = G + _outer(c, S_d) + _outer(S_d, c) + A * _outer(c, c)
        C = C + _outer(S_d, d) + _outer(c, T_e) + A * _outer(c, d)
        S = S_d + A * c
        T = T_e + A * d
        ctx.save_for_backward(dl, ep, a, c, d)
        ctx.v_dtype = v.dtype
        return G, C, S, T, A

    @staticmethod
    def backward(ctx, dG, dC, dS, dT, dA):
        dl, ep, a, c, d = ctx.saved_tensors
        B, H, n, dF = dl.shape
        dV = ep.shape[-1]
        z = lambda *sh: torch.zeros(*sh, device=a.device, dtype=torch.float32)  # noqa: E731
        dG = z(B, H, dF, dF) if dG is None else dG
        dC = z(B, H, dF, dV) if dC is None else dC
        dS = z(B, H, 1, dF) if dS is None else dS
        dT = z(B, H, 1, dV) if dT is None else dT
        dA = z(B, H, 1, 1) if dA is None else dA
        Gsym = dG + _tr(dG)
        u_phi = _vecmat(c, Gsym) + _vecmat(d, _tr(dC)) + dS   # exact fp32: c Gsym nearly cancels dS
        u_v = _vecmat(c, dC) + dT
        const = ((_vecmat(c, dG) * c).sum(-1, keepdim=True) + (_vecmat(c, dC) * d).sum(-1, keepdim=True)
                 + (dS * c).sum(-1, keepdim=True) + (dT * d).sum(-1, keepdim=True) + dA)
        MG = mm_f32(dl, Gsym)                                # [B, H, n, dF]
        MC = mm_f32(ep, _tr(dC))                             # [B, H, n, dF]
        g_phi = g_v = g_a = None
        if ctx.needs_input_grad[0]:
            g_phi = a * (MG + MC + u_phi)
        if ctx.needs_input_grad[1]:
            g_v = (a * (mm_f32(dl, dC) + u_v)).to(ctx.v_dtype)
        if ctx.needs_input_grad[2]:
            g_a = ((dl.float() * (0.5 * MG + MC + u_phi)).sum(-1, keepdim=True)
                   + (ep.float() * u_v).sum(-1, keepdim=True) + const)
        return g_phi, g_v, g_a


def moments(phi, v, a):
    """(G, C, S, T, A) of one chunk of tokens; see ``_Moments``.  All outputs fp32."""
    return _Moments.apply(phi, v, a)


# =====================================================================================================================
#  the read
# =====================================================================================================================
class _ReadMM(torch.autograd.Function):
    """pc @ W with bf16 operands; fp32 output AND fp32 gradients (dW feeds the solve's backward, where it meets a
    difference W0 - W, so it is not rounded to bf16)."""

    @staticmethod
    def forward(ctx, pc, W):
        pcb, Wb = pc.to(BF16), W.to(BF16)
        ctx.save_for_backward(pcb, Wb)
        ctx.W_shape = W.shape
        return mm_f32(pcb, Wb)

    @staticmethod
    def backward(ctx, go):
        pcb, Wb = ctx.saved_tensors
        g_pc = mm_f32(go, _tr(Wb)) if ctx.needs_input_grad[0] else None
        g_W = _sum_to(mm_f32(_tr(pcb), go), ctx.W_shape) if ctx.needs_input_grad[1] else None
        return g_pc, g_W


def read_mm(pc, W):
    """pc [B, H, n, dF] @ W [B, H, dF, dV] -> fp32 [B, H, n, dV] (bf16 operands)."""
    return _ReadMM.apply(pc, W)
