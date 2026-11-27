"""The TTS layer's solve, with the centring and scaling of the moments fused in and a closed-form backward.

Forward (identical to ``TTSLayer.solve``, given the uncentred statistics):

    Gc = G - S^T phi_bar          (= G - S^T S / A)          Cc = C - S^T v_bar
    Gs = Gc * inv^T * inv,        Cs = Cc * inv^T              (inv = D^-1, the diagonal scaling)
    P  = 0.5 (Gs + Gs^T) + lam I = L L^T                      one Cholesky (with jitter escalation if it fails)
    M  = L^-1                                                  triangular inverse, dF right-hand sides
    W  = P^-1 (Cs + lam W0) = M^T (M (Cs + lam W0))            two GEMMs

``M`` is returned detached: its only other consumer is the epistemic term u(q) = ||M pc(q)||^2, which carries no
gradient.  ``phi_bar``, ``v_bar`` and ``inv`` are small vectors computed by autograd outside; this function returns
their gradients.  Backward, with gR = P^-1 gW = M^T (M gW) and E = -sym(gR W^T) = dL/dGs:

    dG = E * inv^T inv,   dC = gR * inv^T
    dinv = inv (E * Gc + (E * Gc)^T) + rowsum(gR * Cc)
    dS = -(phi_bar dG^T + v_bar dC^T),   dphi_bar = -S dG,   dv_bar = -S dC
    dlam = <gR, W0 - W>,   dW0 = lam gR

No dF x dF intermediate is kept for the backward (Gc and Cc are recomputed there from G and C).  The elementwise
dF x dF / dF x dV passes (forming P and Cs + lam W0 forward; dG, dC and the row sums backward) are Triton kernels
(``solve_triton``) with PyTorch twins below.  All matrix products run in IEEE fp32 (TF32 off).
"""
import contextlib

import torch

from . import solve_triton as _TR


@contextlib.contextmanager
def ieee_fp32():
    """TF32 disabled for the fp32 matmuls inside (restored on exit)."""
    m = torch.backends.cuda.matmul
    prev = m.allow_tf32
    m.allow_tf32 = False
    try:
        yield
    finally:
        m.allow_tf32 = prev


def robust_cholesky(P, jitter=1e-4):
    """Cholesky factor of a symmetric positive definite matrix; adds a small jitter (relative to the mean diagonal)
    only if the plain factorisation fails."""
    L, info = torch.linalg.cholesky_ex(P)
    if not bool((info != 0).any()):
        return L
    dF = P.shape[-1]
    eye = torch.eye(dF, device=P.device, dtype=P.dtype)
    base = torch.diagonal(P, dim1=-2, dim2=-1).clamp_min(0).mean(-1, keepdim=True).unsqueeze(-1)
    base = base.clamp_min(torch.finfo(P.dtype).eps * dF)
    for k in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        L, info = torch.linalg.cholesky_ex(P + (jitter + k * base) * eye)
        if not bool((info != 0).any()):
            return L
    return torch.linalg.cholesky(torch.diag_embed(torch.diagonal(P, dim1=-2, dim2=-1).clamp_min(base.squeeze(-1))))


def _sum_to(t, shape):
    """Undo broadcasting: reduce ``t`` to ``shape``."""
    while t.dim() > len(shape):
        t = t.sum(0)
    dims = [i for i, (x, s) in enumerate(zip(t.shape, shape)) if s == 1 and x != 1]
    return t.sum(dim=dims, keepdim=True) if dims else t


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _use_triton(G, C, W0):
    return (_TR.available() and G.is_cuda and G.dtype == torch.float32 and C.dtype == torch.float32
            and (W0 is None or _numel(W0.shape[:-2]) == 1))


class _CentredSolve(torch.autograd.Function):

    @staticmethod
    def forward(ctx, G, C, S, pb, vb, inv, lam, W0, jitter):
        lead = G.shape[:-2]
        n, dV = G.shape[-1], C.shape[-1]
        N = _numel(lead)
        lam_b = lam.expand(*lead, 1, 1)
        W0e = W0.expand(*lead, n, dV) if W0 is not None else None
        use_tr = _use_triton(G, C, W0)
        P = R = None
        if use_tr:
            try:
                lam1 = lam_b.reshape(N).contiguous()
                S2, INV2 = S.reshape(N, n).contiguous(), inv.reshape(N, n).contiguous()
                P = _TR.pform(G.reshape(N, n, n).contiguous(), S2, pb.reshape(N, n).contiguous(), INV2, lam1)
                R = _TR.rform(C.reshape(N, n, dV).contiguous(), S2, vb.reshape(N, dV).contiguous(), INV2, lam1,
                              W0.reshape(n, dV).contiguous() if W0 is not None else None)
            except Exception as ex:  # noqa: BLE001
                _TR.mark_broken(ex)
                use_tr, P, R = False, None, None
        if not use_tr:
            St, it = S.transpose(-1, -2), inv.transpose(-1, -2)
            Gs = (G - St * pb) * it * inv
            P = 0.5 * (Gs + Gs.transpose(-1, -2))
            del Gs
            P.diagonal(dim1=-2, dim2=-1).add_(lam_b.squeeze(-1))
            R = (C - St * vb) * it
            if W0 is not None:
                R = R + lam_b * W0e
            P, R = P.reshape(N, n, n), R.reshape(N, n, dV)
        with ieee_fp32():
            L = robust_cholesky(P, jitter)
            del P
            eye = torch.eye(n, device=G.device, dtype=G.dtype)
            M = torch.linalg.solve_triangular(L, eye.expand(L.shape), upper=False)
            del L
            W = M.transpose(-1, -2) @ (M @ R)
        del R
        M, W = M.view(*lead, n, n), W.view(*lead, n, dV)
        ctx.use_tr = use_tr
        ctx.has_w0 = W0 is not None
        ctx.lam_shape = lam.shape
        ctx.W0_shape = W0.shape if W0 is not None else None
        ctx.save_for_backward(G, C, S, pb, vb, inv, lam, W0 if W0 is not None else G.new_empty(0), M, W)
        ctx.mark_non_differentiable(M)
        return M, W

    @staticmethod
    def backward(ctx, gM, gW):
        if gW is None:
            return (None,) * 9
        G, C, S, pb, vb, inv, lam, W0, M, W = ctx.saved_tensors
        W0 = W0 if ctx.has_w0 else None
        lead = G.shape[:-2]
        n, dV = G.shape[-1], C.shape[-1]
        N = _numel(lead)
        with ieee_fp32():
            gR = M.transpose(-1, -2) @ (M @ gW)                       # P^-1 gW
            gR3, W3 = gR.reshape(N, n, dV), W.reshape(N, n, dV)
            AW = torch.bmm(gR3, W3.transpose(1, 2))
            if ctx.use_tr and _TR.available():
                try:
                    return _CentredSolve._backward_triton(ctx, AW, G, C, S, pb, vb, inv, lam, W0, W3, gR, gR3,
                                                          lead, N, n, dV)
                except Exception as ex:  # noqa: BLE001
                    _TR.mark_broken(ex)
            E = (AW + AW.transpose(1, 2)).mul_(-0.5)                   # dL/dGs (symmetric)
            del AW
            S3, pb3, vb3 = S.reshape(N, 1, n), pb.reshape(N, 1, n), vb.reshape(N, 1, dV)
            inv3 = inv.reshape(N, 1, n)
            invc = inv3.transpose(1, 2)
            g_inv = None
            if ctx.needs_input_grad[5]:
                M1 = (G.reshape(N, n, n) - S3.transpose(1, 2) * pb3) * E            # Gc * E
                g_inv = torch.bmm(inv3, M1) + torch.bmm(inv3, M1.transpose(1, 2))
                del M1
                Cc = C.reshape(N, n, dV) - S3.transpose(1, 2) * vb3
                g_inv = g_inv + (gR3 * Cc).sum(-1).unsqueeze(1)
                del Cc
            gGc = E.mul_(invc).mul_(inv3)
            gCc = gR3 * invc
            gS = -(torch.bmm(pb3, gGc.transpose(1, 2)) + torch.bmm(vb3, gCc.transpose(1, 2)))
            gpb = -torch.bmm(S3, gGc)
            gvb = -torch.bmm(S3, gCc)
        dW = (W0 - W) if W0 is not None else -W
        glam = _sum_to((gR * dW).sum(-1, keepdim=True).sum(-2, keepdim=True), ctx.lam_shape)
        gW0 = _sum_to(lam * gR, ctx.W0_shape) if W0 is not None else None
        return (gGc.view(G.shape), gCc.view(C.shape), gS.view(S.shape), gpb.view(pb.shape), gvb.view(vb.shape),
                g_inv.view(inv.shape) if g_inv is not None else None, glam, gW0, None)

    @staticmethod
    def _backward_triton(ctx, AW, G, C, S, pb, vb, inv, lam, W0, W3, gR, gR3, lead, N, n, dV):
        S2, PB2 = S.reshape(N, n).contiguous(), pb.reshape(N, n).contiguous()
        VB2, INV2 = vb.reshape(N, dV).contiguous(), inv.reshape(N, n).contiguous()
        GG, GINV, GS, GPB = _TR.bwd_g(AW, G.reshape(N, n, n).contiguous(), S2, PB2, INV2)
        del AW
        W02 = W0.reshape(n, dV).contiguous() if W0 is not None else None
        GC, OA, OB, OL = _TR.bwd_c(gR3.contiguous(), C.reshape(N, n, dV).contiguous(), VB2, INV2,
                                   W3.contiguous(), W02)
        g_inv = (GINV + OA - S2 * OB) if ctx.needs_input_grad[5] else None
        gS = GS - INV2 * OB
        gvb = -torch.bmm((S2 * INV2).unsqueeze(1), gR3)
        glam = _sum_to(OL.sum(-1).view(*lead, 1, 1), ctx.lam_shape)
        gW0 = _sum_to(lam * gR, ctx.W0_shape) if W0 is not None else None
        return (GG.view(G.shape), GC.view(C.shape), gS.view(S.shape), GPB.view(pb.shape), gvb.view(vb.shape),
                g_inv.view(inv.shape) if g_inv is not None else None, glam, gW0, None)


def centred_solve(G, C, S, phi_bar, v_bar, inv, lam, W0=None, jitter=1e-4):
    """(M = L^-1 detached, W) of the centred, scaled ridge regression; see the module docstring.

    G [B, H, dF, dF], C [B, H, dF, dV], S / phi_bar / inv [B, H, 1, dF], v_bar [B, H, 1, dV] (fp32);
    lam broadcastable to [B, H, 1, 1]; W0 broadcastable to [B, H, dF, dV] or None."""
    return _CentredSolve.apply(G, C, S, phi_bar, v_bar, inv, lam, W0, float(jitter))
