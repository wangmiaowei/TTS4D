"""Triton kernels for the dF x dF / dF x dV elementwise parts of ``solve.centred_solve`` (fp32 throughout).

Every kernel has a PyTorch twin in ``solve.py``; ``available()`` is False (and the twins are used) when Triton is
missing or a kernel failed once.  Matrices are batched [N, n, n] / [N, n, dV], vectors [N, n].

  pform:   P  = 0.5 (Gs + Gs^T) + lam I,   Gs = (G - S^T pb) * inv^T inv
  rform:   R  = (C - S^T vb) * inv^T + lam W0
  bwd_g:   E  = -0.5 (AW + AW^T),  gG = E * inv^T inv  (stored), and per row
           ginv_i = sum_j E_ij (Gc_ij + Gc_ji) inv_j,   gS_i = -sum_j gG_ij pb_j,   gpb_i = -sum_j gG_ij S_j
  bwd_c:   gC = gR * inv^T  (stored), and per row
           a_i = sum_c gR_ic C_ic,   b_i = sum_c gR_ic vb_c,   l_i = sum_c gR_ic (W0_ic - W_ic)
"""
import warnings

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    triton = tl = None
    HAS_TRITON = False

_BROKEN = False


def available():
    return HAS_TRITON and not _BROKEN


def mark_broken(ex):
    global _BROKEN
    if not _BROKEN:
        warnings.warn(f"solve Triton kernels disabled after error: {type(ex).__name__}: {ex}")
    _BROKEN = True


if HAS_TRITON:
    @triton.jit
    def _pform_kernel(G, S, PB, INV, LAM, P, n, BM: tl.constexpr):
        b = tl.program_id(0)
        pi = tl.program_id(1)
        pj = tl.program_id(2)
        ri = pi * BM + tl.arange(0, BM)
        rj = pj * BM + tl.arange(0, BM)
        mi = ri < n
        mj = rj < n
        m2 = mi[:, None] & mj[None, :]
        base = b.to(tl.int64) * n * n
        vb = b.to(tl.int64) * n
        g_ij = tl.load(G + base + ri[:, None] * n + rj[None, :], mask=m2, other=0.0)
        # tile (j, i) read row-major, transposed in registers
        g_ji = tl.trans(tl.load(G + base + rj[:, None] * n + ri[None, :], mask=mj[:, None] & mi[None, :], other=0.0))
        s_i = tl.load(S + vb + ri, mask=mi, other=0.0)
        s_j = tl.load(S + vb + rj, mask=mj, other=0.0)
        p_i = tl.load(PB + vb + ri, mask=mi, other=0.0)
        p_j = tl.load(PB + vb + rj, mask=mj, other=0.0)
        v_i = tl.load(INV + vb + ri, mask=mi, other=0.0)
        v_j = tl.load(INV + vb + rj, mask=mj, other=0.0)
        gc_ij = ((g_ij - s_i[:, None] * p_j[None, :]) * v_i[:, None]) * v_j[None, :]
        gc_ji = ((g_ji - s_j[None, :] * p_i[:, None]) * v_j[None, :]) * v_i[:, None]
        p = (gc_ij + gc_ji) * 0.5
        lam = tl.load(LAM + b)
        p = p + tl.where(ri[:, None] == rj[None, :], lam, 0.0)
        tl.store(P + base + ri[:, None] * n + rj[None, :], p, mask=m2)

    @triton.jit
    def _rform_kernel(C, S, VB, INV, LAM, W0, R, n, dV, HAS_W0: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        b = tl.program_id(0)
        pi = tl.program_id(1)
        pj = tl.program_id(2)
        ri = pi * BM + tl.arange(0, BM)
        rc = pj * BN + tl.arange(0, BN)
        mi = ri < n
        mc = rc < dV
        m2 = mi[:, None] & mc[None, :]
        base = b.to(tl.int64) * n * dV
        off = ri[:, None] * dV + rc[None, :]
        c = tl.load(C + base + off, mask=m2, other=0.0)
        s_i = tl.load(S + b.to(tl.int64) * n + ri, mask=mi, other=0.0)
        v_c = tl.load(VB + b.to(tl.int64) * dV + rc, mask=mc, other=0.0)
        iv = tl.load(INV + b.to(tl.int64) * n + ri, mask=mi, other=0.0)
        r = (c - s_i[:, None] * v_c[None, :]) * iv[:, None]
        if HAS_W0:
            lam = tl.load(LAM + b)
            w0 = tl.load(W0 + off, mask=m2, other=0.0)
            r = r + lam * w0
        tl.store(R + base + off, r, mask=m2)

    @triton.jit
    def _bwd_g_kernel(AW, G, S, PB, INV, GG, GINV, GS, GPB, n, BM: tl.constexpr, BN: tl.constexpr):
        b = tl.program_id(0)
        pi = tl.program_id(1)
        ri = pi * BM + tl.arange(0, BM)
        mi = ri < n
        base = b.to(tl.int64) * n * n
        vb = b.to(tl.int64) * n
        s_i = tl.load(S + vb + ri, mask=mi, other=0.0)
        p_i = tl.load(PB + vb + ri, mask=mi, other=0.0)
        v_i = tl.load(INV + vb + ri, mask=mi, other=0.0)
        acc_inv = tl.zeros((BM,), dtype=tl.float32)
        acc_s = tl.zeros((BM,), dtype=tl.float32)
        acc_p = tl.zeros((BM,), dtype=tl.float32)
        for j0 in range(0, n, BN):
            rj = j0 + tl.arange(0, BN)
            mj = rj < n
            m2 = mi[:, None] & mj[None, :]
            mt = mj[:, None] & mi[None, :]
            a_ij = tl.load(AW + base + ri[:, None] * n + rj[None, :], mask=m2, other=0.0)
            a_ji = tl.trans(tl.load(AW + base + rj[:, None] * n + ri[None, :], mask=mt, other=0.0))
            e = (a_ij + a_ji) * -0.5
            s_j = tl.load(S + vb + rj, mask=mj, other=0.0)
            p_j = tl.load(PB + vb + rj, mask=mj, other=0.0)
            v_j = tl.load(INV + vb + rj, mask=mj, other=0.0)
            gg = (e * v_i[:, None]) * v_j[None, :]
            g_ij = tl.load(G + base + ri[:, None] * n + rj[None, :], mask=m2, other=0.0)
            g_ji = tl.trans(tl.load(G + base + rj[:, None] * n + ri[None, :], mask=mt, other=0.0))
            gcs = (g_ij - s_i[:, None] * p_j[None, :]) + (g_ji - s_j[None, :] * p_i[:, None])
            acc_inv += tl.sum(e * gcs * v_j[None, :], axis=1)
            tl.store(GG + base + ri[:, None] * n + rj[None, :], gg, mask=m2)
            acc_s += tl.sum(gg * p_j[None, :], axis=1)
            acc_p += tl.sum(gg * s_j[None, :], axis=1)
        tl.store(GINV + vb + ri, acc_inv, mask=mi)
        tl.store(GS + vb + ri, -acc_s, mask=mi)
        tl.store(GPB + vb + ri, -acc_p, mask=mi)

    @triton.jit
    def _bwd_c_kernel(GR, C, VB, INV, W, W0, GC, OA, OB, OL, n, dV,
                      HAS_W0: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        b = tl.program_id(0)
        pi = tl.program_id(1)
        ri = pi * BM + tl.arange(0, BM)
        mi = ri < n
        base = b.to(tl.int64) * n * dV
        iv = tl.load(INV + b.to(tl.int64) * n + ri, mask=mi, other=0.0)
        acc_a = tl.zeros((BM,), dtype=tl.float32)
        acc_b = tl.zeros((BM,), dtype=tl.float32)
        acc_l = tl.zeros((BM,), dtype=tl.float32)
        for c0 in range(0, dV, BN):
            rc = c0 + tl.arange(0, BN)
            mc = rc < dV
            m2 = mi[:, None] & mc[None, :]
            off = ri[:, None] * dV + rc[None, :]
            g = tl.load(GR + base + off, mask=m2, other=0.0)
            c = tl.load(C + base + off, mask=m2, other=0.0)
            v = tl.load(VB + b.to(tl.int64) * dV + rc, mask=mc, other=0.0)
            w = tl.load(W + base + off, mask=m2, other=0.0)
            acc_a += tl.sum(g * c, axis=1)
            acc_b += tl.sum(g * v[None, :], axis=1)
            if HAS_W0:
                w0 = tl.load(W0 + off, mask=m2, other=0.0)
                acc_l += tl.sum(g * (w0 - w), axis=1)
            else:
                acc_l += tl.sum(g * (-w), axis=1)
            tl.store(GC + base + off, g * iv[:, None], mask=m2)
        tl.store(OA + b.to(tl.int64) * n + ri, acc_a, mask=mi)
        tl.store(OB + b.to(tl.int64) * n + ri, acc_b, mask=mi)
        tl.store(OL + b.to(tl.int64) * n + ri, acc_l, mask=mi)


def pform(G3, S2, PB2, INV2, lam1):
    """G3 [N, n, n], S2 / PB2 / INV2 [N, n], lam1 [N] -> P [N, n, n]."""
    N, n, _ = G3.shape
    P = torch.empty_like(G3)
    BM = 64
    _pform_kernel[(N, triton.cdiv(n, BM), triton.cdiv(n, BM))](G3, S2, PB2, INV2, lam1, P, n, BM=BM, num_warps=4)
    return P


def rform(C3, S2, VB2, INV2, lam1, W0_2):
    """C3 [N, n, dV]; W0_2 [n, dV] shared by all N, or None -> R [N, n, dV]."""
    N, n, dV = C3.shape
    R = torch.empty_like(C3)
    BM, BN = 32, 128
    _rform_kernel[(N, triton.cdiv(n, BM), triton.cdiv(dV, BN))](
        C3, S2, VB2, INV2, lam1, W0_2 if W0_2 is not None else C3, R, n, dV,
        HAS_W0=W0_2 is not None, BM=BM, BN=BN, num_warps=4)
    return R


def bwd_g(AW3, G3, S2, PB2, INV2):
    N, n, _ = AW3.shape
    GG = torch.empty_like(AW3)
    GINV, GS, GPB = torch.empty_like(S2), torch.empty_like(S2), torch.empty_like(S2)
    BM, BN = 32, 64
    _bwd_g_kernel[(N, triton.cdiv(n, BM))](AW3, G3, S2, PB2, INV2, GG, GINV, GS, GPB, n, BM=BM, BN=BN, num_warps=4)
    return GG, GINV, GS, GPB


def bwd_c(GR3, C3, VB2, INV2, W3, W0_2):
    N, n, dV = GR3.shape
    GC = torch.empty_like(GR3)
    OA, OB, OL = GR3.new_empty(N, n), GR3.new_empty(N, n), GR3.new_empty(N, n)
    BM, BN = 32, 128
    _bwd_c_kernel[(N, triton.cdiv(n, BM))](GR3, C3, VB2, INV2, W3, W0_2 if W0_2 is not None else W3,
                                           GC, OA, OB, OL, n, dV, HAS_W0=W0_2 is not None, BM=BM, BN=BN, num_warps=4)
    return GC, OA, OB, OL
