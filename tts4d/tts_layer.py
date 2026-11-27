"""The test-time solver (TTS) layer.

All input tokens of a scene are written into one fast weight W by a closed-form, centred and scaled ridge
regression from keys to values, and every token (input or query) reads W.  Because the write only adds up
sufficient statistics over tokens, the solved state is

  * invariant to the order of the input tokens,
  * of fixed size (independent of the number of input views), and
  * equipped with a closed-form posterior variance u(q), which is the epistemic part of our uncertainty.

Notation (per head; the trained model uses a single head of width d = 768 and feature width dF = 1536):

    phi(x)  = SiLU(x W0) * (x W2) / ||.||_2                      feature map of a key / query        [dF]
    a_n     = softplus(w_a^T y_n + b_a)                          learned evidence weight of token n
    A = sum a_n,  S = sum a_n phi_n,  T = sum a_n v_n,
    G = sum a_n phi_n phi_n^T,  C = sum a_n phi_n v_n^T          the five sufficient statistics
    phi_bar = S / A,  v_bar = T / A
    G_c = G - S^T S / A,  C_c = C - S^T v_bar                    centred moments (rank-one corrections)
    D   = diag(G_c)^(1/2)                                         scaling: G_s = D^-1 G_c D^-1 has unit diagonal
    W   = (G_s + lam I)^-1 (D^-1 C_c + lam W_prior)               the solve (W_prior is learned)
    out(q) = v_bar + D^-1 (phi(q) - phi_bar) W                    the read
    u(q)   = || L^-1 D^-1 (phi(q) - phi_bar) ||^2                 with L L^T = G_s + lam I

Here W0, W2, W_prior (``w0``, ``w2``, ``w1``), the evidence head and lam are slow weights learned in training;
everything else is computed at test time from the input views.

Two implementations of the same layer:

  * the reference path (``fast=False``): the plain PyTorch code below, fp32 throughout;
  * the fast path (``fast=True``, the default under bf16 autocast on a GPU with Triton): the operators of
    ``tts4d.kernels`` -- token-sized GEMMs with bf16 operands and fp32 accumulation, fused Triton kernels for the
    feature map tail, the q / k / v preparation and the elementwise parts of the solve, and hand-written backward
    passes.  The accumulators, centring, scaling and solve stay fp32.  This is the path the model was trained with.

Both paths are differentiable (the layer can be fine-tuned); ``u(q)`` never carries a gradient.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from . import kernels as K
from .kernels import memory as _mem
from .kernels.qkv import qkv_prep, qkv_prep_ok, rope_freqs
from .kernels.solve import centred_solve, robust_cholesky  # noqa: F401  (robust_cholesky: public helper)
from .rope import RoPE2DTime


def _inv_softplus(y):
    return y + math.log(-math.expm1(-y))


def feature_map(x, w0, w2, eps=1e-6):
    """phi(x) = SiLU(x W0) * (x W2), L2-normalised.  x: [B, H, N, d] -> [B, H, N, dF]."""
    phi = F.silu(x @ w0) * (x @ w2)
    return phi / (phi.norm(dim=-1, keepdim=True) + eps)


class SolvedState:
    """The solved memory of one TTS layer: everything a query needs to read it (fixed size)."""

    def __init__(self, W, phi_bar, v_bar, inv_scale, L_inv, lam):
        self.W, self.phi_bar, self.v_bar, self.inv_scale, self.L_inv, self.lam = W, phi_bar, v_bar, inv_scale, L_inv, lam

    def numel(self):
        return sum(t.numel() for t in (self.W, self.phi_bar, self.v_bar, self.inv_scale, self.L_inv))


class TTSLayer(nn.Module):
    """Test-time solver layer.  Parameter names match the released checkpoint.

    Args:
        dim: token width.  head_dim: width of a head (768 = one head).  inter_multi: dF = head_dim * inter_multi.
        ridge_init / ridge_min: lam = softplus(raw_ridge) + ridge_min, one per head.
        scale_eps: floor of the scaling, relative to the largest diagonal entry of G_c.
        rope_frequency: base frequency of the 2-D + time rotary embedding of q and k.
        chunk: number of tokens processed at once when accumulating / reading (memory only, the result is exact).
        fast: True = the fast path (``tts4d.kernels``), False = the PyTorch reference path, None (default) = the fast
            path whenever the layer runs on a GPU under bf16 autocast and Triton is installed.
    """

    def __init__(self, dim=768, head_dim=768, inter_multi=2, ridge_init=10.0, ridge_min=1e-3, base_evidence=1.0,
                 scale_eps=1e-3, rope_frequency=100.0, chunk=32768, fast=None):
        super().__init__()
        self.dim, self.head_dim = dim, head_dim
        self.num_heads = dim // head_dim
        self.dF = int(head_dim * inter_multi)
        H, d, dF = self.num_heads, head_dim, self.dF
        self.w0 = nn.Parameter(torch.randn(H, d, dF) * math.sqrt(2.0 / d))      # gate branch of the feature map
        self.w2 = nn.Parameter(torch.randn(H, d, dF) * math.sqrt(2.0 / d))      # linear branch of the feature map
        self.w1 = nn.Parameter(torch.randn(H, dF, d) * math.sqrt(2.0 / dF))     # prior readout W_prior
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.o_norm = nn.RMSNorm(head_dim, eps=1e-5)
        self.raw_ridge = nn.Parameter(torch.full((H,), _inv_softplus(ridge_init)))
        self.ridge_min = float(ridge_min)
        self.evidence_fc = nn.Linear(dim, H)
        self.base_evidence_inv = _inv_softplus(base_evidence)
        self.scale_eps = float(scale_eps)
        self.rope = RoPE2DTime(frequency=rope_frequency)
        self.chunk = int(chunk)
        # evidence-aware ridge: lam_eff = lam * max(N_views / n_ref, 1) ** (-gamma); gamma is set per layer from the config
        self.law_gamma, self.law_nref = 0.0, 32.0
        self.fast = fast
        self._freq = {}

    def use_fast(self, x):
        """Whether this call takes the fast path (see ``fast``)."""
        if self.fast is not None:
            return bool(self.fast)
        return x.is_cuda and K.HAS_TRITON and K.autocast_bf16(x.device.type)

    # ------------------------------------------------------------------ projections
    def project(self, x, pos):
        """q, k, v [B, H, N, d] (fp32) and evidence a [B, H, N, 1] of the tokens x [B, N, dim]."""
        B, N, _ = x.shape
        H, d = self.num_heads, self.head_dim
        qkv = F.silu(self.to_qkv(x)).float()
        q, k, v = qkv.view(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-5)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-5)
        q, k = self.rope(q, pos), self.rope(k, pos)               # rotation: keeps the unit norm
        with torch.autocast(device_type=x.device.type, enabled=False):
            a = F.softplus(self.evidence_fc(x.float()) + self.base_evidence_inv)   # [B, N, H]
        return q, k, v, a.permute(0, 2, 1).unsqueeze(-1)

    def project_fast(self, x, pos):
        """``project`` with SiLU, the q / k L2 norm and RoPE fused into one Triton kernel; q, k, v come out in bf16.
        Falls back to ``project`` when the projection is not a bf16 tensor on the GPU (no autocast)."""
        y = self.to_qkv(x)
        if not qkv_prep_ok(y, self.head_dim):
            return self.project(x, pos)
        key = str(y.device)
        if key not in self._freq:
            self._freq[key] = rope_freqs(self.head_dim, self.rope.base, y.device)
        q, k, v = qkv_prep(y, self.num_heads, self.head_dim, pos, self._freq[key], eps=1e-5)
        with torch.autocast(device_type=x.device.type, enabled=False):
            a = F.softplus(self.evidence_fc(x.float()) + self.base_evidence_inv)   # [B, N, H]
        return q, k, v, a.permute(0, 2, 1).unsqueeze(-1)

    @property
    def ridge(self):
        return F.softplus(self.raw_ridge.float()) + self.ridge_min

    def effective_ridge(self, n_views):
        """lam [1, H, 1, 1], optionally relaxed as evidence grows (evidence-aware ridge)."""
        lam = self.ridge.view(1, self.num_heads, 1, 1)
        if self.law_gamma:
            lam = lam * max(n_views / self.law_nref, 1.0) ** (-self.law_gamma)
        return lam

    def _spans(self, n):
        return [(s, min(n, s + self.chunk)) for s in range(0, n, self.chunk)]

    # ------------------------------------------------------------------ write + solve (reference)
    def solve(self, k, v, a, n_views):
        """Closed-form write of the input tokens.  k, v: [B, H, N_in, d], a: [B, H, N_in, 1] -> SolvedState."""
        B, H = k.shape[:2]
        dF, dV = self.dF, self.head_dim
        w0, w2 = self.w0.float(), self.w2.float()
        spans = self._spans(k.shape[2])
        # Mathematically G_c = G - S^T S / A and C_c = C - S^T v_bar (all five statistics are plain sums over tokens,
        # hence order invariant).  The features share a strong mean direction, so in floating point we accumulate the
        # centred products directly: one pass for (A, S, T), one for (G_c, C_c).  Same algebra, no cancellation.
        S = k.new_zeros(B, H, 1, dF); T = k.new_zeros(B, H, 1, dV); A = k.new_zeros(B, H, 1, 1)
        for s, e in spans:
            phi, vv, aa = feature_map(k[:, :, s:e], w0, w2), v[:, :, s:e], a[:, :, s:e]
            S += (phi * aa).sum(-2, keepdim=True); T += (vv * aa).sum(-2, keepdim=True); A += aa.sum(-2, keepdim=True)
        Ac = A.clamp_min(1e-12)
        phi_bar, v_bar = S / Ac, T / Ac
        Gc = k.new_zeros(B, H, dF, dF); Cc = k.new_zeros(B, H, dF, dV)
        for s, e in spans:
            dphi = feature_map(k[:, :, s:e], w0, w2) - phi_bar
            adphi = dphi * a[:, :, s:e]
            Gc += adphi.transpose(-1, -2) @ dphi
            Cc += adphi.transpose(-1, -2) @ (v[:, :, s:e] - v_bar)
        # scale: G_s = D^-1 G_c D^-1 has unit diagonal (a correlation matrix)
        dg = Gc.diagonal(dim1=-2, dim2=-1).clamp_min(0)
        dch = dg.clamp_min(self.scale_eps * dg.amax(-1, keepdim=True)).sqrt().clamp_min(1e-12)
        inv = (1.0 / dch).unsqueeze(-2)                           # [B, H, 1, dF]
        Gs = Gc * inv.transpose(-1, -2) * inv
        Cs = Cc * inv.transpose(-1, -2)
        lam = self.effective_ridge(n_views)                     # the ridge, relaxed as evidence grows
        # solve with the triangular inverse, so that the read and u(q) are plain matrix products
        eye = torch.eye(dF, device=k.device, dtype=k.dtype)
        Lc = robust_cholesky(0.5 * (Gs + Gs.transpose(-1, -2)) + lam * eye)
        L_inv = torch.linalg.solve_triangular(Lc, eye.expand(Lc.shape), upper=False)
        W_prior = self.w1.float().unsqueeze(0).expand_as(Cs)
        P_inv = lambda X: L_inv.transpose(-1, -2) @ (L_inv @ X)   # noqa: E731   (G_s + lam I)^-1 X
        W = P_inv(Cs) + lam * P_inv(W_prior)
        return SolvedState(W, phi_bar, v_bar, inv, L_inv, lam)

    # ------------------------------------------------------------------ write + solve (fast)
    def solve_fast(self, k, v, a, n_views):
        """``solve`` on the fast path: the statistics from bf16 GEMM operands (fp32 accumulation, one pass over the
        tokens), the centring and scaling of the dF x dF moments fused into the solve.  Same SolvedState."""
        G = C = S = T = A = None
        for s, e in self._spans(k.shape[2]):
            phi = _mem.feature_map(k[:, :, s:e], self.w0, self.w2)
            st = _mem.moments(phi, v[:, :, s:e], a[:, :, s:e])
            G, C, S, T, A = st if G is None else tuple(x + y for x, y in zip((G, C, S, T, A), st))
        Ac = A.clamp_min(1e-12)
        phi_bar, v_bar = S / Ac, T / Ac
        # the scaling needs only diag(G_c) = diag(G) - S * phi_bar: vectors, by autograd
        dg = (G.diagonal(dim1=-2, dim2=-1) - (S * phi_bar).squeeze(-2)).clamp_min(0)
        dch = dg.clamp_min(self.scale_eps * dg.amax(-1, keepdim=True)).sqrt().clamp_min(1e-12)
        inv = (1.0 / dch).unsqueeze(-2)                           # [B, H, 1, dF]
        lam = self.effective_ridge(n_views)
        L_inv, W = centred_solve(G, C, S, phi_bar, v_bar, inv, lam, self.w1.float().unsqueeze(0))
        return SolvedState(W, phi_bar, v_bar, inv, L_inv, lam)

    # ------------------------------------------------------------------ read
    def read(self, state, q, want_u=False):
        """out [B, H, N, d] and (optionally) the epistemic scalar u [B, H, N] of the queries q [B, H, N, d]."""
        w0, w2 = self.w0.float(), self.w2.float()
        outs, us = [], []
        for s, e in self._spans(q.shape[2]):
            pc = (feature_map(q[:, :, s:e], w0, w2) - state.phi_bar) * state.inv_scale
            outs.append(state.v_bar + pc @ state.W)
            if want_u:
                with torch.no_grad():
                    z = pc.detach() @ state.L_inv.transpose(-1, -2)
                    us.append((z * z).sum(-1))
        return torch.cat(outs, 2), (torch.cat(us, 2) if want_u else None)

    def read_fast(self, state, q, want_u=False):
        """``read`` on the fast path: feature map and read GEMMs with bf16 operands and fp32 outputs.  u(q) =
        ||L^-1 pc||^2 is a bf16 GEMM too: a sum of squares, so the bf16 operands cost only their own rounding."""
        outs, us = [], []
        for s, e in self._spans(q.shape[2]):
            pc = (_mem.feature_map(q[:, :, s:e], self.w0, self.w2) - state.phi_bar) * state.inv_scale
            outs.append(state.v_bar + _mem.read_mm(pc, state.W))
            if want_u:
                with torch.no_grad():
                    z = _mem.mm_f32(pc.detach(), state.L_inv.transpose(-1, -2))
                    us.append((z * z).sum(-1))
        return torch.cat(outs, 2), (torch.cat(us, 2) if want_u else None)

    def output(self, o, dtype):
        """o_norm per head, merge heads, output projection.  o: [B, H, N, d] -> [B, N, dim]."""
        B, H, N, d = o.shape
        o = self.o_norm(o.to(dtype))
        return self.c_proj(o.permute(0, 2, 1, 3).reshape(B, N, H * d))

    def forward(self, x, ctx, state=None):
        """Two uses, which together are exactly one joint forward over inputs and queries:

        * encode (state is None): x are the INPUT tokens; they are written into the memory (solve) and read it.
          Returns (y, None, state).
        * render (state given): x are QUERY tokens; they only read the solved state (queries never write).
          Returns (y, u, state) with u [B, N] the epistemic scalar averaged over heads.
        """
        if self.use_fast(x):
            q, k, v, a = self.project_fast(x, ctx["pos"])
            with torch.autocast(device_type=x.device.type, enabled=False):   # the kernels set their own precision
                if state is None:
                    state = self.solve_fast(k, v, a, ctx["n_views"])
                    o, u = self.read_fast(state, q)
                else:
                    o, u = self.read_fast(state, q, want_u=True)
        else:
            q, k, v, a = self.project(x, ctx["pos"])
            if state is None:
                state = self.solve(k, v, a, ctx["n_views"])
                o, u = self.read(state, q)
            else:
                o, u = self.read(state, q, want_u=True)
        return self.output(o, x.dtype), (u.mean(1) if u is not None else None), state
