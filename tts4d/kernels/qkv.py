"""q / k / v preparation of the TTS layer in one Triton kernel (forward and backward).

On the ``to_qkv`` output y = [q | k | v] (bf16, channel layout "(qkv h d)") the layer computes

    s = SiLU(y);   q = s_q / (||s_q|| + eps),  q = RoPE(q, pos)   (k likewise);   v = s_v

``RoPE2DTime`` on a D-wide head (see ``tts4d/rope.py``): channels [0, S) carry the 2-D spatial rotation (row band
[0, S/2), column band [S/2, S)), channels [S, D) the time rotation (positions in seconds).  Every band rotates pairs
(j, j + half): out_j = x_j cos - x_{j+h} sin, out_{j+h} = x_{j+h} cos + x_j sin, angle = position * inv_freq.
One program per (token, q|k|v, head); the backward is the transposed rotation, the norm gradient and the SiLU
gradient in one pass.

``qkv_prep_ref`` is the same computation in PyTorch (used as the fallback and as the test reference).
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

if HAS_TRITON:
    @triton.jit
    def _rope_meta(j, S: tl.constexpr, HS: tl.constexpr, HT: tl.constexpr):
        # band-local pair partner, sign of the partner term, frequency index, band
        in_t = j >= S
        js = tl.where(in_t, 0, j)
        band_s = js // (2 * HS)                       # 0 row band, 1 column band
        loc_s = js - band_s * (2 * HS)
        first_s = loc_s < HS
        pj_s = tl.where(first_s, js + HS, js - HS)
        fi_s = tl.where(first_s, loc_s, loc_s - HS)
        jt = tl.where(in_t, j - S, 0)
        first_t = jt < HT
        pj_t = S + tl.where(first_t, jt + HT, jt - HT)
        fi_t = HS + tl.where(first_t, jt, jt - HT)
        pj = tl.where(in_t, pj_t, pj_s)
        fi = tl.where(in_t, fi_t, fi_s)
        first = tl.where(in_t, first_t, first_s)
        sgn = tl.where(first, -1.0, 1.0)
        band = tl.where(in_t, 2, band_s)
        return pj, fi, sgn, band

    @triton.jit
    def _qkv_fwd(Y, POS, FREQ, QO, KO, VO, NRM, L, NB, s_yb, s_yl, s_pb, s_pl, eps,
                 H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, HS: tl.constexpr, HT: tl.constexpr,
                 ROPE: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        wh = tl.program_id(1)
        which = wh // H
        h = wh % H
        b = row // L
        l = row % L
        j = tl.arange(0, BLOCK)
        msk = j < D
        base = Y + b * s_yb + l * s_yl + (which * H + h) * D
        y = tl.load(base + j, mask=msk, other=0.0).to(tl.float32)
        s = y * tl.sigmoid(y)                           # 0 on masked lanes: norms unchanged
        orow = ((b * H + h) * L + l) * D
        if which == 2:
            tl.store(VO + orow + j, s.to(tl.bfloat16), mask=msk)
        else:
            n = tl.sqrt(tl.sum(s * s, axis=0))
            den = n + eps
            xn = s / den
            if ROPE:
                pj, fi, sgn, band = _rope_meta(j, S, HS, HT)
                yp = tl.load(base + pj, mask=msk, other=0.0).to(tl.float32)
                xp = yp * tl.sigmoid(yp) / den
                p0 = tl.load(POS + b * s_pb + l * s_pl)
                p1 = tl.load(POS + b * s_pb + l * s_pl + 1)
                p2 = tl.load(POS + b * s_pb + l * s_pl + 2)
                p = tl.where(band == 0, p0, tl.where(band == 1, p1, p2))
                ang = p * tl.load(FREQ + fi, mask=msk, other=0.0)
                out = xn * tl.cos(ang) + sgn * xp * tl.sin(ang)
            else:
                out = xn
            if which == 0:
                tl.store(QO + orow + j, out.to(QO.dtype.element_ty), mask=msk)
            else:
                tl.store(KO + orow + j, out.to(KO.dtype.element_ty), mask=msk)
            tl.store(NRM + (which * H + h) * NB + row, n)

    @triton.jit
    def _qkv_bwd(Y, POS, FREQ, GQ, GK, GV, NRM, DY, L, NB, s_yb, s_yl, s_pb, s_pl, eps,
                 H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, HS: tl.constexpr, HT: tl.constexpr,
                 ROPE: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        wh = tl.program_id(1)
        which = wh // H
        h = wh % H
        b = row // L
        l = row % L
        j = tl.arange(0, BLOCK)
        msk = j < D
        off = b * s_yb + l * s_yl + (which * H + h) * D
        y = tl.load(Y + off + j, mask=msk, other=0.0).to(tl.float32)
        sig = tl.sigmoid(y)
        s = y * sig
        dsilu = sig * (1.0 + y * (1.0 - sig))
        orow = ((b * H + h) * L + l) * D
        if which == 2:
            gv = tl.load(GV + orow + j, mask=msk, other=0.0).to(tl.float32)
            tl.store(DY + off + j, (gv * dsilu).to(tl.bfloat16), mask=msk)
        else:
            if which == 0:
                g = tl.load(GQ + orow + j, mask=msk, other=0.0).to(tl.float32)
            else:
                g = tl.load(GK + orow + j, mask=msk, other=0.0).to(tl.float32)
            n = tl.load(NRM + (which * H + h) * NB + row)
            den = n + eps
            if ROPE:
                pj, fi, sgn, band = _rope_meta(j, S, HS, HT)
                if which == 0:
                    gp = tl.load(GQ + orow + pj, mask=msk, other=0.0).to(tl.float32)
                else:
                    gp = tl.load(GK + orow + pj, mask=msk, other=0.0).to(tl.float32)
                p0 = tl.load(POS + b * s_pb + l * s_pl)
                p1 = tl.load(POS + b * s_pb + l * s_pl + 1)
                p2 = tl.load(POS + b * s_pb + l * s_pl + 2)
                p = tl.where(band == 0, p0, tl.where(band == 1, p1, p2))
                ang = p * tl.load(FREQ + fi, mask=msk, other=0.0)
                # transpose of the rotation: d xn_j = g_j cos - sgn_j sin g_{p(j)}
                gx = g * tl.cos(ang) - sgn * tl.sin(ang) * gp
            else:
                gx = g
            # xn = s / den:  ds = gx / den - s (gx . s) / (den^2 n)
            dot = tl.sum(gx * s, axis=0)
            coef = tl.where(n > 0, dot / (den * den * tl.where(n > 0, n, 1.0)), 0.0)
            ds = gx / den - s * coef
            tl.store(DY + off + j, (ds * dsilu).to(tl.bfloat16), mask=msk)


def _split(D):
    """Spatial / time channel split of ``RoPE2DTime`` (kept in sync with tts4d/rope.py)."""
    t = (D // 4) & ~1
    s = D - t
    s -= s % 4
    return s, D - s


def rope_freqs(D, base, device):
    """[spatial inv_freq (S/4) | time inv_freq (T/2)] of a D-wide head, as ``RoPE2DTime`` builds them."""
    S, T = _split(D)
    ax = S // 2
    fs = 1.0 / (base ** (torch.arange(0, ax, 2, device=device, dtype=torch.float32) / float(ax)))
    if T > 0:
        ft = 1.0 / (100.0 ** (torch.arange(0, T, 2, device=device, dtype=torch.float32) / float(T)))
        fs = torch.cat([fs, ft])
    return fs.contiguous()


class _QKVPrep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, pos, freq, H, D, eps):
        B, L, _ = y.shape
        S = _split(D)[0]
        q = torch.empty(B, H, L, D, device=y.device, dtype=BF16)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        nrm = torch.empty(2 * H, B * L, device=y.device, dtype=torch.float32)
        rope = pos is not None
        posx = pos if rope else nrm
        sp = (pos.stride(0), pos.stride(1)) if rope else (0, 0)
        frq = freq if rope else nrm
        _qkv_fwd[(B * L, 3 * H)](y, posx, frq, q, k, v, nrm, L, B * L, y.stride(0), y.stride(1), sp[0], sp[1], eps,
                                 H=H, D=D, S=S, HS=max(S // 4, 1), HT=max((D - S) // 2, 1), ROPE=rope,
                                 BLOCK=triton.next_power_of_2(D), num_warps=4)
        ctx.save_for_backward(y, pos if rope else None, freq if rope else None, nrm)
        ctx.cfg = (H, D, S, eps, rope, sp)
        return q, k, v

    @staticmethod
    def backward(ctx, gq, gk, gv):
        y, pos, freq, nrm = ctx.saved_tensors
        H, D, S, eps, rope, sp = ctx.cfg
        B, L, _ = y.shape
        zeros = None

        def fix(g):
            nonlocal zeros
            if g is None:
                if zeros is None:
                    zeros = torch.zeros(B, H, L, D, device=y.device, dtype=BF16)
                return zeros
            return g.contiguous()
        gq, gk, gv = fix(gq), fix(gk), fix(gv)
        dy = torch.empty_like(y)
        posx = pos if rope else nrm
        frq = freq if rope else nrm
        _qkv_bwd[(B * L, 3 * H)](y, posx, frq, gq, gk, gv, nrm, dy, L, B * L, y.stride(0), y.stride(1), sp[0], sp[1],
                                 eps, H=H, D=D, S=S, HS=max(S // 4, 1), HT=max((D - S) // 2, 1), ROPE=rope,
                                 BLOCK=triton.next_power_of_2(D), num_warps=4)
        return dy, None, None, None, None, None


def qkv_prep_ok(y, D):
    """The Triton kernel covers a contiguous bf16 [B, L, 3 H D] tensor on the GPU."""
    return (HAS_TRITON and y.is_cuda and y.dim() == 3 and y.dtype == BF16 and y.is_contiguous()
            and 16 <= D <= 8192 and D % 4 == 0)


def _rotate_ref(x, pos, inv_freq):
    ang = pos.float().unsqueeze(-1) * inv_freq
    ang = torch.cat((ang, ang), dim=-1)[:, None]          # [B, 1, L, D']
    h = x.shape[-1] // 2
    rot = torch.cat((-x[..., h:], x[..., :h]), dim=-1)
    return x * ang.cos() + rot * ang.sin()


def qkv_prep_ref(y, H, D, pos=None, base=100.0, eps=1e-5):
    """PyTorch reference of ``qkv_prep`` (fp32 math; q, k, v returned in bf16 like the kernel)."""
    B, L, _ = y.shape
    s = F.silu(y.float()).view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = s[0], s[1], s[2]
    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    k = k / (k.norm(dim=-1, keepdim=True) + eps)
    if pos is not None:
        S, T = _split(D)
        fr = rope_freqs(D, base, y.device)
        fs, ft = fr[:S // 4], fr[S // 4:]

        def rope(x):
            r, c, t = x[..., :S // 2], x[..., S // 2:S], x[..., S:]
            return torch.cat([_rotate_ref(r, pos[..., 0], fs), _rotate_ref(c, pos[..., 1], fs),
                              _rotate_ref(t, pos[..., 2], ft)], dim=-1)
        q, k = rope(q), rope(k)
    return q.to(BF16), k.to(BF16), v.to(BF16)


def qkv_prep(y, H, D, pos=None, freq=None, eps=1e-5):
    """(q, k, v) [B, H, L, D] (bf16) from the ``to_qkv`` output y [B, L, 3 H D] (bf16, contiguous, on the GPU).

    pos: [B, L, 3] (patch row, patch column, time in seconds) or None (no RoPE); freq: ``rope_freqs(D, base)``."""
    if pos is not None:
        pos = pos.float().contiguous()
    return _QKVPrep.apply(y, pos, freq, int(H), int(D), float(eps))
