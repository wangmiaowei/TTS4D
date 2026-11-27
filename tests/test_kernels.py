"""Per-operator check of the fast kernels (tts4d.kernels) against the PyTorch reference, forward and backward.

For every operator: the forward against an fp32 PyTorch implementation of the same math, and the gradients of the
hand-written backward against autograd through that implementation (relative error ||a - b|| / ||b||; the loss is
a fixed random projection of all outputs).  The bf16-operand operators are expected to agree to bf16 precision
(~1e-3 .. 1e-2), the fp32 solve to fp32 precision.  At the end the whole TTS layer (fast vs reference, forward,
u(q) and all parameter gradients).

    python tests/test_kernels.py                       # d = 768, dF = 1536, 8K and 32K tokens
    python tests/test_kernels.py --tokens 8192 --json out.json
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tts4d import kernels as K                                                       # noqa: E402
from tts4d.kernels import memory, qk_norm, qkv, solve                                 # noqa: E402
from tts4d.rope import RoPE2DTime                                                     # noqa: E402
from tts4d.tts_layer import TTSLayer, feature_map as feature_map_ref                  # noqa: E402


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def check(name, fast_fn, ref_fn, inputs, results, n_tokens):
    """fast_fn / ref_fn: inputs -> tuple of outputs.  Compares outputs and the input gradients of a random
    projection of the outputs."""
    xs_f = [x.detach().clone().requires_grad_(x.is_floating_point()) for x in inputs]
    xs_r = [x.detach().clone().float().requires_grad_(x.is_floating_point()) for x in inputs]
    out_f = fast_fn(*xs_f)
    out_r = ref_fn(*xs_r)
    g = torch.Generator(device=out_r[0].device).manual_seed(1)
    proj = [torch.randn(o.shape, device=o.device, generator=g) for o in out_r]
    loss_f = sum((o.float() * p).sum() for o, p in zip(out_f, proj))
    loss_r = sum((o.float() * p).sum() for o, p in zip(out_r, proj))
    loss_f.backward()
    loss_r.backward()
    row = dict(op=name, tokens=n_tokens,
               fwd=[round(rel(a.float(), b), 7) for a, b in zip(out_f, out_r)],
               bwd=[round(rel(a.grad.float(), b.grad), 7) if b.grad is not None else None
                    for a, b in zip(xs_f, xs_r) if b.requires_grad])
    results.append(row)
    print(f"{name:14s} N={n_tokens:6d}  fwd {['%.2e' % e for e in row['fwd']]}  "
          f"bwd {['%.2e' % e if e is not None else '-' for e in row['bwd']]}", flush=True)


def unit_keys(B, H, N, d, dev, gen):
    """Inputs like the layer's: unit-norm keys with a shared mean direction (the features then share one too)."""
    k = torch.randn(B, H, N, d, device=dev, generator=gen) + 0.7 * torch.randn(1, 1, 1, d, device=dev, generator=gen)
    return k / k.norm(dim=-1, keepdim=True)


def moments_ref(phi, v, a):
    aw = phi * a
    return (aw.transpose(-1, -2) @ phi, aw.transpose(-1, -2) @ v, aw.sum(-2, keepdim=True),
            (v * a).sum(-2, keepdim=True), a.sum(-2, keepdim=True))


def centred_solve_ref(G, C, S, pb, vb, inv, lam, W0):
    """The reference layer's solve on the same inputs (autograd through Cholesky / triangular solve)."""
    Gs = (G - S.transpose(-1, -2) @ pb) * inv.transpose(-1, -2) * inv
    Cs = (C - S.transpose(-1, -2) @ vb) * inv.transpose(-1, -2)
    n = G.shape[-1]
    eye = torch.eye(n, device=G.device, dtype=G.dtype)
    L = torch.linalg.cholesky(0.5 * (Gs + Gs.transpose(-1, -2)) + lam * eye)
    M = torch.linalg.solve_triangular(L, eye.expand(L.shape), upper=False)
    return M, M.transpose(-1, -2) @ (M @ (Cs + lam * W0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--dF", type=int, default=1536)
    ap.add_argument("--layer_tokens", type=int, default=8192, help="input tokens of the whole-layer test")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    print("Triton kernels available:", K.available("cuda"))
    torch.backends.cuda.matmul.allow_tf32 = False
    dev, B, H, d, dF = "cuda", 1, 1, args.d, args.dF
    gen = torch.Generator(device=dev).manual_seed(0)
    res = []
    for N in args.tokens:
        # ---- feature map
        k = unit_keys(B, H, N, d, dev, gen).to(torch.bfloat16)
        w0 = torch.randn(H, d, dF, device=dev, generator=gen) * (2.0 / d) ** 0.5
        w2 = torch.randn(H, d, dF, device=dev, generator=gen) * (2.0 / d) ** 0.5
        check("feature_map", lambda x, a, b: (memory.feature_map(x, a, b),),
              lambda x, a, b: (feature_map_ref(x, a, b),), [k, w0, w2], res, N)
        # ---- moments
        with torch.no_grad():
            phi = feature_map_ref(k.float(), w0, w2)
        v = F.silu(torch.randn(B, H, N, d, device=dev, generator=gen)).to(torch.bfloat16)
        a = F.softplus(torch.randn(B, H, N, 1, device=dev, generator=gen) + 0.5)
        check("moments", memory.moments, moments_ref, [phi, v, a], res, N)
        # ---- read
        W = torch.randn(B, H, dF, d, device=dev, generator=gen) * 0.05
        pc = (phi - phi.mean(-2, keepdim=True)) * 30.0
        check("read_mm", lambda x, w: (memory.read_mm(x, w),), lambda x, w: (x @ w,), [pc, W], res, N)
        # ---- q / k / v preparation (SiLU, L2 norm, 2-D + time RoPE)
        y = torch.randn(B, N, 3 * H * d, device=dev, generator=gen).to(torch.bfloat16)
        pos = torch.cat([torch.rand(B, N, 2, device=dev, generator=gen) * 16,
                         torch.rand(B, N, 1, device=dev, generator=gen) * 10], -1)
        freq = qkv.rope_freqs(d, 100.0, dev)
        rope = RoPE2DTime(100.0)

        def qkv_ref(y_):
            s = F.silu(y_).view(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
            q_, k_ = (t / (t.norm(dim=-1, keepdim=True) + 1e-5) for t in (s[0], s[1]))
            return rope(q_, pos), rope(k_, pos), s[2]
        check("qkv_prep", lambda y_: qkv.qkv_prep(y_, H, d, pos, freq), qkv_ref, [y], res, N)
        # ---- attention q / k RMSNorm (packed qkv, 12 heads of 64)
        qkv_t = torch.randn(B, N, 3, 12, 64, device=dev, generator=gen).to(torch.bfloat16)
        wq = 1 + 0.1 * torch.randn(64, device=dev, generator=gen)
        wk = 1 + 0.1 * torch.randn(64, device=dev, generator=gen)
        check("qk_norm", lambda x, a_, b_: (qk_norm.qk_norm_packed(x, a_, b_),),
              lambda x, a_, b_: (qk_norm.qk_norm_packed_ref(x, a_, b_),), [qkv_t, wq, wk], res, N)
        # ---- centred solve (fp32), statistics of the features above
        with torch.no_grad():
            G, C, S, T, A = moments_ref(phi, v.float(), a)
            pb, vb = S / A, T / A
            dg = (G.diagonal(dim1=-2, dim2=-1) - (S * pb).squeeze(-2)).clamp_min(0)
            inv = (1.0 / dg.clamp_min(1e-3 * dg.amax(-1, keepdim=True)).sqrt()).unsqueeze(-2)
        lam = torch.full((1, H, 1, 1), 1.0, device=dev)
        W0 = torch.randn(1, H, dF, d, device=dev, generator=gen) * (2.0 / dF) ** 0.5
        check("centred_solve", lambda *z: solve.centred_solve(*z)[1:], lambda *z: centred_solve_ref(*z)[1:],
              [G, C, S, pb, vb, inv, lam, W0], res, N)
        del k, phi, v, a, G, C, y, qkv_t
        torch.cuda.empty_cache()

    # ---- the whole TTS layer: fast vs reference, forward, u(q) and the parameter gradients
    torch.manual_seed(0)
    layer = TTSLayer(dim=d, head_dim=d, inter_multi=dF / d).to(dev)
    layer.law_gamma = 0.2
    Nin, Nq = args.layer_tokens, args.layer_tokens // 2
    x = torch.randn(1, Nin, d, device=dev, generator=gen)
    xq = torch.randn(1, Nq, d, device=dev, generator=gen)
    pos = lambda n: torch.cat([torch.rand(1, n, 2, device=dev, generator=gen) * 16,          # noqa: E731
                               torch.rand(1, n, 1, device=dev, generator=gen) * 10], -1)
    ctx, ctxq = dict(pos=pos(Nin), n_views=32), dict(pos=pos(Nq), n_views=32)
    outs = {}
    for fast in (False, True):
        layer.fast = fast
        layer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", torch.bfloat16, enabled=fast):
            y, _, st = layer(x, ctx)
            yq, u, _ = layer(xq, ctxq, st)
            loss = y.float().square().mean() + yq.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        outs[fast] = (y.float().detach(), yq.float().detach(), u.detach(),
                      {n: p.grad.detach().clone() for n, p in layer.named_parameters()}, time.time() - t0)
    row = dict(op="TTSLayer", tokens=Nin, fwd_inputs=rel(outs[True][0], outs[False][0]),
               fwd_queries=rel(outs[True][1], outs[False][1]), u=rel(outs[True][2], outs[False][2]),
               grads={n: round(rel(outs[True][3][n], g), 5) for n, g in outs[False][3].items()})
    res.append(row)
    print(f"TTSLayer (fast, bf16 autocast) vs reference (fp32), {Nin} input / {Nq} query tokens:")
    print(f"  forward: inputs {row['fwd_inputs']:.2e}  queries {row['fwd_queries']:.2e}  u(q) {row['u']:.2e}")
    print("  grads: " + "  ".join(f"{n} {e:.2e}" for n, e in row["grads"].items()))
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
