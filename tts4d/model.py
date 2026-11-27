"""TTS4D: uncertainty-aware 4D novel view synthesis with test-time solver layers (inference only).

Every view (input or query) is cut into p x p patches.  A patch token carries the Plucker rays of its pixels
(ray origin, direction, moment o x d), the time of its view and, for input views only, its RGB values; query views
have zeros in the RGB slot.  All tokens pass through L identical layers (within-view attention, TTS layer, MLP) and
the query tokens are decoded into pixels (RGB) and a per-pixel log-variance (uncertainty).

Uncertainty (all in nats of log variance):
    U_epi(q) = beta_0 + sum_l beta_l log(u_l(q) / u_bar_l)       epistemic: u_l from the TTS layers, beta_l >= 0
    U(q)     = U_epi(q) + r(q)                                      total: r = learned aleatoric residual head
The predicted standard deviation is exp(U / 2).  For display, each map is also returned as a "confidence"
sigmoid(U_ref - U) in (0, 1) against its running reference level U_ref recorded during training.
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .blocks import Layer
from .geometry import plucker_rays

DEFAULT_CONFIG = dict(patch_size=8, dim=768, num_layers=24, attn_head_dim=64, max_time_seconds=30.0, rope_ref_grid=16,
                      tts=dict(head_dim=768, inter_multi=2, ridge_init=10.0, ridge_min=1e-3, base_evidence=1.0,
                               scale_eps=1e-3, rope_frequency=100.0),
                      law_gamma=None, law_nref=32.0, aleatoric_hidden=768,
                      fast=None)   # None: fast kernels under bf16 autocast on a GPU with Triton; True / False: force


class TTS4D(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG, **(config or {}))
        self.cfg = cfg
        self.p, self.dim = cfg["patch_size"], cfg["dim"]
        p, dim = self.p, self.dim
        in_ch = 9 + 3 + 1                                       # Plucker (9) + RGB (3) + time (1) per pixel
        self.input_linear = nn.Linear(in_ch * p * p, dim, bias=False)
        self.input_layernorm = nn.LayerNorm(dim, bias=False)
        self.blocks = nn.ModuleList([Layer(dim, cfg["attn_head_dim"], cfg["tts"], fast=cfg["fast"])
                                     for _ in range(cfg["num_layers"])])
        if cfg.get("law_gamma"):
            for blk, g in zip(self.blocks, cfg["law_gamma"]):
                blk.module_list[1]["f"].law_gamma, blk.module_list[1]["f"].law_nref = float(g), float(cfg["law_nref"])
        self.image_token_decoder = nn.Sequential(nn.LayerNorm(dim, bias=False), nn.Linear(dim, p * p * 3, bias=False), nn.Sigmoid())
        # ---- uncertainty
        hh = cfg["aleatoric_hidden"]
        self.logvar_token_decoder = nn.Sequential(nn.LayerNorm(dim, bias=False), nn.Linear(dim, hh), nn.GELU(),
                                                  nn.Linear(hh, p * p, bias=False))  # aleatoric residual r (per pixel)
        self.logvar_offset = nn.Parameter(torch.zeros(()))
        L = cfg["num_layers"]
        self.raw_epi_gain = nn.Parameter(torch.zeros(L))       # beta_l = softplus(raw_epi_gain_l)
        self.register_buffer("epi_beta", torch.zeros(()))      # beta_0 (closed-form ML level, recorded in training)
        self.register_buffer("epi_u_scale", torch.ones(L))     # u_bar_l (running mean of u_l, recorded in training)
        self.register_buffer("log_sigma0_sq", torch.zeros(()))      # display reference of U
        self.register_buffer("log_sigma0_sq_epi", torch.zeros(()))  # display reference of U_epi

    def set_fast(self, fast):
        """Select the implementation of the TTS layers and of the attention q / k norm: True = the fast kernels
        (``tts4d.kernels``), False = the PyTorch reference, None = automatic (fast under bf16 autocast on a GPU with
        Triton).  Both compute the same function; see ``tests/verify_fast_path.py``."""
        self.cfg["fast"] = fast
        for blk in self.blocks:
            blk.module_list[0]["f"].fast = fast
            blk.module_list[1]["f"].fast = fast
        return self

    # ------------------------------------------------------------------ tokens
    def tokens(self, views, with_image):
        """views: image [B, V, 3, H, W] in [0, 1] (inputs), c2w [B, V, 4, 4], K [B, V, 4] = (fx, fy, cx, cy) in pixels,
        time [B, V] in seconds.  -> [B, V * h * w, in_ch * p * p]"""
        B, V = views["c2w"].shape[:2]
        H, W = self.hw
        rays = plucker_rays(views["K"], views["c2w"], H, W)                       # [B, V, 9, H, W]
        rgb = views["image"] * 2.0 - 1.0 if with_image else rays.new_zeros(B, V, 3, H, W)
        tau = (views["time"].float() / self.cfg["max_time_seconds"] * 2.0 - 1.0).view(B, V, 1, 1, 1).expand(B, V, 1, H, W)
        x = torch.cat([rays, rgb, tau], dim=2)
        return rearrange(x, "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)", ph=self.p, pw=self.p)

    def positions(self, times, B, V):
        """RoPE positions [B, V * h * w, 3]: patch row / column on a 16 x 16 reference grid, time in seconds."""
        hh, ww = self.hw[0] // self.p, self.hw[1] // self.p
        dev = times.device
        yx = torch.cartesian_prod(torch.arange(hh, device=dev), torch.arange(ww, device=dev)).float()
        yx = yx * (self.cfg["rope_ref_grid"] / math.sqrt(hh * ww))
        yx = yx.view(1, 1, hh * ww, 2).expand(B, V, -1, -1)
        t = times.float().view(B, V, 1, 1).expand(B, V, hh * ww, 1)
        return torch.cat([yx, t], -1).reshape(B, V * hh * ww, 3)

    # ------------------------------------------------------------------ forward
    def _embed(self, views, with_image):
        dev = views["c2w"].device
        with torch.autocast(device_type=dev.type, enabled=False):
            x = self.tokens(views, with_image).float()
            old = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False          # camera baselines of 1e-3 must survive the first projection
            x = self.input_linear(x)
            torch.backends.cuda.matmul.allow_tf32 = old
        return self.input_layernorm(x)

    @torch.no_grad()
    def encode(self, inputs):
        """Test-time solve: run the input views through the network; every TTS layer writes their tokens into its
        memory in closed form.  Returns the list of the 24 solved states (fixed size, independent of the number of
        input views and of their order).

        inputs: dict(image [B, V, 3, H, W] in [0, 1], c2w [B, V, 4, 4], K [B, V, 4], time [B, V])
        Cameras are OpenCV camera-to-world matrices in the normalised scene frame (geometry.normalize_scene), times in
        seconds relative to the first input; H and W must be multiples of 8.
        """
        B, V, _, H, W = inputs["image"].shape
        self.hw = (H, W)
        n_tok = (H // self.p) * (W // self.p)
        x = self._embed(inputs, True)
        ctx = dict(n_views=V, tokens_per_view=n_tok, pos=self.positions(inputs["time"], B, V))
        states = []
        for blk in self.blocks:
            x, _, st = blk(x, ctx)
            states.append(st)
        states_meta = dict(hw=(H, W), n_views=V)
        return dict(states=states, **states_meta)

    @torch.no_grad()
    def render(self, memory, queries):
        """Read the solved memory for query views: dict(c2w [B, Vq, 4, 4], K [B, Vq, 4], time [B, Vq]).
        Queries never write, so any number of them can be rendered from one solve, in any batches."""
        H, W = memory["hw"]
        self.hw = (H, W)
        p = self.p
        hh, ww = H // p, W // p
        n_tok = hh * ww
        B, Vq = queries["c2w"].shape[:2]
        x = self._embed(queries, False)
        ctx = dict(n_views=memory["n_views"], tokens_per_view=n_tok, pos=self.positions(queries["time"], B, Vq))
        us = []
        for blk, st in zip(self.blocks, memory["states"]):
            x, u, _ = blk(x, ctx, st)
            us.append(u)
        return self.decode(x, us, Vq)

    def forward(self, inputs, queries):
        """One forward pass = encode the inputs + render the queries."""
        return self.render(self.encode(inputs), queries)

    def decode(self, xq, us, Vq):
        """RGB and uncertainty of the query tokens."""
        H, W = self.hw
        p = self.p
        hh, ww = H // p, W // p
        B = xq.shape[0]
        unpatch = dict(v=Vq, hh=hh, ww=ww, ph=p, pw=p)
        rgb = rearrange(self.image_token_decoder(xq), "b (v hh ww) (ph pw c) -> b v c (hh ph) (ww pw)", c=3, **unpatch)
        out = dict(rgb=rgb.float())
        # ---- uncertainty
        r = rearrange(self.logvar_token_decoder(xq), "b (v hh ww) (ph pw c) -> b v c (hh ph) (ww pw)", c=1, **unpatch).float()
        r = r + self.logvar_offset
        u = torch.stack(us, 1).float().clamp_min(0.0) / self.epi_u_scale.view(1, -1, 1).clamp_min(1e-12)   # [B, L, Nq]
        beta = F.softplus(self.raw_epi_gain.float())
        a = (torch.log(u.clamp_min(1e-12)) * beta.view(1, -1, 1)).sum(1)                                  # [B, Nq]
        a = rearrange(a, "b (v hh ww) -> (b v) 1 hh ww", v=Vq, hh=hh)
        a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False).view(B, Vq, 1, H, W)       # patch -> pixel
        U_epi = self.epi_beta + a
        U = U_epi + r
        out.update(log_var=U, log_var_epi=U_epi, log_var_alea=r, std=torch.exp(0.5 * U),
                   conf=torch.sigmoid(self.log_sigma0_sq - U),
                   conf_epi=torch.sigmoid(self.log_sigma0_sq_epi - U_epi),
                   conf_alea=torch.sigmoid(self.log_sigma0_sq - self.log_sigma0_sq_epi - r))
        return out
