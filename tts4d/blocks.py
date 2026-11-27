"""Transformer layer of the model: within-view self-attention, the TTS layer across all views, and an MLP, each as a
pre-LayerNorm residual sub-block.  Module names match the released checkpoint (``module_list.{0,1,2}.{ln,f}``)."""
import torch.nn.functional as F
from torch import nn

from .tts_layer import TTSLayer


class SelfAttention(nn.Module):
    """Self-attention among the tokens of ONE view (the view axis is folded into the batch), with q / k RMS norm."""

    def __init__(self, dim=768, head_dim=64):
        super().__init__()
        self.dim, self.head_dim = dim, head_dim
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def forward(self, x):
        b, l, _ = x.shape
        q, k, v = self.to_qkv(x).view(b, l, 3, self.dim // self.head_dim, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.c_proj(o.transpose(1, 2).reshape(b, l, self.dim))


class MLP(nn.Module):
    def __init__(self, dim=768, inter_multi=4):
        super().__init__()
        self.c_fc = nn.Linear(dim, dim * inter_multi, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(dim * inter_multi, dim, bias=False)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Layer(nn.Module):
    def __init__(self, dim=768, attn_head_dim=64, tts_kwargs=None):
        super().__init__()
        self.module_list = nn.ModuleList([
            nn.ModuleDict({"ln": nn.LayerNorm(dim, bias=False), "f": SelfAttention(dim, attn_head_dim)}),
            nn.ModuleDict({"ln": nn.LayerNorm(dim, bias=False), "f": TTSLayer(dim, **(tts_kwargs or {}))}),
            nn.ModuleDict({"ln": nn.LayerNorm(dim, bias=False), "f": MLP(dim, 4)}),
        ])

    def forward(self, x, ctx, state=None):
        attn, tts, mlp = self.module_list
        b, vl, d = x.shape
        l = ctx["tokens_per_view"]
        h = attn["f"](attn["ln"](x).reshape(b * (vl // l), l, d)).reshape(b, vl, d)
        x = x + h
        h, u, state = tts["f"](tts["ln"](x), ctx, state)
        x = x + h
        x = x + mlp["f"](mlp["ln"](x))
        return x, u, state
