"""2-D spatial + 1-D continuous-time rotary position embedding (no parameters), applied to the q / k of the TTS layer."""
import torch
from torch import nn


def _rot(x):
    d = x.shape[-1]
    return torch.cat((-x[..., d // 2:], x[..., :d // 2]), dim=-1)


def _rotate(x, pos, base):
    """x [B, H, N, D], pos [B, N]: rotate pairs of channels by pos * base^(-2i / D)."""
    D = x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / float(D)))
    ang = pos.to(torch.float32).unsqueeze(-1) * inv_freq
    ang = torch.cat((ang, ang), dim=-1)
    return x * ang.cos().to(x.dtype)[:, None] + _rot(x) * ang.sin().to(x.dtype)[:, None]


class RoPE2DTime(nn.Module):
    """Channels split into a spatial band (row half / column half) and a time band (about a quarter of the head).

    positions: [B, N, 3] = (patch row, patch column, time in seconds); rows and columns are given in units of a
    16 x 16 reference grid (see ``model.TTS4D.positions``), so the angles do not depend on the image size."""

    def __init__(self, frequency=100.0):
        super().__init__()
        self.base = float(frequency)

    @staticmethod
    def split(d):
        t = (d // 4) & ~1
        s = d - t
        s -= s % 4
        return s, d - s

    def forward(self, x, positions):
        s, _ = self.split(x.shape[-1])
        sp, tm = x[..., :s], x[..., s:]
        row, col = sp.chunk(2, dim=-1)
        sp = torch.cat([_rotate(row, positions[..., 0], self.base), _rotate(col, positions[..., 1], self.base)], dim=-1)
        tm = _rotate(tm, positions[..., 2], 100.0)
        return torch.cat((sp, tm), dim=-1)
