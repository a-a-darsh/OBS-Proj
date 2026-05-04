"""
Core building blocks:
  - EqualLinear        : linear with equalized learning rate (StyleGAN2)
  - ModulatedConv2d    : weight-modulated + demodulated conv (StyleGAN2)
  - NoiseInjection     : per-pixel learned noise scale
  - StyledResBlock     : residual block with two modulated convolutions
  - DownBlock          : strided conv + InstanceNorm + LeakyReLU
  - StyledUpBlock      : bilinear upsample + modulated conv
  - FourierTimeEmbed   : sinusoidal embedding for continuous year values
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True,
                 activation: bool = False):
        super().__init__()
        self.scale = 1.0 / math.sqrt(in_dim)
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight * self.scale, self.bias)
        if self.activation:
            out = F.leaky_relu(out, 0.2) * math.sqrt(2)
        return out


class ModulatedConv2d(nn.Module):
    """
    StyleGAN2 weight-modulated convolution.
    Style vector s modulates per-input-channel weight scales;
    demodulation prevents feature-map magnitude explosion.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 style_dim: int, demodulate: bool = True, upsample: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ksize = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2

        fan_in = in_ch * kernel_size ** 2
        self.scale = 1.0 / math.sqrt(fan_in)
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))

        # Projects style to per-input-channel modulation scalar
        self.modulation = EqualLinear(style_dim, in_ch, bias=True)
        nn.init.ones_(self.modulation.bias)

        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Modulate
        s = self.modulation(style).view(B, 1, C, 1, 1)     # (B, 1, in_ch, 1, 1)
        w = self.scale * self.weight * s                    # (B, out_ch, in_ch, k, k)

        # Demodulate
        if self.demodulate:
            dmod = torch.rsqrt(w.pow(2).sum([2, 3, 4], keepdim=True) + 1e-8)
            w = w * dmod

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            H, W = H * 2, W * 2

        # Fold batch into groups for a single fused conv call
        x = x.reshape(1, B * C, H, W)
        w = w.reshape(B * self.out_ch, self.in_ch, self.ksize, self.ksize)
        x = F.conv2d(x, w, padding=self.padding, groups=B)
        return x.reshape(B, self.out_ch, H, W) + self.bias.view(1, -1, 1, 1)


class NoiseInjection(nn.Module):
    """Adds learned-scale Gaussian noise to introduce fine stochastic detail."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        return x + self.weight * noise


class StyledResBlock(nn.Module):
    """Residual block with two style-modulated convolutions (two style inputs)."""
    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.conv1 = ModulatedConv2d(channels, channels, 3, style_dim)
        self.conv2 = ModulatedConv2d(channels, channels, 3, style_dim)
        self.noise1 = NoiseInjection()
        self.noise2 = NoiseInjection()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor,
                style1: torch.Tensor, style2: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.noise1(self.conv1(x, style1)))
        x = self.noise2(self.conv2(x, style2))
        return self.act(x + residual)


class DownBlock(nn.Module):
    """Stride-2 conv + normalisation + LeakyReLU.
    norm: 'instance' | None
    """
    def __init__(self, in_ch: int, out_ch: int, norm: str = 'instance'):
        super().__init__()
        use_norm = norm is not None
        layers: list = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1,
                                  bias=not use_norm)]
        if norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StyledUpBlock(nn.Module):
    """Bilinear 2× upsample + modulated conv + noise + LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int, style_dim: int):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, 3, style_dim, upsample=True)
        self.noise = NoiseInjection()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.act(self.noise(self.conv(x, style)))


class SelfAttention(nn.Module):
    """
    SAGAN-style spatial self-attention.
    Applied at an intermediate feature resolution so the model can relate
    spatially distant strokes/components to each other.
    gamma starts at 0 so the block is a no-op at init and learns its influence.
    """
    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 8, 1)
        self.query = nn.Conv2d(channels, mid, 1, bias=False)
        self.key   = nn.Conv2d(channels, mid, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self._scale = mid ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)   # (B, HW, mid)
        k = self.key(x).view(B, -1, H * W)                        # (B, mid, HW)
        attn = torch.softmax(torch.bmm(q, k) * self._scale, dim=-1)  # (B, HW, HW)
        v = self.value(x).view(B, -1, H * W)                      # (B, C, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class FourierTimeEmbed(nn.Module):
    """
    Converts a normalized year value t ∈ [0, 1] to a dense embedding via
    sinusoidal Fourier features, then projects to out_dim.

    This gives the model continuous temporal awareness — it can distinguish
    a character from 900 BCE from one at 221 BCE, even within the same
    discrete stage.
    """
    def __init__(self, n_freqs: int, out_dim: int):
        super().__init__()
        freqs = torch.arange(1, n_freqs + 1).float()
        self.register_buffer("freqs", freqs)
        self.proj = EqualLinear(n_freqs * 2, out_dim, activation=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        angles = t.unsqueeze(-1) * (2 * math.pi * self.freqs)   # (B, n_freqs)
        feats = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, 2*n_freqs)
        return self.proj(feats)                                   # (B, out_dim)
