"""
CharacterGenerator — adapts DLAT's DLATimg texture stream to Chinese character morphing.

Architecture:
  ContentEncoder   : 4× stride-2 convolutions, 64 → 4 spatial, channels 1→512
  StageMapper      : noise + stage/time conditioning → 12 per-layer style codes
  StyledDecoder    : 4 styled residual blocks (bottleneck) + 4 styled upsampling blocks

The content encoder captures structural identity ("which character is this"),
while the style codes inject temporal identity ("what era does it belong to").
Separating these two streams — analogous to DLAT separating content from age —
is what allows the generator to morph a character across stages without
losing its semantic identity.
"""
import torch
import torch.nn as nn
from .blocks import DownBlock, StyledResBlock, StyledUpBlock, SelfAttention
from .mapper import StageMapper
from config import STAGE_YEARS, YEAR_MIN, YEAR_MAX


def stage_to_year(stage_idx: torch.Tensor) -> torch.Tensor:
    """Convert integer stage indices to normalized year values in [0, 1]."""
    lows  = torch.tensor([STAGE_YEARS[i][0] for i in range(len(STAGE_YEARS))], dtype=torch.float32, device=stage_idx.device)
    highs = torch.tensor([STAGE_YEARS[i][1] for i in range(len(STAGE_YEARS))], dtype=torch.float32, device=stage_idx.device)
    year_table = lows + torch.rand(len(STAGE_YEARS), device=stage_idx.device) * (highs - lows)
    years = year_table[stage_idx]
    return (years - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)


class ContentEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nf = cfg.nf
        self.blocks = nn.ModuleList([
            DownBlock(cfg.in_channels, nf,     norm=None),
            DownBlock(nf,              nf * 2, norm='dropout'),
            DownBlock(nf * 2,          nf * 4, norm='dropout'),
            DownBlock(nf * 4,          nf * 8, norm='dropout'),
        ])
        # Applied at 16×16 (after block 2, channels=nf*4)
        self.attn = SelfAttention(nf * 4)

    def forward(self, x: torch.Tensor):
        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 2:
                x = self.attn(x)
            skips.append(x)
        # skips: [64×64 nf, 32×32 nf*2, 16×16 nf*4, 8×8 nf*8]
        return x, skips


class StyledDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nf = cfg.nf
        sd = cfg.style_dim

        # Bottleneck: n_res_blocks × StyledResBlock, each consuming 2 style vectors
        self.res_blocks = nn.ModuleList([
            StyledResBlock(nf * 8, sd) for _ in range(cfg.n_res_blocks)
        ])

        # Upsampling: 4 blocks, each consuming 1 style vector
        self.up_blocks = nn.ModuleList([
            StyledUpBlock(nf * 8, nf * 4, sd),
            StyledUpBlock(nf * 4, nf * 2, sd),
            StyledUpBlock(nf * 2, nf,     sd),
            StyledUpBlock(nf,     nf // 2, sd),
        ])
        # Applied at 16×16 (after up_blocks[0], channels=nf*4)
        self.attn = SelfAttention(nf * 4)

        # U-Net skip projections: 1×1 convs fuse encoder features into decoder
        # Order matches injection depth: [bottleneck, 16×16, 32×32]
        # 64×64 skip omitted — too close to pixel-level, causes input shadow in output.
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(nf * 8, nf * 8, 1),
            nn.Conv2d(nf * 4, nf * 4, 1),
            nn.Conv2d(nf * 2, nf * 2, 1),
        ])

        self.to_img = nn.Sequential(
            nn.Conv2d(nf // 2, cfg.in_channels, 1),
            nn.Tanh(),
        )

    def forward(self, content: torch.Tensor,
                styles: torch.Tensor,
                skips: list = None) -> torch.Tensor:
        # styles: (B, n_style_layers, style_dim)
        x = content
        if skips is not None:
            x = x + self.skip_projs[0](skips[3])   # inject 8×8 bottleneck skip
        idx = 0
        for res in self.res_blocks:
            x = res(x, styles[:, idx], styles[:, idx + 1])
            idx += 2
        for i, up in enumerate(self.up_blocks):
            x = up(x, styles[:, idx])
            idx += 1
            if i == 0:
                x = self.attn(x)
                if skips is not None:
                    x = x + self.skip_projs[1](skips[2])   # 16×16
            elif i == 1 and skips is not None:
                x = x + self.skip_projs[2](skips[1])       # 32×32
        return self.to_img(x)


class CharacterGenerator(nn.Module):
    """
    Full generator wrapping encoder + mapper + decoder.

    forward(src_img, src_stage, tgt_stage, noise=None) → tgt_img

    The mapper is bidirectional: both src_stage and tgt_stage are embedded,
    and src/tgt Fourier time embeddings are concatenated with the noise.
    This allows direct prediction between any two stages (not just adjacent),
    which is used during multi-directional training.
    """
    def __init__(self, cfg):
        super().__init__()
        self.encoder  = ContentEncoder(cfg)
        self.mapper   = StageMapper(cfg)
        self.decoder  = StyledDecoder(cfg)
        self.latent_dim = cfg.latent_dim

    def forward(self,
                src_imgs:  torch.Tensor,           # (B, R, 1, H, W) or (B, 1, H, W)
                src_stage: torch.Tensor,            # (B,) int
                tgt_stage: torch.Tensor,            # (B,) int
                noise:     torch.Tensor = None,     # (B, latent_dim) or None
                src_mask:  torch.Tensor = None,     # (B, R) bool or None
                ) -> torch.Tensor:                  # (B, 1, H, W)
        # Accept a plain (B, 1, H, W) image — single reference, no mask needed
        if src_imgs.ndim == 4:
            src_imgs = src_imgs.unsqueeze(1)

        B, R, C, H, W = src_imgs.shape

        if src_mask is None:
            src_mask = torch.ones(B, R, dtype=torch.bool, device=src_imgs.device)
        if noise is None:
            noise = torch.randn(B, self.latent_dim, device=src_imgs.device)

        # Encode every reference image, then max-pool over valid ones.
        # Max-pool preserves the strongest feature activation from any single
        # reference rather than averaging, which prevents stroke patterns from
        # washing out when refs show the character in different layouts.
        content_flat, skips_flat = self.encoder(src_imgs.reshape(B * R, C, H, W))
        _, fc, fH, fW = content_flat.shape
        invalid = ~src_mask                                          # (B, R) bool
        encoded = content_flat.reshape(B, R, fc, fH, fW)
        encoded = encoded.masked_fill(invalid.view(B, R, 1, 1, 1), float('-inf'))
        content = encoded.max(dim=1).values

        # Max-pool each skip map over refs the same way
        skips = []
        for s in skips_flat:
            s = s.reshape(B, R, *s.shape[1:])
            s = s.masked_fill(invalid.view(B, R, 1, 1, 1), float('-inf'))
            skips.append(s.max(dim=1).values)

        src_year = stage_to_year(src_stage)
        tgt_year = stage_to_year(tgt_stage)
        styles   = self.mapper(src_stage, tgt_stage, src_year, tgt_year, noise)
        return   self.decoder(content, styles, skips)
