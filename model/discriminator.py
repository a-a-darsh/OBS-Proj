"""
MultiStageDiscriminator — num_stages parallel PatchGAN branches.

Each branch judges: "given this source character image, is this target
image a genuine rendering of that character in stage k?"
Src and tgt are concatenated along the channel axis (pix2pix style)
before being passed to the branch, so the discriminator sees both.
"""
import torch
import torch.nn as nn
from .blocks import DownBlock, SelfAttention


class PatchDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nf = cfg.nf
        # src + tgt concatenated → 2 × in_channels input
        self.blocks = nn.ModuleList([
            DownBlock(cfg.in_channels * 2, nf, normalize=False),
            DownBlock(nf,                  nf * 2),
            DownBlock(nf * 2,              nf * 4),
        ])
        # src_stage injected as spatial bias after block 1 (at nf*2 channels)
        self.src_stage_emb = nn.Embedding(cfg.num_stages, nf * 2)
        self.attn = SelfAttention(nf * 2)
        self.head = nn.Conv2d(nf * 4, 1, 4, padding=1)

    def forward(self, x: torch.Tensor, src_stage: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 1:
                x = x + self.src_stage_emb(src_stage).view(x.shape[0], -1, 1, 1)
                x = self.attn(x)
        return self.head(x)


class MultiStageDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.branches = nn.ModuleList([
            PatchDiscriminator(cfg) for _ in range(cfg.num_stages)
        ])

    def forward(self,
                src_img:   torch.Tensor,   # (B, 1, H, W)
                tgt_img:   torch.Tensor,   # (B, 1, H, W)
                src_stage: torch.Tensor,   # (B,) int
                tgt_stage: torch.Tensor,   # (B,) int
                ) -> torch.Tensor:         # (B, 1, h', w')
        x = torch.cat([src_img, tgt_img], dim=1)   # (B, 2, H, W)
        all_preds = torch.stack([b(x, src_stage) for b in self.branches], dim=1)
        idx = tgt_stage.view(-1, 1, 1, 1, 1).expand(
            -1, 1, 1, *all_preds.shape[-2:])
        return all_preds.gather(1, idx).squeeze(1)
