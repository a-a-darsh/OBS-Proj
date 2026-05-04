"""
MultiStageDiscriminator — 5 parallel PatchGAN branches, one per stage.

Each branch independently learns "does this image look like a valid
stage-k character?". When the batch contains a mix of target stages,
the gather operation routes each sample's prediction through the
correct branch without redundant masking.
"""
import torch
import torch.nn as nn
from .blocks import DownBlock, SelfAttention


class PatchDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nf = cfg.nf
        self.blocks = nn.ModuleList([
            DownBlock(cfg.in_channels, nf,     norm=None),
            DownBlock(nf,              nf * 2, norm='instance'),
            DownBlock(nf * 2,          nf * 4, norm='instance'),
        ])
        # Applied at 32×32 (after block 1, channels=nf*2)
        self.attn = SelfAttention(nf * 2)
        self.head = nn.Conv2d(nf * 4, 1, 4, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 1:
                x = self.attn(x)
        return self.head(x)


class MultiStageDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.branches = nn.ModuleList([
            PatchDiscriminator(cfg) for _ in range(cfg.num_stages)
        ])

    def forward(self, img: torch.Tensor,
                stage_idx: torch.Tensor) -> torch.Tensor:
        """
        Routes each sample to its stage's branch and returns the
        corresponding prediction.

        img:       (B, 1, H, W)
        stage_idx: (B,) int in [0, num_stages)
        returns:   (B, 1, h', w')
        """
        # Run all branches; gather the one that matches each sample's stage.
        all_preds = torch.stack([b(img) for b in self.branches], dim=1)
        # all_preds: (B, K, 1, h', w')
        idx = stage_idx.view(-1, 1, 1, 1, 1).expand(
            -1, 1, 1, *all_preds.shape[-2:])
        return all_preds.gather(1, idx).squeeze(1)  # (B, 1, h', w')

    def branch_forward(self, img: torch.Tensor, stage: int) -> torch.Tensor:
        """Evaluate a single branch — used for R1 penalty computation."""
        return self.branches[stage](img)
