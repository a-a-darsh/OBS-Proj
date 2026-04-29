"""
StageMapper: the core of multi-directional conditioning.

Given (source_stage, target_stage, source_year, target_year, noise),
produces K per-layer style codes for the decoder.

Key difference from DLAT: we condition on BOTH source AND target stage
embeddings simultaneously, so the mapper learns the full 5×5 transition
space rather than a one-way aging direction.

Time conditioning adds continuous temporal resolution — a character from
900 BCE looks subtly different from one at 221 BCE even if both are
labeled "Warring States".
"""
import torch
import torch.nn as nn
from .blocks import EqualLinear, FourierTimeEmbed


class StageMapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stage_emb = nn.Embedding(cfg.num_stages, cfg.stage_emb_dim)
        self.time_emb = FourierTimeEmbed(cfg.n_time_freqs, cfg.time_emb_dim)

        # Input: noise || src_stage_emb || tgt_stage_emb || src_time_emb || tgt_time_emb
        in_dim = (cfg.latent_dim
                  + 2 * cfg.stage_emb_dim
                  + 2 * cfg.time_emb_dim)

        layers = [EqualLinear(in_dim, cfg.style_dim, activation=True)]
        for _ in range(cfg.n_mapper_layers - 2):
            layers.append(EqualLinear(cfg.style_dim, cfg.style_dim, activation=True))
        layers.append(EqualLinear(cfg.style_dim, cfg.style_dim * cfg.n_style_layers))

        self.net = nn.Sequential(*layers)
        self.n_style_layers = cfg.n_style_layers
        self.style_dim = cfg.style_dim

    def forward(self,
                src_stage: torch.Tensor,   # (B,) int
                tgt_stage: torch.Tensor,   # (B,) int
                src_year: torch.Tensor,    # (B,) float in [0,1]
                tgt_year: torch.Tensor,    # (B,) float in [0,1]
                noise: torch.Tensor        # (B, latent_dim)
                ) -> torch.Tensor:         # (B, n_style_layers, style_dim)
        src_se = self.stage_emb(src_stage)
        tgt_se = self.stage_emb(tgt_stage)
        src_te = self.time_emb(src_year)
        tgt_te = self.time_emb(tgt_year)
        x = torch.cat([noise, src_se, tgt_se, src_te, tgt_te], dim=-1)
        out = self.net(x)
        return out.view(out.shape[0], self.n_style_layers, self.style_dim)
