"""
StagePredictor — classifies a character image into one of the K stages.

Used for the stage consistency loss: the generated image G(src, s→t)
should be classifiable as stage t.

Trained jointly on real images (in the D step) and used to penalise
the generator (in the G step) — analogous to DLAT's age predictor Pimg.
"""
import torch.nn as nn
from .blocks import DownBlock


class StagePredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nf = cfg.nf
        self.net = nn.Sequential(
            DownBlock(cfg.in_channels, nf,     norm=None),
            DownBlock(nf,              nf * 2, norm='instance'),
            DownBlock(nf * 2,          nf * 4, norm='instance'),
            DownBlock(nf * 4,          nf * 8, norm='instance'),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 8, nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nf * 4, cfg.num_stages),
        )

    def forward(self, x):
        return self.net(x)   # (B, num_stages) — raw logits
