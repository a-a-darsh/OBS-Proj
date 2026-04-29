"""
Loss functions for CharacterEvolutionGAN.

Non-saturating GAN (softplus formulation) + lazy R1 regularisation.
Cycle, stage-consistency, diversity, and reconstruction losses follow
DLAT's design, adapted for bidirectional character morphing.

Complexity-weighted reconstruction:
  Simpler characters (lower edge density → fewer strokes) receive a
  higher loss weight.  This prevents the model from collapsing to
  always predicting the trivially simple/unchanged form — e.g., 中
  barely changes across stages, so its contribution is up-weighted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── GAN losses ────────────────────────────────────────────────────────────────

def adv_loss_g(fake_pred: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: maximise log D(fake)."""
    return F.softplus(-fake_pred).mean()


def adv_loss_d(real_pred: torch.Tensor,
               fake_pred: torch.Tensor) -> torch.Tensor:
    """Discriminator loss: maximise log D(real) + log(1-D(fake))."""
    return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()


def r1_penalty(real_pred: torch.Tensor,
               real_img: torch.Tensor) -> torch.Tensor:
    """
    R1 gradient penalty on the discriminator's real-data gradient.
    Stabilises training without spectral normalisation.
    Applied lazily (every r1_every steps) to amortise cost.
    """
    grad, = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True,
        retain_graph=True,
    )
    return grad.pow(2).flatten(1).sum(1).mean()


# ── Generator auxiliary losses ────────────────────────────────────────────────

def cycle_loss(reconstructed: torch.Tensor,
               original: torch.Tensor) -> torch.Tensor:
    """L1 cycle-consistency: G(G(src, s→t), t→s) ≈ src."""
    return F.l1_loss(reconstructed, original)


def reconstruction_loss(fake: torch.Tensor,
                        real: torch.Tensor) -> torch.Tensor:
    """Direct L1 reconstruction when ground-truth target is available."""
    return F.l1_loss(fake, real)


def diversity_loss(fake1: torch.Tensor,
                   fake2: torch.Tensor) -> torch.Tensor:
    """
    Penalise mode collapse: two different noise vectors should produce
    visually distinct outputs (analogous to DLAT's L_div^img).
    Returns a negative value; minimising it maximises diversity.
    """
    return -F.l1_loss(fake1.detach(), fake2)


def stage_consistency_loss(predictor: nn.Module,
                           fake_img: torch.Tensor,
                           tgt_stage: torch.Tensor) -> torch.Tensor:
    """Generated image should be classified as the target stage."""
    return F.cross_entropy(predictor(fake_img), tgt_stage)


# ── Perceptual loss ───────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG16 feature matching at relu2_2 and relu3_3.
    Grayscale inputs are expanded to 3-channel before passing through VGG.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg)[:9]).eval()
        self.slice2 = nn.Sequential(*list(vgg)[9:16]).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, 3, 1, 1)                 # grayscale → RGB
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        return x

    def forward(self, fake: torch.Tensor,
                real: torch.Tensor) -> torch.Tensor:
        f, r = self._prep(fake), self._prep(real.detach())
        loss  = F.l1_loss(self.slice1(f), self.slice1(r))
        loss += F.l1_loss(self.slice2(f), self.slice2(r))
        return loss


# ── Complexity-aware weighting ────────────────────────────────────────────────

def edge_density(img: torch.Tensor) -> torch.Tensor:
    """
    Per-image Sobel edge density as a stroke-complexity proxy.
    img: (B, 1, H, W) in [-1, 1]
    Returns: (B,) density in (0, ∞)
    """
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                             dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    img01 = img * 0.5 + 0.5                              # [0, 1]
    gx = F.conv2d(img01, sobel_x, padding=1)
    gy = F.conv2d(img01, sobel_y, padding=1)
    return (gx.pow(2) + gy.pow(2)).sqrt().mean(dim=[1, 2, 3])


def complexity_weighted_recon(fake: torch.Tensor,
                               real: torch.Tensor) -> torch.Tensor:
    """
    L1 reconstruction loss with per-sample weights inversely proportional
    to character complexity.  Simple characters (low edge density) get
    higher weight — they are expected to be predicted more precisely and
    must not degenerate into trivially easy outputs.
    """
    density = edge_density(real.detach())
    # Invert and normalise within batch: simpler → higher weight
    w = 1.0 - density / (density.max() + 1e-8)           # (B,)
    per_sample = F.l1_loss(fake, real, reduction="none").mean(dim=[1, 2, 3])
    return (w * per_sample).mean()
