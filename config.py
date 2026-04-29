from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# Approximate historical years for continuous time conditioning.
# Each stage prefix maps to a BCE/CE year.
STAGE_PREFIX_ORDER = ["O", "B", "S", "C", "M", "A", "L"]
STAGE_YEARS: Dict[int, Tuple[int, int]] = {
    0: (-1250, -1050),   # O — Oracle Bone Script (Shang 1250-1050 BCE)
    1: (-1046,-771),   # B — Bronze Inscription (Zhou ~1000 BCE)
    2:  (-500,20),   #  S - Seal
    3:  (-141,280),  # C — Clerical
    4:   (800,1100),   # M — Modern, regular Script (楷书 ~800 CE),
    #extras, add after here
    5: (-771, -220), #A - spring & (A)utumn period; For samples from state of Chu
    6: (-250,20) #L - Liushutong characters, basically later era seal script
}
YEAR_MIN = min(v[0] for v in STAGE_YEARS.values())   # -1250
YEAR_MAX = max(v[1] for v in STAGE_YEARS.values())   #  1100


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────
    data_dir: str = r"C:\Users\slitf\Documents\workspace\GM\OBS-Proj\data\datasets\Graph-Based Evolutionary Dataset for Oracle Bone Characters\organized"
    num_stages: int = 5
    stage_prefixes: List[str] = field(default_factory=lambda: STAGE_PREFIX_ORDER)
    stage_names: List[str] = field(default_factory=lambda: [
        "Oracle Bone",
        "Bronze",
        "Small Seal",
        "Clerical",
        "Modern",
    ])

    # ── Preprocessing ─────────────────────────────────────────────────
    image_size: int = 64
    in_channels: int = 1          # grayscale
    max_refs: int = 4             # max source-stage images pooled per sample

    # ── Architecture ──────────────────────────────────────────────────
    nf: int = 64                  # base feature-map count
    style_dim: int = 256          # per-layer style code dimension
    stage_emb_dim: int = 64       # discrete stage embedding dimension
    time_emb_dim: int = 64        # Fourier time embedding output dimension
    n_time_freqs: int = 16        # number of Fourier frequencies
    latent_dim: int = 256         # noise vector dimension
    n_mapper_layers: int = 8      # MLP depth in StageMapper
    n_res_blocks: int = 4         # styled residual blocks at bottleneck
    n_enc_blocks: int = 4         # encoder downsampling blocks (64→32→16→8→4)
    # n_style_layers = n_res_blocks * 2 + n_enc_blocks = 12
    n_style_layers: int = 12

    # ── Training ──────────────────────────────────────────────────────
    batch_size: int = 16
    n_epochs: int = 200
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.0
    beta2: float = 0.99
    r1_every: int = 16            # apply R1 penalty every N discriminator steps
    save_every: int = 20          # checkpoint every N epochs
    sample_every: int = 5         # save sample images every N epochs
    checkpoint_dir: str = "checkpoints"
    sample_dir: str = "samples"
    num_workers: int = 4

    # ── Loss weights ──────────────────────────────────────────────────
    lambda_adv: float = 1.0
    lambda_r1: float = 10.0
    lambda_cycle: float = 10.0
    lambda_stage: float = 5.0
    lambda_div: float = 1.0
    lambda_recon: float = 10.0    # reconstruction vs. known target

    # ── Diversity ─────────────────────────────────────────────────────
    n_div_samples: int = 2

    # ── Inference ─────────────────────────────────────────────────────
    unicode_font: str = r"C:\Windows\Fonts\msyh.ttc"   # MS YaHei fallback
    device: str = "cuda"
