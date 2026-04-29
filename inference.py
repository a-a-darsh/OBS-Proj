"""
Inference module — two public entry points:

  predict_ancient(char, G, cfg, device)
    Given a modern Unicode Chinese character (e.g. '水'), renders it as
    a Regular Script (stage 4) image and steps backwards through all 5
    stages, returning a list of images [stage4, stage3, stage2, stage1, stage0].

  predict_modern(obs_image, G, cfg, device)
    Given an Oracle Bone Script image tensor (stage 0), steps forward
    through all 5 stages, returning [stage0, stage1, stage2, stage3, stage4].

Both methods perform stepwise inference: each generated image is fed as
input to the next step.  Direct (skip-step) generation is also available
via predict_direct().

Rendering pipeline for predict_ancient:
  1. Render unicode char using a CJK font (PIL) → 64×64 grayscale image
  2. Normalise to [-1, 1] tensor (same preprocessing as training data)
  3. Run backwards through the generator
"""
import os
from typing import List, Optional
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from config import Config
from model import CharacterGenerator


# ── Font rendering ────────────────────────────────────────────────────────────

_FALLBACK_FONTS = [
    r"C:\Windows\Fonts\msyh.ttc",        # Microsoft YaHei (Windows)
    r"C:\Windows\Fonts\simsun.ttc",      # SimSun
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Linux
    "/System/Library/Fonts/PingFang.ttc",                        # macOS
]


def _get_font(path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    candidates = ([path] if path else []) + _FALLBACK_FONTS
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_unicode(char: str, image_size: int = 64,
                   font_path: Optional[str] = None) -> torch.Tensor:
    """
    Render a single Unicode Chinese character to a normalised tensor.
    Returns: (1, 1, H, W) in [-1, 1] on CPU.
    """
    canvas = Image.new("L", (image_size, image_size), color=255)
    draw   = ImageDraw.Draw(canvas)
    font   = _get_font(font_path, image_size - 8)

    bbox = draw.textbbox((0, 0), char, font=font)
    x = (image_size - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (image_size - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return tf(canvas).unsqueeze(0)   # (1, 1, H, W)


# ── Core stepwise inference ───────────────────────────────────────────────────

@torch.no_grad()
def _step(G: CharacterGenerator,
          img: torch.Tensor,
          src_stage: int, tgt_stage: int,
          device: torch.device) -> torch.Tensor:
    """Single stage-to-stage generation step."""
    img = img.to(device)
    s = torch.tensor([src_stage], dtype=torch.long, device=device)
    t = torch.tensor([tgt_stage], dtype=torch.long, device=device)
    return G(img, s, t)   # (1, 1, H, W)


# ── Public API ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_ancient(
    char: str,
    G: CharacterGenerator,
    cfg: Config,
    device: torch.device,
    font_path: Optional[str] = None,
    n_steps: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Modern → Ancient inference.

    Renders `char` as a stage-4 (Regular Script) image, then steps
    backwards: 4→3→2→1→0.

    Args:
        char      : a single Unicode Chinese character (e.g. '水')
        G         : trained CharacterGenerator
        cfg       : Config object
        device    : torch device
        font_path : override CJK font path (uses system font if None)
        n_steps   : how many backward steps (default = all = 4)

    Returns:
        List of (1, 1, H, W) tensors: [stage4, stage3, ..., stage(4-n_steps)]
    """
    G.eval()
    n_steps = n_steps or (cfg.num_stages - 1)
    n_steps = min(n_steps, cfg.num_stages - 1)

    current = render_unicode(char, cfg.image_size, font_path or cfg.unicode_font)
    sequence = [current]

    src = cfg.num_stages - 1       # start from latest stage (Regular)
    for step in range(n_steps):
        tgt = src - 1
        current = _step(G, current, src, tgt, device)
        sequence.append(current.cpu())
        src = tgt

    return sequence   # index 0 = modern, index -1 = most ancient generated


@torch.no_grad()
def predict_modern(
    obs_image: torch.Tensor,
    G: CharacterGenerator,
    cfg: Config,
    device: torch.device,
    n_steps: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Ancient → Modern inference.

    Takes an Oracle Bone Script image and steps forward: 0→1→2→3→4.

    Args:
        obs_image : (1, 1, H, W) or (1, H, W) tensor of an OBS character
                    in [-1, 1].
        G         : trained CharacterGenerator
        cfg       : Config object
        device    : torch device
        n_steps   : how many forward steps (default = all = 4)

    Returns:
        List of (1, 1, H, W) tensors: [stage0, stage1, ..., stage(n_steps)]
    """
    G.eval()
    if obs_image.dim() == 3:
        obs_image = obs_image.unsqueeze(0)

    n_steps = n_steps or (cfg.num_stages - 1)
    n_steps = min(n_steps, cfg.num_stages - 1)

    current = obs_image
    sequence = [current.cpu()]

    src = 0
    for step in range(n_steps):
        tgt = src + 1
        current = _step(G, current, src, tgt, device)
        sequence.append(current.cpu())
        src = tgt

    return sequence   # index 0 = oracle bone, index -1 = most modern generated


@torch.no_grad()
def predict_direct(
    src_image: torch.Tensor,
    src_stage: int,
    tgt_stage: int,
    G: CharacterGenerator,
    cfg: Config,
    device: torch.device,
    n_samples: int = 1,
) -> List[torch.Tensor]:
    """
    Direct (skip-step) prediction from any stage to any other stage.
    Generates n_samples diverse outputs (different noise vectors).

    Returns: list of n_samples tensors each (1, 1, H, W).
    """
    G.eval()
    if src_image.dim() == 3:
        src_image = src_image.unsqueeze(0)
    src_image = src_image.to(device)
    s = torch.tensor([src_stage], dtype=torch.long, device=device)
    t = torch.tensor([tgt_stage], dtype=torch.long, device=device)
    results = [G(src_image, s, t).cpu() for _ in range(n_samples)]
    return results


# ── Visualisation helper ──────────────────────────────────────────────────────

def sequence_to_grid(sequences: List[List[torch.Tensor]],
                     stage_names: List[str]) -> "PIL.Image.Image":
    """
    Arrange multiple sequences into a side-by-side grid image.
    sequences: list of lists of (1, 1, H, W) tensors
    Returns: PIL image suitable for saving / display
    """
    from torchvision.utils import make_grid
    import torchvision.transforms.functional as TF

    all_imgs = []
    for seq in sequences:
        for img in seq:
            t = img.squeeze(0)                  # (1, H, W)
            all_imgs.append(t * 0.5 + 0.5)     # → [0, 1]

    n_cols = max(len(s) for s in sequences)
    grid = make_grid(all_imgs, nrow=n_cols, padding=2, pad_value=1.0)
    return TF.to_pil_image(grid)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from torchvision.utils import save_image

    parser = argparse.ArgumentParser(description="Chinese character evolution inference")
    sub = parser.add_subparsers(dest="mode")

    p_anc = sub.add_parser("ancient", help="Modern → Ancient")
    p_anc.add_argument("char", type=str, help="Unicode Chinese character")
    p_anc.add_argument("--checkpoint", required=True)
    p_anc.add_argument("--output", default="ancient.png")
    p_anc.add_argument("--device", default="cuda")

    p_mod = sub.add_parser("modern", help="Ancient → Modern")
    p_mod.add_argument("image", type=str, help="Path to OBS image")
    p_mod.add_argument("--checkpoint", required=True)
    p_mod.add_argument("--output", default="modern.png")
    p_mod.add_argument("--device", default="cuda")

    args = parser.parse_args()

    cfg    = Config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    G = CharacterGenerator(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ckpt["G"])

    if args.mode == "ancient":
        seq = predict_ancient(args.char, G, cfg, device)
        grid = torch.cat([t * 0.5 + 0.5 for t in seq], dim=-1)  # horizontal strip
        save_image(grid.squeeze(0), args.output)
        print(f"Saved {len(seq)}-stage sequence to {args.output}")

    elif args.mode == "modern":
        tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        obs = tf(Image.open(args.image).convert("L")).unsqueeze(0)
        seq = predict_modern(obs, G, cfg, device)
        grid = torch.cat([t * 0.5 + 0.5 for t in seq], dim=-1)
        save_image(grid.squeeze(0), args.output)
        print(f"Saved {len(seq)}-stage sequence to {args.output}")
