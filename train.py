"""
Training loop for CharacterEvolutionGAN.

Per-step logic:
  D step:
    1. real_pred  = D(tgt_img, tgt_stage)          ← real character at target stage
    2. fake_tgt   = G(src_img, src→tgt) [no grad]
    3. fake_pred  = D(fake_tgt, tgt_stage)
    4. d_loss     = adv_loss_d(real, fake) [+ lazy R1]
    5. P_real     = CE(P(tgt_img), tgt_stage)       ← train predictor on real imgs

  G step:
    1. fake_tgt   = G(src_img, src→tgt, noise)
    2. g_adv      = adv_loss_g(D(fake, tgt))
    3. g_stage    = CE(P(fake), tgt_stage)          ← consistency with predictor
    4. g_cycle    = L1(G(fake, tgt→src), src)       ← round-trip
    5. g_div      = −L1(G(src,n1), G(src,n2))       ← diversity
    6. g_recon    = complexity_weighted_recon(fake, tgt)

Multi-directional: every batch contains randomly-sampled (src, tgt) pairs
that can span ANY two stages in either direction.  No unidirectional bias.
"""
import os
import random
import argparse
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from model import CharacterGenerator, MultiStageDiscriminator, StagePredictor
from data import CharacterEvolutionDataset
from losses import (
    adv_loss_g, adv_loss_d, r1_penalty,
    cycle_loss, diversity_loss, PerceptualLoss,
    complexity_weighted_recon, stage_consistency_loss,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_checkpoint(G, D, P, opt_G, opt_D, opt_P, epoch: int, cfg: Config):
    path = os.path.join(cfg.checkpoint_dir, f"ckpt_epoch{epoch:04d}.pt")
    torch.save({
        "epoch": epoch,
        "G": G.state_dict(), "D": D.state_dict(), "P": P.state_dict(),
        "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
        "opt_P": opt_P.state_dict(),
    }, path)
    print(f"  → saved {path}")


@torch.no_grad()
def save_samples(G: CharacterGenerator, val_set: CharacterEvolutionDataset,
                 epoch: int, cfg: Config, device: torch.device):
    """
    For 8 validation characters: show all available stages and generated
    conversions in a grid.
    """
    G.eval()
    rows = []
    rich_chars = [c for c in val_set.chars if len(c["stages"]) >= 4]
    n_show = min(8, len(rich_chars))
    for char_info in rich_chars[:n_show]:
        available = char_info["stages"]

        # Show real images for each stage (blank if missing)
        real_row = []
        for s in range(cfg.num_stages):
            if s in available:
                img = val_set._load(random.choice(available[s]))
                real_row.append(img.to(device))
            else:
                real_row.append(torch.ones(1, cfg.image_size, cfg.image_size,
                                           device=device))   # white placeholder

        # Generate the full sequence from the earliest available stage (matches training)
        earliest = min(available.keys())
        src_img = val_set._load(random.choice(available[earliest])).unsqueeze(0).to(device)
        src_img_refs = src_img.unsqueeze(1)  # (1, 1, C, H, W)
        src_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
        src_stage_t = torch.tensor([earliest], device=device)

        gen_row = []
        for t in range(cfg.num_stages):
            tgt_stage_t = torch.tensor([t], device=device)
            fake = G(src_img_refs, src_stage_t, tgt_stage_t, src_mask=src_mask)
            gen_row.append(fake.squeeze(0))

        rows.append(torch.stack(real_row))
        rows.append(torch.stack(gen_row))

    grid = torch.cat(rows, dim=0)
    path = os.path.join(cfg.sample_dir, f"samples_epoch{epoch:04d}.png")
    os.makedirs(cfg.sample_dir, exist_ok=True)
    save_image(grid * 0.5 + 0.5, path, nrow=cfg.num_stages)
    G.train()


# ── Main training loop ────────────────────────────────────────────────────────

def load_checkpoint(path: str, G, D, P, opt_G, opt_D, opt_P, device) -> int:
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    P.load_state_dict(ckpt["P"])
    for opt, key in [(opt_G, "opt_G"), (opt_D, "opt_D"), (opt_P, "opt_P")]:
        if key not in ckpt:
            continue
        try:
            opt.load_state_dict(ckpt[key])
        except ValueError:
            print(f"  ! skipping {key} state (parameter group mismatch — optimizer will restart)")
    print(f"  ← resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


def train(cfg: Config, resume: str | None = None):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.sample_dir, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_set = CharacterEvolutionDataset(cfg.data_dir, cfg.image_size,
                                          split="train", max_refs=cfg.max_refs,
                                          num_stages=cfg.num_stages,
                                          min_pair_count=cfg.min_pair_count)
    val_set   = CharacterEvolutionDataset(cfg.data_dir, cfg.image_size,
                                          split="val", max_refs=cfg.max_refs,
                                          num_stages=cfg.num_stages,
                                          min_pair_count=cfg.min_pair_count)
    sampler = WeightedRandomSampler(
        weights=train_set.sample_weights,
        num_samples=len(train_set),
        replacement=True,
    )
    loader = DataLoader(train_set, batch_size=cfg.batch_size, sampler=sampler,
                        num_workers=cfg.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=cfg.num_workers > 0)
    print(f"Train pairs: {len(train_set):,}  Val chars: {len(val_set.chars):,}")

    # ── Models ────────────────────────────────────────────────────────
    G = CharacterGenerator(cfg).to(device)
    D = MultiStageDiscriminator(cfg).to(device)
    P = StagePredictor(cfg).to(device)

    # torch.compile fuses kernels for faster execution.
    # Warmup takes a few minutes on the first epoch — normal.
    if cfg.use_amp and hasattr(torch, "compile"):
        G = torch.compile(G)
        P = torch.compile(P)
        # D skipped: R1 penalty uses create_graph=True (double backward),
        # which torch.compile does not support.

    perceptual = PerceptualLoss(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
    opt_P = torch.optim.Adam(P.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))

    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(resume, G, D, P, opt_G, opt_D, opt_P, device)

    d_step = 0
    # BF16 autocast kwargs — passed to autocast() each step.
    # BF16 has the same exponent range as FP32 so no GradScaler is needed.
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_amp)

    for epoch in tqdm(range(start_epoch + 1, cfg.n_epochs + 1), desc="Epochs"):
        G.train()
        D.train()
        P.train()
        epoch_d_loss = epoch_g_loss = 0.0
        n_batches = 0

        for batch in (pbar := tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
            src_imgs = batch["src_imgs"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_img = batch["tgt_img"].to(device)
            src_stage = batch["src_stage"].to(device)
            tgt_stage = batch["tgt_stage"].to(device)
            B = src_imgs.shape[0]

            # Pixel mean of valid refs — used as cycle-consistency target
            mask_f = src_mask.float().view(B, src_imgs.shape[1], 1, 1, 1)
            mean_src = (src_imgs * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

            # ── Discriminator step ──────────────────────────────
            with autocast(**amp):
                with torch.no_grad():
                    fake_tgt_d = G(src_imgs, src_stage, tgt_stage, src_mask=src_mask)
                real_pred = D(tgt_img, tgt_stage)
                fake_pred = D(fake_tgt_d.detach(), tgt_stage)
                d_loss = cfg.lambda_adv * adv_loss_d(real_pred, fake_pred)

            # R1 runs outside autocast: create_graph=True requires float32 precision
            if d_step % cfg.r1_every == 0:
                tgt_img_r1 = tgt_img.detach().requires_grad_(True)
                r1 = r1_penalty(D(tgt_img_r1, tgt_stage), tgt_img_r1)
                d_loss = d_loss + (cfg.lambda_r1 / 2) * r1

            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()
            d_step += 1

            # ── Predictor step (real images only) ───────────────
            with autocast(**amp):
                p_loss = F.cross_entropy(P(tgt_img.detach()), tgt_stage)
            opt_P.zero_grad(set_to_none=True)
            p_loss.backward()
            opt_P.step()

            # ── Generator step ──────────────────────────────────
            noise1 = torch.randn(B, cfg.latent_dim, device=device)
            noise2 = torch.randn(B, cfg.latent_dim, device=device)

            with autocast(**amp):
                fake_tgt = G(src_imgs, src_stage, tgt_stage, noise1, src_mask=src_mask)
                fake_pred = D(fake_tgt, tgt_stage)

                # 1. Adversarial
                g_loss = cfg.lambda_adv * adv_loss_g(fake_pred)

                # 2. Stage consistency — gradient flows through G only; P not updated
                g_loss += cfg.lambda_stage * stage_consistency_loss(P, fake_tgt, tgt_stage)

                # 3. Cycle consistency
                recon_src = G(fake_tgt, tgt_stage, src_stage, noise1)
                g_loss += cfg.lambda_cycle * cycle_loss(recon_src, mean_src)
                g_loss += cfg.lambda_percep * perceptual(recon_src, mean_src)

                # 4. Diversity  (two noise samples3 should differ)
                fake_tgt2 = G(src_imgs, src_stage, tgt_stage, noise2, src_mask=src_mask)
                g_loss += cfg.lambda_div * diversity_loss(fake_tgt, fake_tgt2)

                # 5. Complexity-weighted reconstruction + perceptual vs. ground-truth target
                g_loss += cfg.lambda_recon * complexity_weighted_recon(fake_tgt, tgt_img)
                g_loss += cfg.lambda_percep * perceptual(fake_tgt, tgt_img)

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            n_batches += 1
            pbar.set_postfix(D=f"{epoch_d_loss/n_batches:.4f}",
                             G=f"{epoch_g_loss/n_batches:.4f}")

        tqdm.write(f"Epoch {epoch:4d}/{cfg.n_epochs}  "
                   f"D={epoch_d_loss / n_batches:.4f}  "
                   f"G={epoch_g_loss / n_batches:.4f}")

        if epoch % cfg.sample_every == 0:
            save_samples(G, val_set, epoch, cfg, device)

        if epoch % cfg.save_every == 0:
            save_checkpoint(G, D, P, opt_G, opt_D, opt_P, epoch, cfg)

    print("Training complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--lr_g", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--sample_dir", default=None)
    parser.add_argument("--resume", default=None,
                        help="path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = Config()
    for k, v in vars(args).items():
        if k != "resume" and v is not None:
            setattr(cfg, k, v)

    train(cfg, resume=args.resume)
