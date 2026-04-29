"""
Evaluation metrics for character evolution GAN.

SDR (Stage Deviation Rationality) — analogous to DLAT's IDAG metric.
  IDAG asks: "as the face-age gap increases, does identity deviation
  increase proportionally relative to real data?"
  SDR asks:  "as the stage gap increases, does visual dissimilarity
  increase proportionally relative to ground-truth character sequences?"

Additional metrics:
  - complexity_weighted_fid  : FID weighted by inverse character complexity
  - monotonicity_score       : fraction of stage pairs where dissimilarity
                               from stage 0 increases monotonically
  - predictor_accuracy       : % of generated images classified correctly
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional


# ── Per-image complexity ──────────────────────────────────────────────────────

def edge_density_np(img: np.ndarray) -> float:
    """
    Sobel edge density for a single HxW image in [0, 1].
    Lower → simpler character (fewer strokes, less change → harder to evaluate).
    """
    from scipy import ndimage
    gx = ndimage.sobel(img, axis=0)
    gy = ndimage.sobel(img, axis=1)
    return float(np.hypot(gx, gy).mean())


# ── Stage Deviation Rationality ───────────────────────────────────────────────

def pairwise_cosine(sequence: List[torch.Tensor]) -> np.ndarray:
    """
    Compute n×n cosine-similarity matrix for a sequence of images.
    sequence: list of (B, 1, H, W) tensors or single (1, H, W) tensors.
    Returns:  (n, n) float array averaged over the batch.
    """
    n = len(sequence)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a = sequence[i].float().flatten(1)   # (B, D)
            b = sequence[j].float().flatten(1)
            sim = F.cosine_similarity(a, b, dim=1).mean().item()
            mat[i, j] = sim
    return mat


def stage_deviation_rationality(
    generated_sequence: List[torch.Tensor],
    reference_matrix: Optional[np.ndarray] = None,
) -> dict:
    """
    SDR score for a generated character sequence.

    Args:
        generated_sequence : list of (1, H, W) or (B, 1, H, W) tensors,
                             one per stage in chronological order.
        reference_matrix   : (n_stages, n_stages) expected pairwise
                             similarity matrix from real data.
                             If None, returns only the observed matrix.

    Returns dict with:
        'similarity_matrix' : observed pairwise cosine similarities
        'sdr_kl'            : KL divergence vs. reference (if provided)
        'monotonicity'      : fraction of adjacent pairs where
                             dissimilarity from stage-0 increases
    """
    obs = pairwise_cosine(generated_sequence)
    n = len(generated_sequence)

    # Monotonicity: diss from stage 0 should increase with stage index
    diss = 1.0 - obs[0, :]         # dissimilarity from Oracle Bone
    mono_count = sum(diss[i] <= diss[i + 1] for i in range(n - 1))
    mono = mono_count / max(n - 1, 1)

    result = {
        "similarity_matrix": obs,
        "monotonicity": float(mono),
    }

    if reference_matrix is not None:
        mask = ~np.eye(n, dtype=bool)
        p = obs[mask].clip(1e-8, 1.0)
        q = reference_matrix[mask].clip(1e-8, 1.0)
        p /= p.sum(); q /= q.sum()
        kl = float(np.sum(p * np.log(p / q)))
        result["sdr_kl"] = kl

    return result


def build_reference_matrix(dataset, n_chars: int = 200,
                            image_size: int = 64) -> np.ndarray:
    """
    Compute expected pairwise cosine-similarity matrix from real data.
    Used once after dataset loading; store and reuse for evaluation.
    """
    import random
    from torchvision import transforms
    from PIL import Image

    tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    n_stages = 5
    sum_mat = np.zeros((n_stages, n_stages))
    count_mat = np.zeros((n_stages, n_stages))

    sample_chars = random.sample(dataset.chars, min(n_chars, len(dataset.chars)))
    for char in sample_chars:
        avail = char["stages"]
        keys = list(avail.keys())
        for i in keys:
            for j in keys:
                if i == j:
                    continue
                pi = random.choice(avail[i])
                pj = random.choice(avail[j])
                try:
                    ti = tf(Image.open(pi).convert("L")).flatten().unsqueeze(0)
                    tj = tf(Image.open(pj).convert("L")).flatten().unsqueeze(0)
                    sim = F.cosine_similarity(ti.float(), tj.float()).item()
                    sum_mat[i, j] += sim
                    count_mat[i, j] += 1
                except Exception:
                    pass

    ref = np.where(count_mat > 0, sum_mat / (count_mat + 1e-8), 0.0)
    np.fill_diagonal(ref, 1.0)
    return ref


# ── Predictor accuracy ────────────────────────────────────────────────────────

@torch.no_grad()
def predictor_accuracy(predictor, loader, device) -> float:
    """Fraction of real images correctly classified by StagePredictor."""
    predictor.eval()
    correct = total = 0
    for batch in loader:
        for stage_idx in range(5):
            imgs = batch["tgt_img"][batch["tgt_stage"] == stage_idx].to(device)
            if len(imgs) == 0:
                continue
            labels = torch.full((len(imgs),), stage_idx, dtype=torch.long,
                                device=device)
            preds = predictor(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += len(imgs)
    return correct / max(total, 1)


# ── FID helper ────────────────────────────────────────────────────────────────

def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """
    Fréchet distance between Gaussian fits to real and fake feature sets.
    real_feats / fake_feats: (N, D) arrays of extracted features.
    """
    from scipy.linalg import sqrtm

    mu_r, mu_f = real_feats.mean(0), fake_feats.mean(0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_f = np.cov(fake_feats, rowvar=False)

    diff = mu_r - mu_f
    cov_mean = sqrtm(sigma_r @ sigma_f)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * cov_mean))
    return fid
