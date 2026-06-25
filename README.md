# CharacterEvolutionGAN

A bidirectional conditional generative model that morphs Chinese characters across **7 historical script stages spanning 2,000+ years** — from modern Regular Script all the way back to ancient Oracle Bone Script, and forward again.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange) ![License](https://img.shields.io/badge/license-MIT-green)
<img width="2100" height="803" alt="image" src="https://github.com/user-attachments/assets/04c6be4a-81ed-438c-9426-eef7ec595165" />


---

## Overview

Chinese writing has evolved dramatically over its 3,000-year history. A single character written today shares ancestry with markings carved onto oracle bones during the Shang Dynasty — yet visually they look almost nothing alike.

This project trains a GAN-based architecture to learn that transformation end-to-end. Given any Chinese character at any historical stage, the model can generate a plausible rendering of that same character at any other stage, in either temporal direction.

**Supported script stages:**

| Stage | Script | Period |
|-------|--------|--------|
| 0 | Oracle Bone Script (甲骨文) | ~1250–1050 BCE |
| 1 | Bronze Inscription (金文) | ~1046–771 BCE |
| 2 | Spring & Autumn / Chu Bamboo Script | ~771–500 BCE |
| 3 | Small Seal Script (小篆) | ~300–0 BCE |
| 4 | Liushutong / Late Seal | ~100 BCE–20 CE |
| 5 | Clerical Script (隶书) | ~200–600 CE |
| 6 | Regular Script / Modern (楷书) | ~1100–1500 CE |

---

## Architecture

The model follows a **content/style disentanglement** design inspired by StyleGAN2, adapted for conditional cross-domain image translation.

```
Source Character(s)
      │
      ▼
 ContentEncoder ──────────────────────────────────────────┐
 (4× stride-2 conv + self-attention)                      │ skip connections
      │                                                    │ (attention-gated)
      ▼                                                    │
 content features (8×8)                                   │
                                                          ▼
 Noise + Stage Embeddings + Fourier Year Embeddings → StageMapper (8-layer MLP)
                                                          │
                                                          │ style codes (12 × 256)
                                                          ▼
                                              StyledDecoder (StyleGAN2-style)
                                              ModulatedConv2d + noise injection
                                              4 residual + 4 upsampling blocks
                                                          │
                                                          ▼
                                               Generated Character (128×128)
```

**Key components:**

- **ContentEncoder** — Extracts structural identity of the character (stroke topology, spatial layout) independently of era-specific style. Produces skip connections at 4 resolution levels.

- **StageMapper** — An 8-layer MLP mapping noise + source/target stage embeddings + continuous Fourier year embeddings into per-layer style codes. Handles the temporal conditioning.

- **StyledDecoder** — A ResNet-style decoder using StyleGAN2's modulated convolutions. Style codes modulate each layer independently, enabling fine-grained control over era-specific texture while preserving content from the encoder.

- **AttentionGate** — Applied to all encoder skip connections before fusion. Suppresses era-specific texture in skip features, preventing the source style from leaking into the target output.

- **MultiStageDiscriminator** — 7 parallel PatchGAN branches, one per stage. Each branch receives the source + target image pair and predicts realism conditioned on source stage embedding.

- **StagePredictor** — An auxiliary classifier trained on real images to enforce that generated characters are classifiable as the intended target stage.

---

## Loss Functions

Training balances nine loss terms to enforce both perceptual quality and character-specific topological fidelity:

| Loss | Weight | Purpose |
|------|--------|---------|
| Adversarial (non-saturating) | 6.0 | Realism vs. discriminator |
| R1 gradient penalty (lazy) | 4.0 | Discriminator regularization |
| Cycle consistency | 8.0 | Round-trip identity preservation |
| Stage consistency | 5.0 | Output classifiable as target era |
| Reconstruction (complexity-weighted) | 8.0 | Pixel-level fidelity; upweights simple chars |
| Perceptual (VGG relu2_2 + relu3_3) | 4.0 | High-level feature alignment |
| Stroke loss (multi-scale Sobel) | 1.0 | Stroke edge fidelity |
| Skeleton loss (morphological) | 2.0 | Stroke topology / connectivity |
| Content loss | 6.0 | Encoder-space structural consistency |

The **stroke** and **skeleton** losses are domain-specific additions not present in standard image translation pipelines. Stroke topology — the connectivity, branching, and relative weight of strokes — is a defining feature of each historical script, making these losses critical for character authenticity.

The **complexity-weighted reconstruction** loss addresses a common failure mode: characters that change minimally across eras tend to have trivial gradient signal, causing the generator to collapse to blurry averages. Weighting by Sobel edge density counteracts this.

---

## Inference Modes

```python
# Modern → Ancient (stepwise chain: 6 → 5 → 4 → ... → 0)
images = predict_ancient('水')  # Unicode character string

# Ancient → Modern (stepwise chain: 0 → 1 → 2 → ... → 6)
images = predict_modern(oracle_bone_image)

# Direct (any src stage → any tgt stage, no chaining)
images = predict_direct(src_img, src_stage=6, tgt_stage=0, n_samples=4)
```

Stepwise chain inference yields higher quality at the cost of more forward passes. Direct inference is faster and supports sampling multiple diverse outputs (different noise vectors) to visualize the distribution over plausible historical forms.

---

## Training

```bash
python train.py
```

**Key hyperparameters** (`config.py`):

```python
image_size      = 128          # 128×128 grayscale
num_stages      = 7
nf              = 64           # base channel count
style_dim       = 256          # style code dimension
latent_dim      = 256          # noise vector dimension
n_style_layers  = 12           # style vectors per forward pass
n_mapper_layers = 8            # StageMapper MLP depth
batch_size      = 32
n_epochs        = 2000
lr_g            = 2e-4
lr_d            = 1e-4
use_amp         = True         # BF16 mixed precision
r1_every        = 8            # lazy R1 frequency
```

**Optimizers:** Adam with β₁=0.0, β₂=0.99 (standard for GAN stability). Gradient clipping at max norm 1.0. BF16 autocast for throughput.

---

## Dataset

The model trains on multiple Chinese historical character datasets with two supported layouts:

**Era-subfolder format:**
```
Dataset/
  character_id/
    oracle_bone_script/img.png
    bronze_script/img.png
    ...
    modern/img.png
```

**Flat prefix format** (legacy OBC-Dataset):
```
Dataset/
  00011/
    O_G_㐁_後2.36.5合33075.png    ← Oracle Bone (O prefix)
    J_㐁_奚子车鼎.png              ← Bronze (J prefix → stage 1)
```

**Preprocessing:** Grayscale, white background compositing for alpha channels, automatic inversion for dark-background oracle bone photographs, 128×128 bicubic resize, normalized to `[-1, 1]`.

**Augmentation** (training only, applied consistently across image pairs): random rotation ±5°, random resized crop (scale 0.85–1.0).

**Sampling:** All valid (src, tgt) stage pairs generated per character, with weighted sampling to balance under-represented era pairs. Identity pairs (same stage) capped at ~10% of training pairs.

---

## Technical Stack

- **PyTorch 2.x** — model, training loop, BF16 autocast
- **torchvision** — VGG16 for perceptual loss
- **Pillow** — image I/O, CJK font rendering for inference
- **tqdm** — training progress

---

## Design Highlights

**Bidirectional conditioning:** Both source and target stage embeddings (plus continuous Fourier year embeddings) feed into the style mapper. This lets the model reason jointly about where the character is coming *from* and where it's going, enabling a single model to handle all 42 directed stage pairs without separate models per direction.

**Continuous temporal embedding:** Historical stages are not discrete points — they span centuries of gradual change. Fourier embeddings of the year value within each stage give the model a continuous temporal signal, so it can in principle interpolate within an era, not just between discrete class labels.

**Content/style separation with attention-gated skips:** Standard U-Net skip connections carry both content and style from source to target, causing source era texture to leak. AttentionGates use the decoder state to selectively filter each skip connection, preserving structural information while suppressing era-specific appearance.

**Lazy R1 regularization:** Computing R1 requires a second backward pass through the discriminator. Applying it every 8 steps instead of every step cuts training cost significantly with negligible impact on stability — the same amortization trick used in StyleGAN2.
