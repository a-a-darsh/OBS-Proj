"""
CharacterEvolutionDataset

Supports two directory layouts, detected automatically per character folder:

Era-subfolder layout (GBOBC, IEHCV, deciphered):
  Dataset/
    Ai_01/
      oracle_bone_script/Ai_011.png
      bronze_script/Ai_012.png
      small_seal_script/Ai_013.png
      clerical_script/Ai_014.png
      modern/Ai_015.png
    U+3401/
      oracle_bone_script/...

Flat prefix layout (legacy OBC-Dataset):
  Dataset/
    00011/
      O_G_㐁_後2.36.5合33075...png   ← Oracle Bone
      J_奚子㐁車鼎(金)春秋早期.png   ← Bronze Inscription
      Z_說文‧五部.png                  ← Small Seal / Shuowen
      K_楷体.png                       ← Regular Script (modern)
    00014/ ...

Stage mapping:
  oracle_bone_script / O → stage 0
  bronze_script      / B → stage 1
  small_seal_script  / S → stage 2
  clerical_script    / C → stage 3
  modern             / M → stage 4
  chu_bamboo_silk_script / A → stage 5

Each __getitem__ returns one (src_stage, tgt_stage) pair sampled from
the available stages of a single character folder.  Missing stages are
handled naturally — only available-stage pairs are generated.
"""
import os
import re
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

PREFIX_TO_STAGE = {"O": 0, "B": 1, "S": 2, "C": 3, "M": 4, "A": 5, "L": 6}

ERA_FOLDER_TO_STAGE = {
    "oracle_bone_script":     0,
    "bronze_script":          1,
    "small_seal_script":      2,
    "clerical_script":        3,
    "modern":                 4,
    "chu_bamboo_silk_script": 5,
    "Liushutong characters 六书通的字": 6

}

# JSON path — used to build a number→character lookup
_DEFAULT_JSON = r"C:\Users\slitf\Downloads\List_of_EVOBC.json"


def _build_number_to_char(json_path: str) -> dict:
    """
    Parse List_of_EVOBC.json to build a mapping from the image number
    embedded in every filename to the Unicode character for that entry.

    JSON filename format:  {ID}_Book_{Type}_{subtype}_{number}.ext
    Dataset filename format: {prefix}_{subtype}_{char}_{number}.ext

    Both share the trailing number, so we use it as the join key.
    """
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    num_to_char: dict = {}
    for entry in data:
        char = entry.get("Character", "")
        if not char:
            continue
        for img in entry.get("images", []):
            fname = img.get("file", "")
            # Extract the last numeric token before the extension
            m = re.search(r"_(\d+)(?:_\d+)*\.\w+$", fname)
            if m:
                num_to_char[m.group(1)] = char
    return num_to_char


def _extract_unicode_from_folder(filenames: list,
                                 num_to_char: dict) -> str:
    """
    Extract the Unicode character for a dataset folder using two strategies:
      1. Direct: parse single CJK char from O_{subtype}_{char}_{...} filenames.
      2. JSON lookup: find the trailing number in any filename and look it up.
    """
    # Strategy 1 — fast, works for O_ files with clean naming
    for fname in filenames:
        if not fname.startswith("O_"):
            continue
        parts = fname.split("_")
        if len(parts) >= 3:
            c = parts[2]
            if len(c) == 1 and ord(c) > 0x2E7F:
                return c

    # Strategy 2 — JSON lookup via trailing number
    if num_to_char:
        for fname in filenames:
            m = re.search(r"_(\d{3,})(?:_\d+)*\.\w+$", fname)
            if m:
                char = num_to_char.get(m.group(1), "")
                if char:
                    return char
    return ""


def _char_from_folder_name(name: str) -> str:
    """Return the Unicode character if the folder is named U+XXXX[+U+YYYY...]."""
    if not name.startswith("U+"):
        return ""
    try:
        return "".join(chr(int(seg, 16)) for seg in name.split("+") if seg and seg != "U")
    except ValueError:
        return ""


def _invert_if_dark(img: Image.Image) -> Image.Image:
    """Oracle bone photos often have dark backgrounds; normalise to white background."""
    pixels = list(img.getdata())
    if sum(pixels) / len(pixels) < 127:
        img = img.point(lambda p: 255 - p)
    return img


class CharacterEvolutionDataset(Dataset):
    """
    Loads character evolution data.  Each sample is a (src, tgt) stage pair
    from the same character folder.

    Args:
        data_dir   : path to OBC-Dataset/Dataset/
        image_size : resize target (square)
        split      : "train" or "val"
        val_fraction: fraction of characters held out for validation
        seed       : random seed for reproducible split
    """

    def __init__(self, data_dir: str, image_size: int = 64,
                 split: str = "train", val_fraction: float = 0.05,
                 seed: int = 42, max_refs: int = 4,
                 num_stages: int = 7,
                 min_pair_count: int = 10,
                 json_path: str = _DEFAULT_JSON):
        self.max_refs = max_refs
        self.image_size = image_size
        self.num_stages = num_stages

        _base = [
            transforms.Grayscale(),
            transforms.Resize((image_size, image_size),
                               interpolation=transforms.InterpolationMode.BICUBIC),
        ]
        _to_tensor = [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
        ]
        if split == "train":
            self.transform = transforms.Compose(_base + [
                transforms.RandomApply([
                    transforms.RandomRotation(degrees=5, fill=255),
                ], p=0.4),
                transforms.RandomApply([
                    # Crop then resize back — simulates centering variation
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.85, 1.0), ratio=(0.92, 1.08),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                ], p=0.4),
            ] + _to_tensor)
        else:
            self.transform = transforms.Compose(_base + _to_tensor)

        num_to_char = _build_number_to_char(json_path)

        # ── Index characters ──────────────────────────────────────────
        chars: list = []
        for folder_name in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            stage_files: dict = {}
            entries = os.listdir(folder_path)

            # Detect era-subfolder layout (GBOBC / IEHCV / deciphered)
            era_subdirs = [
                e for e in entries
                if e in ERA_FOLDER_TO_STAGE
                and os.path.isdir(os.path.join(folder_path, e))
            ]

            if era_subdirs:
                for era_name in era_subdirs:
                    stage_idx = ERA_FOLDER_TO_STAGE[era_name]
                    era_path = os.path.join(folder_path, era_name)
                    files = [
                        os.path.join(era_path, f)
                        for f in os.listdir(era_path)
                        if os.path.isfile(os.path.join(era_path, f))
                    ]
                    if files:
                        stage_files.setdefault(stage_idx, []).extend(files)
                unicode_char = _char_from_folder_name(folder_name)
            else:
                # Flat prefix layout (legacy OBC-Dataset)
                flat_files = [e for e in entries if os.path.isfile(os.path.join(folder_path, e))]
                for fname in flat_files:
                    prefix = fname[0] if fname else ""
                    if prefix not in PREFIX_TO_STAGE:
                        continue
                    stage_idx = PREFIX_TO_STAGE[prefix]
                    stage_files.setdefault(stage_idx, []).append(
                        os.path.join(folder_path, fname)
                    )
                unicode_char = _extract_unicode_from_folder(flat_files, num_to_char)

            available = {k: v for k, v in stage_files.items() if v}
            if len(available) < 2:
                continue   # need at least two stages to form a pair

            chars.append({
                "id":      folder_name,
                "char":    unicode_char,
                "stages":  available,
            })

        # ── Train / val split ─────────────────────────────────────────
        rng = random.Random(seed)
        rng.shuffle(chars)
        n_val = max(1, int(len(chars) * val_fraction))
        self.chars = chars[n_val:] if split == "train" else chars[:n_val]

        # ── Build all valid (src_stage, tgt_stage) pairs ──────────────
        # All forward pairs (s < t): any earlier era → any later era.
        from collections import Counter
        raw_pairs: list = []
        for char_idx, char in enumerate(self.chars):
            avail = sorted(char["stages"].keys())
            for i, s in enumerate(avail):
                for t in avail[i + 1:]:
                    raw_pairs.append((char_idx, s, t))

        # Count samples per (s, t) bucket; drop buckets below threshold.
        bucket_counts = Counter((s, t) for _, s, t in raw_pairs)
        valid_buckets = {k for k, v in bucket_counts.items() if v >= min_pair_count}
        self.pairs = [(c, s, t) for c, s, t in raw_pairs if (s, t) in valid_buckets]

        # Weight each sample so every surviving bucket is equally likely.
        self.sample_weights = [
            1.0 / bucket_counts[(s, t)] for _, s, t in self.pairs
        ]

    # ── Helpers ───────────────────────────────────────────────────────
    def _load(self, path: str) -> torch.Tensor:
        img = Image.open(path)
        if img.mode in ("RGBA", "LA", "PA"):
            # Composite onto white before dropping alpha — otherwise transparent
            # pixels with black RGB become black in "L" mode, causing _invert_if_dark
            # to flip the whole image white (strokes and background alike).
            bg = Image.new("L", img.size, 255)
            alpha = img.convert("RGBA").split()[3]
            bg.paste(img.convert("L"), mask=alpha)
            img = bg
        else:
            img = img.convert("L")
        img = _invert_if_dark(img)
        return self.transform(img)

    def char_info(self, idx: int) -> dict:
        """Return metadata for a character index (used by inference)."""
        return self.chars[idx]

    # ── Dataset interface ─────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        char_idx, s, t = self.pairs[idx]
        char = self.chars[char_idx]

        # Load up to max_refs source images; pad remainder with zeros
        src_paths = char["stages"][s]
        if len(src_paths) > self.max_refs:
            src_paths = random.sample(src_paths, self.max_refs)
        src_imgs = torch.zeros(self.max_refs, 1, self.image_size, self.image_size)
        src_mask = torch.zeros(self.max_refs, dtype=torch.bool)
        for i, p in enumerate(src_paths):
            src_imgs[i] = self._load(p)
            src_mask[i] = True

        tgt_img = self._load(random.choice(char["stages"][t]))

        # Availability mask: which stages this character has ground-truth for
        stage_mask = torch.zeros(self.num_stages, dtype=torch.bool)
        for k in char["stages"]:
            stage_mask[k] = True

        return {
            "src_imgs":  src_imgs,                             # (max_refs, 1, H, W)
            "src_mask":  src_mask,                             # (max_refs,) bool
            "tgt_img":   tgt_img,                              # (1, H, W)
            "src_stage": torch.tensor(s, dtype=torch.long),
            "tgt_stage": torch.tensor(t, dtype=torch.long),
            "mask":      stage_mask,
            "char_id":   char_idx,
        }
