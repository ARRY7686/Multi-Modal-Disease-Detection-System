"""
Dataset loading for honest multimodal training.

Two rules the original notebooks broke, enforced here:

1. **No label leakage through the text.** The text paired with each image
   comes from real recorded metadata (age, sex, X-ray view, lesion site),
   never from the diagnosis. HAM10000's `dx_type` column is deliberately
   unused: how a lesion was confirmed (histopathology vs follow-up) is
   decided by the diagnosis itself.
2. **Patient-disjoint splits.** NIH uses its official patient-disjoint
   test list and splits validation by Patient ID; HAM10000 is split by
   `lesion_id` (several photos per lesion); brain MRI uses the dataset's
   official Testing folder, with near-duplicates of training images removed.
"""

from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset

from api.config import Config
from api.preprocessing.text_processing import TextPreprocessor
from utils.augmentation import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

MODALITY_INDEX = {"chest": 0, "skin": 1, "brain": 2}
NO_DETAILS = "No clinical details provided."
SPLITS = ("train", "val", "test")

HAM_DX = {
    "mel": "Melanoma", "nv": "Melanocytic_Nevi", "bcc": "Basal_Cell_Carcinoma",
    "akiec": "Actinic_Keratosis", "bkl": "Benign_Keratosis",
    "df": "Dermatofibroma", "vasc": "Vascular_Lesion",
}
# Folder names in the Sartaj (…_tumor) and Masoud (glioma, notumor, …) releases
BRAIN_CLASSES = {
    "glioma_tumor": "Glioma", "glioma": "Glioma",
    "meningioma_tumor": "Meningioma", "meningioma": "Meningioma",
    "pituitary_tumor": "Pituitary_Tumor", "pituitary": "Pituitary_Tumor",
    "no_tumor": "No_Tumor", "notumor": "No_Tumor",
}


@dataclass
class Record:
    image_path: str
    labels: np.ndarray       # (num_classes,) multi-hot
    modality: str
    text: str                # metadata rendered as a short clinical note
    group: str               # patient / lesion id, for grouped splitting


# ── File discovery ───────────────────────────────────────────────────────────

def find_path(root: Path, name_pattern: str, want_dir: bool = False, max_depth: int = 5) -> Optional[Path]:
    """First file (or directory) under root whose name fully matches a regex (sorted depth-first walk)."""
    pattern = re.compile(name_pattern)
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth >= max_depth:
            dirnames[:] = []
        dirnames.sort()
        for name in sorted(dirnames if want_dir else filenames):
            if pattern.fullmatch(name):
                return Path(dirpath) / name
    return None


def index_files(root: Path, suffixes: Iterable[str]) -> Dict[str, str]:
    """Map file name -> full path for every file with one of `suffixes` under root."""
    suffixes = tuple(s.lower() for s in suffixes)
    index = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(suffixes):
                index[name] = os.path.join(dirpath, name)
    return index


# ── Metadata → text ──────────────────────────────────────────────────────────

def _age(value) -> Optional[int]:
    """Leading integer of '58', '058Y' or 45.0; None if missing or implausible."""
    match = re.match(r"\s*(\d+)", str(value))
    if not match:
        return None
    age = int(match.group(1))
    return age if 0 < age <= 100 else None  # NIH has a few entries > 100 (data errors)


def _sex(value) -> Optional[str]:
    return {"m": "male", "male": "male", "f": "female", "female": "female"}.get(str(value).strip().lower())


def _patient_sentence(age, sex) -> Optional[str]:
    """'58-year-old male.' with whatever parts are known, or None."""
    age, sex = _age(age), _sex(sex)
    words = [f"{age}-year-old" if age else None, sex]
    who = " ".join(w for w in words if w)
    return f"{who[0].upper()}{who[1:]}." if who else None


def chest_text(age, sex, view) -> str:
    parts = [_patient_sentence(age, sex)]
    view = str(view).strip().upper()
    if view in ("PA", "AP"):
        parts.append(f"{view} chest radiograph.")
    return " ".join(p for p in parts if p) or NO_DETAILS


def skin_text(age, sex, localization) -> str:
    parts = [_patient_sentence(age, sex)]
    site = str(localization).strip().lower()
    if site and site not in ("unknown", "nan"):
        parts.append(f"Lesion on the {site}.")
    return " ".join(p for p in parts if p) or NO_DETAILS


# ── Loaders ──────────────────────────────────────────────────────────────────

def _label_vector(cfg: Config, names: Sequence[str]) -> np.ndarray:
    vec = np.zeros(cfg.num_classes, dtype=np.float32)
    index = {n: i for i, n in enumerate(cfg.all_labels)}
    for n in names:
        vec[index[n]] = 1.0
    return vec


def _sample_groups(groups: Sequence[str], fraction: float, seed: int) -> set:
    unique = sorted(set(groups))
    if fraction >= 1.0:
        return set(unique)
    rng = random.Random(seed)
    return set(rng.sample(unique, max(1, round(len(unique) * fraction))))


def load_nih(
    cfg: Config,
    root: Path,
    train_fraction: float = 1.0,
    test_fraction: float = 1.0,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Record]]:
    """NIH ChestX-ray14 with the official patient-disjoint test split."""
    csv_path = find_path(root, r"Data_Entry_2017.*\.csv")
    test_list = find_path(root, r"test_list\.txt")
    if csv_path is None or test_list is None:
        raise FileNotFoundError(f"NIH Data_Entry_2017*.csv / test_list.txt not found under {root}")

    df = pd.read_csv(csv_path)
    test_names = set(Path(test_list).read_text().split())
    images = index_files(csv_path.parent, [".png"])
    chest = set(cfg.chest_labels)

    records: Dict[str, List[Record]] = {s: [] for s in SPLITS}
    is_test = df["Image Index"].isin(test_names)
    keep_test = _sample_groups(df.loc[is_test, "Patient ID"].astype(str), test_fraction, seed)
    trainval_patients = _sample_groups(df.loc[~is_test, "Patient ID"].astype(str), train_fraction, seed)
    val_patients = _sample_groups(sorted(trainval_patients), val_fraction, seed + 1)

    missing = 0
    columns = zip(
        df["Image Index"], df["Finding Labels"], df["Patient ID"].astype(str),
        df["Patient Age"], df["Patient Gender"], df["View Position"], strict=True,
    )
    for name, finding_labels, patient, age, sex, view in columns:
        if name not in images:
            missing += 1
            continue
        if name in test_names:
            if patient not in keep_test:
                continue
            split = "test"
        elif patient in trainval_patients:
            split = "val" if patient in val_patients else "train"
        else:
            continue
        # "No Finding" -> all-zero vector: normal X-rays are kept, not dropped
        findings = [f.strip() for f in str(finding_labels).split("|") if f.strip() in chest]
        records[split].append(Record(
            image_path=images[name],
            labels=_label_vector(cfg, findings),
            modality="chest",
            text=chest_text(age, sex, view),
            group=f"nih-{patient}",
        ))
    if missing:
        logger.warning("NIH: %d CSV rows had no image file on disk", missing)
    return records


def load_ham10000(
    cfg: Config,
    root: Path,
    test_fraction: float = 0.15,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Record]]:
    """HAM10000, split by lesion_id so photos of one lesion stay in one split."""
    csv_path = find_path(root, r"HAM10000_metadata\.csv")
    if csv_path is None:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found under {root}")
    df = pd.read_csv(csv_path)
    df = df[df["dx"].isin(HAM_DX)].reset_index(drop=True)
    images = {Path(n).stem: p for n, p in index_files(csv_path.parent, [".jpg"]).items()}
    df = df[df["image_id"].isin(images)].reset_index(drop=True)

    split_of = _grouped_split(df["dx"].tolist(), df["lesion_id"].tolist(), test_fraction, val_fraction, seed)
    records: Dict[str, List[Record]] = {s: [] for s in SPLITS}
    for i, row in df.iterrows():
        records[split_of[i]].append(Record(
            image_path=images[row["image_id"]],
            labels=_label_vector(cfg, [HAM_DX[row["dx"]]]),
            modality="skin",
            text=skin_text(row.get("age"), row.get("sex"), row.get("localization")),
            group=f"ham-{row['lesion_id']}",
        ))
    return records


def _grouped_split(labels, groups, test_fraction, val_fraction, seed) -> List[str]:
    """Stratified, group-disjoint train/val/test assignment for each row."""
    n = len(labels)
    out = ["train"] * n
    idx = np.arange(n)

    def take(indices, fraction, name, seed_offset):
        k = max(2, round(1 / fraction))
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + seed_offset)
        lab = [labels[i] for i in indices]
        grp = [groups[i] for i in indices]
        _, held = next(splitter.split(indices, lab, grp))
        for j in held:
            out[indices[j]] = name
        return np.array([indices[j] for j in range(len(indices)) if out[indices[j]] == "train"])

    rest = take(idx, test_fraction, "test", 0)
    take(rest, val_fraction / (1 - test_fraction), "val", 1)
    return out


def dhash(path: str, size: int = 8) -> int:
    """64-bit difference hash; equal hashes mean near-identical images."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def load_brain_mri(
    cfg: Config,
    root: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Record]]:
    """Brain tumour MRI: official Training/Testing folders, cross-split duplicates removed."""
    train_dir = find_path(root, r"Training", want_dir=True)
    test_dir = find_path(root, r"Testing", want_dir=True)
    if train_dir is None or test_dir is None:
        raise FileNotFoundError(f"Brain MRI Training/ and Testing/ folders not found under {root}")

    def scan(folder: Path):
        items = []
        for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            label = BRAIN_CLASSES.get(cls_dir.name.lower())
            if label is None:
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    items.append((str(f), label))
        return items

    train_items, test_items = scan(train_dir), scan(test_dir)
    train_hashes = {dhash(p) for p, _ in train_items}
    deduped = [(p, lab) for p, lab in test_items if dhash(p) not in train_hashes]
    if len(deduped) < len(test_items):
        logger.info("Brain MRI: dropped %d test images duplicated in Training", len(test_items) - len(deduped))

    tr, va = train_test_split(
        train_items, test_size=val_fraction, random_state=seed, stratify=[lab for _, lab in train_items],
    )
    records: Dict[str, List[Record]] = {}
    for split, items in (("train", tr), ("val", va), ("test", deduped)):
        records[split] = [
            # No patient IDs in this dataset; file names repeat across folders, so key by path
            Record(p, _label_vector(cfg, [lab]), "brain", NO_DETAILS, f"brain-{p}")
            for p, lab in items
        ]
    return records


def assert_disjoint(splits: Dict[str, List[Record]]) -> None:
    """Fail loudly if any patient / lesion appears in more than one split."""
    seen: Dict[str, str] = {}
    for split, recs in splits.items():
        for r in recs:
            other = seen.setdefault(r.group, split)
            if other != split:
                raise AssertionError(f"Group {r.group} appears in both {other} and {split}")


# ── Torch datasets ───────────────────────────────────────────────────────────

class MultiModalDataset(Dataset):
    """
    Yields image, tokenised metadata text, labels and modality index.

    Text goes through the API's TextPreprocessor, so training sees exactly
    the cleaning and tokenisation used at inference. In training, the text
    is replaced by NO_DETAILS with probability `blank_text_prob`, so the
    model works without clinical details and that placeholder does not
    become a marker for brain scans (which never have metadata).
    """

    def __init__(
        self,
        records: List[Record],
        text_preprocessor: TextPreprocessor,
        image_size: int,
        train: bool,
        blank_text_prob: float = 0.0,
    ):
        self.records = records
        self.text_preprocessor = text_preprocessor
        self.train = train
        self.blank_text_prob = blank_text_prob
        if train:
            self.transforms = {m: get_train_transforms(image_size, m) for m in MODALITY_INDEX}
        else:
            shared = get_val_transforms(image_size)
            self.transforms = {m: shared for m in MODALITY_INDEX}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("RGB")
        text = rec.text
        if self.train and random.random() < self.blank_text_prob:
            text = NO_DETAILS
        encoded = self.text_preprocessor.tokenize(text)
        return {
            "image": self.transforms[rec.modality](image),
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": torch.from_numpy(rec.labels),
            "modality": torch.tensor(MODALITY_INDEX[rec.modality]),
        }


class ModalityDataset(Dataset):
    """Image -> modality index, for the MobileNetV3 modality classifier."""

    def __init__(self, records: List[Record], image_size: int = 224, train: bool = False):
        self.records = records
        self.transform = get_train_transforms(image_size) if train else get_val_transforms(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("RGB")
        return self.transform(image), MODALITY_INDEX[rec.modality]
