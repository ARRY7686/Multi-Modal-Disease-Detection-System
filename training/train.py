"""
Train the multimodal detector and write every artifact the API loads.

    python -m training.train --data-root /kaggle/input --out /kaggle/working/models

What differs from the original notebooks (see training/data.py for the data side):

- One model, three heads trained together: every step supervises the
  cross-attention fusion path *and* the image-only and text-only heads
  (forward_all), so all three API modes have trained heads.
- Loss masked to each sample's own modality label range, so the model is
  not rewarded for learning that a chest X-ray is not a melanoma.
- Square-root modality sampling: small datasets are oversampled less
  aggressively than with fully balanced sampling.
- Model selection on validation; temperatures and per-class thresholds
  fitted on validation; test touched once, at the end.

Outputs in --out:
  fusion_model_best.pth, modality_classifier.pth, tokenizer/,
  multimodal_config.json (thresholds, temperatures, test results, history),
  results.md (tables for the README)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from api.config import CONFIG_PATH, Config
from api.models import ModalityClassifier, MultiModalDetector
from api.preprocessing.text_processing import TextPreprocessor
from training import data as D
from training import evaluate as E

logger = logging.getLogger("training")

MODES = ("multimodal", "image_only", "text_only")


@dataclass
class TrainSettings:
    out: Path = Path("models")
    data_root: Optional[Path] = None      # searched for all three datasets
    chest_root: Optional[Path] = None     # per-dataset overrides
    skin_root: Optional[Path] = None
    brain_root: Optional[Path] = None

    chest_train_fraction: float = 0.3     # fraction of NIH train/val *patients*
    chest_test_fraction: float = 1.0      # fraction of NIH official-test patients
    val_fraction: float = 0.1
    skin_test_fraction: float = 0.15

    epochs: int = 8
    frozen_epochs: int = 1                # backbones frozen, heads + fusion warm up
    batch_size: int = 16
    num_workers: int = 4
    lr_head: float = 3e-4
    lr_image: float = 1e-4
    lr_text: float = 2e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    aux_weight: float = 0.5               # weight of image-only / text-only head losses
    blank_text_prob: float = 0.2
    sampling_power: float = 0.5           # modality share ∝ count ** power
    max_steps_per_epoch: Optional[int] = None

    modality_epochs: int = 1
    modality_max_steps: Optional[int] = 600
    bootstrap: int = 500
    pretrained: bool = True
    seed: int = 42
    device: str = "auto"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    defaults = TrainSettings()
    for f in fields(TrainSettings):
        default = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif f.name.endswith(("root", "out")):
            parser.add_argument(flag, type=Path, default=default)
        elif default is None:
            parser.add_argument(flag, type=int, default=None)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    return Config(config_path), TrainSettings(**args)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Data ─────────────────────────────────────────────────────────────────────

def load_splits(cfg: Config, s: TrainSettings) -> Dict[str, List[D.Record]]:
    roots = {
        "chest": s.chest_root or s.data_root,
        "skin": s.skin_root or s.data_root,
        "brain": s.brain_root or s.data_root,
    }
    if not all(roots.values()):
        raise ValueError("Pass --data-root or all of --chest-root/--skin-root/--brain-root")

    parts = [
        D.load_nih(cfg, roots["chest"], s.chest_train_fraction, s.chest_test_fraction, s.val_fraction, s.seed),
        D.load_ham10000(cfg, roots["skin"], s.skin_test_fraction, s.val_fraction, s.seed),
        D.load_brain_mri(cfg, roots["brain"], s.val_fraction, s.seed),
    ]
    splits = {split: [r for p in parts for r in p[split]] for split in D.SPLITS}
    D.assert_disjoint(splits)
    for split, recs in splits.items():
        logger.info("%-5s %s", split, dict(Counter(r.modality for r in recs)))
    return splits


def modality_sampler(records: List[D.Record], power: float, seed: int) -> WeightedRandomSampler:
    counts = Counter(r.modality for r in records)
    weights = [counts[r.modality] ** (power - 1.0) for r in records]
    return WeightedRandomSampler(
        weights, num_samples=len(records), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


# ── Model helpers ────────────────────────────────────────────────────────────

def backbones(model: MultiModalDetector) -> List[nn.Module]:
    return [model.image_encoder.features, model.text_encoder.bert]


def set_backbones_trainable(model: MultiModalDetector, trainable: bool) -> None:
    for module in backbones(model):
        for p in module.parameters():
            p.requires_grad = trainable
        if not trainable:
            module.eval()  # freeze BatchNorm statistics and dropout too


def build_optimizer(model: MultiModalDetector, s: TrainSettings) -> torch.optim.Optimizer:
    image_params = list(model.image_encoder.features.parameters())
    text_params = list(model.text_encoder.bert.parameters())
    backbone_ids = {id(p) for p in image_params + text_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": s.lr_head},
            {"params": image_params, "lr": s.lr_image},
            {"params": text_params, "lr": s.lr_text},
        ],
        weight_decay=s.weight_decay,
    )


def warmup_cosine(total_steps: int, warmup_frac: float):
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# ── Loops ────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, scaler, mask_table, s, device, frozen) -> float:
    model.train()
    if frozen:
        set_backbones_trainable(model, False)
    use_amp = device.type == "cuda"
    total, steps = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = mask_table[batch["modality"].to(device)]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model.forward_all(image, ids, attn)
        loss = masked_bce(out["multimodal"], labels, mask) + s.aux_weight * (
            masked_bce(out["image_only"], labels, mask) + masked_bce(out["text_only"], labels, mask)
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total += loss.item()
        steps += 1
        if s.max_steps_per_epoch and steps >= s.max_steps_per_epoch:
            break
    return total / max(1, steps)


@torch.no_grad()
def predict(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()
    use_amp = device.type == "cuda"
    out: Dict[str, list] = {m: [] for m in MODES}
    labels, modality = [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model.forward_all(
                batch["image"].to(device), batch["input_ids"].to(device), batch["attention_mask"].to(device),
            )
        for m in MODES:
            out[m].append(logits[m].float().cpu().numpy())
        labels.append(batch["labels"].numpy())
        modality.append(batch["modality"].numpy())
    result = {m: np.concatenate(v) for m, v in out.items()}
    result["labels"] = np.concatenate(labels)
    result["modality"] = np.concatenate(modality)
    return result


def train_modality_classifier(splits, s: TrainSettings, device) -> tuple[dict, float]:
    clf = ModalityClassifier(num_classes=3, pretrained=s.pretrained).to(device)
    loader = DataLoader(
        D.ModalityDataset(splits["train"], train=True), batch_size=32, num_workers=s.num_workers,
        sampler=modality_sampler(splits["train"], 0.0, s.seed),  # fully balanced: 3 easy classes
    )
    optimizer = torch.optim.AdamW(clf.parameters(), lr=3e-4)
    step = 0
    for _ in range(s.modality_epochs):
        clf.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(clf(x.to(device)), y.to(device)).backward()
            optimizer.step()
            step += 1
            if s.modality_max_steps and step >= s.modality_max_steps:
                break

    clf.eval()
    correct = total = 0
    test_loader = DataLoader(D.ModalityDataset(splits["test"]), batch_size=64, num_workers=s.num_workers)
    with torch.no_grad():
        for x, y in test_loader:
            correct += (clf(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return {k: v.cpu() for k, v in clf.state_dict().items()}, correct / max(1, total)


# ── Reporting ────────────────────────────────────────────────────────────────

def _fmt(value, ci=None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    text = f"{value:.3f}"
    if ci and not any(math.isnan(c) for c in ci):
        text += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return text


def results_markdown(cfg: Config, test: Dict[str, Dict], counts: Dict, modality_acc: float) -> str:
    lines = [
        "# Evaluation results",
        "",
        "Held-out test sets: NIH ChestX-ray14 official test list (patient-disjoint), HAM10000 "
        "lesion-grouped hold-out, Brain MRI official Testing folder (training duplicates removed). "
        "Temperatures and thresholds were fitted on validation only. 95% bootstrap CIs in brackets.",
        "",
        "| Mode | Chest macro AUROC | Skin balanced acc. | Skin macro AUROC | Brain accuracy | Brain macro AUROC | ECE raw → calibrated |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        r = test[mode]
        chest, skin, brain = r.get("chest", {}), r.get("skin", {}), r.get("brain", {})
        lines.append(
            f"| {mode} | {_fmt(chest.get('macro_auroc'), chest.get('macro_auroc_ci'))} "
            f"| {_fmt(skin.get('balanced_accuracy'), skin.get('balanced_accuracy_ci'))} "
            f"| {_fmt(skin.get('macro_auroc'))} "
            f"| {_fmt(brain.get('accuracy'))} | {_fmt(brain.get('macro_auroc'))} "
            f"| {_fmt(r['ece_raw'])} → {_fmt(r['ece'])} |"
        )
    lines += ["", "## Chest per-class AUROC", "", "| Finding | Multimodal | Image only |", "|---|---|---|"]
    mm = test["multimodal"].get("chest", {}).get("per_class_auroc", {})
    io_ = test["image_only"].get("chest", {}).get("per_class_auroc", {})
    for name in cfg.chest_labels:
        lines.append(f"| {name.replace('_', ' ')} | {_fmt(mm.get(name))} | {_fmt(io_.get(name))} |")
    lines += [
        "",
        f"Modality classifier test accuracy: {modality_acc:.4f}",
        "",
        "## Data",
        "",
        "| Split | Chest | Skin | Brain |",
        "|---|---|---|---|",
    ]
    for split in D.SPLITS:
        c = counts[split]
        lines.append(f"| {split} | {c.get('chest', 0)} | {c.get('skin', 0)} | {c.get('brain', 0)} |")
    return "\n".join(lines) + "\n"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> Dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, s = parse_args(argv)
    seed_everything(s.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if s.device == "auto" else s.device
    )
    s.out.mkdir(parents=True, exist_ok=True)
    logger.info("Settings: %s", s)

    splits = load_splits(cfg, s)
    counts = {split: dict(Counter(r.modality for r in recs)) for split, recs in splits.items()}

    text_preprocessor = TextPreprocessor(model_name=cfg.text_model_name, max_length=cfg.text_max_length)

    def loader(split: str, train: bool) -> DataLoader:
        dataset = D.MultiModalDataset(
            splits[split], text_preprocessor, cfg.image_size, train=train,
            blank_text_prob=s.blank_text_prob if train else 0.0,
        )
        return DataLoader(
            dataset,
            batch_size=s.batch_size,
            sampler=modality_sampler(splits[split], s.sampling_power, s.seed) if train else None,
            num_workers=s.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=train,
        )

    train_loader, val_loader, test_loader = loader("train", True), loader("val", False), loader("test", False)

    model = MultiModalDetector.from_config(cfg, pretrained=s.pretrained).to(device)
    optimizer = build_optimizer(model, s)
    steps_per_epoch = min(len(train_loader), s.max_steps_per_epoch or len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(steps_per_epoch * s.epochs, s.warmup_frac),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    mask_table = torch.tensor(
        E.modality_mask(cfg, np.arange(len(E.MODALITIES))), device=device,
    )

    best_auc, history = -math.inf, []
    best_path = s.out / "fusion_model_best.pth"
    best_path.unlink(missing_ok=True)
    for epoch in range(s.epochs):
        frozen = epoch < s.frozen_epochs
        set_backbones_trainable(model, not frozen)
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, mask_table, s, device, frozen)
        val = predict(model, val_loader, device)
        val_auc = {m: E.evaluate_mode(cfg, val[m], val["labels"], val["modality"])["auc"] for m in MODES}
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "frozen": frozen, "val_auc": val_auc,
                        "seconds": round(time.time() - start, 1)})
        logger.info("epoch %d/%d loss %.4f val AUC %s (%.0fs)", epoch + 1, s.epochs, train_loss,
                    {m: round(v, 4) for m, v in val_auc.items()}, time.time() - start)
        score = val_auc["multimodal"]
        improved = not math.isnan(score) and score > best_auc
        if improved or not best_path.exists():
            best_auc = max(best_auc, score) if improved else best_auc
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    # Calibration and thresholds: validation only
    val = predict(model, val_loader, device)
    val_mask = E.modality_mask(cfg, val["modality"])
    temperatures, thresholds = {}, {}
    for m in MODES:
        temperatures[m] = E.fit_temperature(val[m], val["labels"], val_mask)
        val_probs = E.sigmoid(val[m] / temperatures[m])
        thresholds[m] = E.fit_thresholds(cfg, val_probs, val["labels"], val["modality"]).tolist()

    # Test: touched once
    test = predict(model, test_loader, device)
    test_results = {
        m: E.evaluate_mode(cfg, test[m], test["labels"], test["modality"], temperatures[m], s.bootstrap)
        for m in MODES
    }

    clf_state, modality_acc = train_modality_classifier(splits, s, device)
    torch.save(clf_state, s.out / "modality_classifier.pth")
    text_preprocessor.save(str(s.out / "tokenizer"))

    metadata = {
        "image_backbone": cfg.image_backbone,
        "image_feature_dim": cfg.image_feature_dim,
        "image_size": cfg.image_size,
        "text_model_name": cfg.text_model_name,
        "text_hidden_dim": cfg.text_hidden_dim,
        "projection_dim": cfg.projection_dim,
        "num_classes": cfg.num_classes,
        "fusion_type": cfg.fusion_type,
        "fusion_num_heads": cfg.fusion_num_heads,
        "fusion_num_cross_layers": cfg.fusion_num_cross_layers,
        "all_labels": cfg.all_labels,
        "modality_index_ranges": {m: list(cfg.modality_index_range(m)) for m in E.MODALITIES},
        "thresholds": thresholds,
        "temperatures": temperatures,
        "training_results": test_results,
        "modality_classifier_test_accuracy": modality_acc,
        "evaluation_protocol": {
            "chest": "NIH official test_list.txt (patient-disjoint); validation split by Patient ID",
            "skin": "HAM10000 split by lesion_id (StratifiedGroupKFold); dx_type not used",
            "brain": "official Testing folder, exact dHash duplicates of Training removed",
            "text": "real metadata only (age, sex, view position / lesion site), never the label",
        },
        "data_counts": counts,
        "training_history": history,
        "settings": asdict(s),
    }
    with open(s.out / "multimodal_config.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(metadata), f, indent=2)
    report = results_markdown(cfg, test_results, counts, modality_acc)
    (s.out / "results.md").write_text(report, encoding="utf-8")
    logger.info("\n%s", report)
    return metadata


if __name__ == "__main__":
    main()
