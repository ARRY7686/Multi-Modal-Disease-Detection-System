"""
End-to-end check of the training pipeline on miniature fake datasets laid out
like the real Kaggle ones, followed by loading the outputs into the API.
"""

import json
import shutil

import numpy as np
import pytest
import yaml
from PIL import Image

from api.config import Config
from api.service import InferenceService
from training import data as D
from training.train import main as train_main

HAM_DX = list(D.HAM_DX)
CHEST_FINDINGS = ["No Finding", "Effusion", "Cardiomegaly|Edema", "Nodule", "Pneumonia|Effusion"]
BRAIN_DIRS = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
# Lesion site depends on the class: a real, label-free signal the text head can learn
SITES = ["back", "face", "scalp", "hand", "foot", "chest", "abdomen"]


def _noise_image(path, rng, mode="RGB", size=48):
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = 3 if mode == "RGB" else 1
    arr = rng.integers(0, 256, (size, size, channels), dtype=np.uint8).squeeze()
    Image.fromarray(arr).convert(mode).save(path)


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("kaggle_input")
    rng = np.random.default_rng(0)

    # NIH ChestX-ray14: 16 patients x 2 images; patients 0-3 form the official test list
    nih = root / "nih-chest-xrays"
    rows, test_names, trainval_names = [], [], []
    for patient in range(16):
        for k in range(2):
            name = f"{patient:08d}_{k:03d}.png"
            _noise_image(nih / "images_001" / "images" / name, rng, mode="L")
            rows.append([name, CHEST_FINDINGS[(patient + k) % len(CHEST_FINDINGS)], k, patient,
                         30 + patient, "M" if patient % 2 else "F", "PA" if k == 0 else "AP"])
            (test_names if patient < 4 else trainval_names).append(name)
    header = "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position\n"
    (nih / "Data_Entry_2017.csv").write_text(header + "".join(",".join(map(str, r)) + "\n" for r in rows))
    (nih / "test_list.txt").write_text("\n".join(test_names))
    (nih / "train_val_list.txt").write_text("\n".join(trainval_names))

    # HAM10000: 7 classes x 5 lesions x 2 photos, site determined by class
    ham = root / "skin-cancer-mnist-ham10000"
    lines = ["lesion_id,image_id,dx,dx_type,age,sex,localization"]
    n = 0
    for c, dx in enumerate(HAM_DX):
        for lesion in range(5):
            for _ in range(2):
                image_id = f"ISIC_{n:07d}"
                _noise_image(ham / "HAM10000_images_part_1" / f"{image_id}.jpg", rng)
                lines.append(f"HAM_{c}{lesion},{image_id},{dx},histo,{40 + lesion}.0,male,{SITES[c]}")
                n += 1
    (ham / "HAM10000_metadata.csv").write_text("\n".join(lines) + "\n")

    # Brain MRI: Training 6 per class, Testing 3 per class + one copy of a training image
    brain = root / "brain-tumor-classification-mri"
    for cls in BRAIN_DIRS:
        for i in range(6):
            _noise_image(brain / "Training" / cls / f"image({i}).jpg", rng, mode="L")
        for i in range(3):
            _noise_image(brain / "Testing" / cls / f"image({i}).jpg", rng, mode="L")
    shutil.copy(brain / "Training" / "glioma_tumor" / "image(0).jpg",
                brain / "Testing" / "glioma_tumor" / "dup.jpg")
    return root


def _splits(cfg, root):
    return {
        "nih": D.load_nih(cfg, root, val_fraction=0.25),
        "ham": D.load_ham10000(cfg, root, val_fraction=0.15),
        "brain": D.load_brain_mri(cfg, root, val_fraction=0.2),
    }


def test_splits_are_group_disjoint_and_deduplicated(tiny_cfg, fake_data):
    parts = _splits(tiny_cfg, fake_data)
    merged = {s: [r for p in parts.values() for r in p[s]] for s in D.SPLITS}
    D.assert_disjoint(merged)

    # NIH official test list is honoured and normal X-rays are kept
    assert {r.group for r in parts["nih"]["test"]} == {f"nih-{p}" for p in range(4)}
    assert any(r.labels.sum() == 0 for r in parts["nih"]["train"])
    # the copied training image is removed from the brain test split
    assert not any(r.image_path.endswith("dup.jpg") for r in parts["brain"]["test"])
    assert len(parts["brain"]["test"]) == 12


def test_paired_text_never_contains_the_diagnosis(tiny_cfg, fake_data):
    label_words = {w.lower() for name in tiny_cfg.all_labels for w in name.split("_")}
    label_words -= {"no", "tumor", "lesion"}  # generic words that may appear in metadata text
    for part in _splits(tiny_cfg, fake_data).values():
        for recs in part.values():
            for r in recs:
                assert not label_words & set(r.text.lower().replace(".", " ").split()), r.text


def test_metadata_text_rendering():
    assert D.chest_text("058Y", "M", "AP") == "58-year-old male. AP chest radiograph."
    assert D.chest_text(412, "F", "PA") == "Female. PA chest radiograph."  # impossible age dropped
    assert D.skin_text(45.0, "female", "upper extremity") == "45-year-old female. Lesion on the upper extremity."
    assert D.skin_text(float("nan"), "unknown", "unknown") == D.NO_DETAILS


def test_train_end_to_end_and_serve(tiny_cfg, fake_data, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(tiny_cfg.raw))
    out = tmp_path / "models"

    metadata = train_main([
        "--config", str(config_path), "--data-root", str(fake_data), "--out", str(out),
        "--epochs", "6", "--frozen-epochs", "1", "--batch-size", "8", "--num-workers", "0",
        "--lr-head", "3e-3", "--lr-text", "1e-3", "--blank-text-prob", "0",
        "--no-pretrained", "--bootstrap", "10", "--val-fraction", "0.25",
        "--chest-train-fraction", "1.0", "--modality-max-steps", "2", "--device", "cpu",
    ])

    for name in ["fusion_model_best.pth", "modality_classifier.pth", "multimodal_config.json",
                 "results.md", "tokenizer"]:
        assert (out / name).exists(), name
    saved = json.loads((out / "multimodal_config.json").read_text())
    assert set(saved["thresholds"]) == {"multimodal", "image_only", "text_only"}
    assert all(len(v) == 25 for v in saved["thresholds"].values())
    assert all(t > 0 for t in saved["temperatures"].values())
    assert {"chest", "skin", "brain"} <= set(saved["training_results"]["multimodal"])
    assert len(metadata["training_history"]) == 6
    # Training learns: loss falls, and the paired text (lesion site) separates skin classes on
    # held-out lesions. Image-only scores stay near chance here: a randomly initialised
    # EfficientNet in eval mode maps every input to almost the same features; real runs
    # start from ImageNet weights.
    losses = [h["train_loss"] for h in metadata["training_history"]]
    assert losses[-1] < 0.85 * losses[0]
    skin_auc = max(saved["training_results"][m]["skin"]["macro_auroc"] for m in ("multimodal", "text_only"))
    assert skin_auc > 0.7

    # The API loads everything training wrote
    serve_raw = dict(tiny_cfg.raw)
    serve_raw["inference"] = {**tiny_cfg.raw["inference"], "model_dir": str(out)}
    svc = InferenceService.from_disk(Config.from_dict(serve_raw))
    try:
        assert svc.weights_loaded == {"fusion_model": True, "modality_classifier": True}
        response = svc.predict(None, "45-year-old female. Lesion on the back.", modality="skin")
        assert response.calibrated
        assert all(p.threshold == pytest.approx(saved["thresholds"]["text_only"][p.label_index], abs=1e-4)
                   for p in response.predictions)
    finally:
        svc.close()
