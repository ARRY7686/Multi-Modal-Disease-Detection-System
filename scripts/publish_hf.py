r"""
Publish trained weights and the demo Space to the Hugging Face Hub.

    python scripts/publish_hf.py weights --repo USER/multimodal-disease-detection --models-dir models
    python scripts/publish_hf.py space   --repo USER/Multi-Modal-Disease-Detection --model-repo USER/multimodal-disease-detection

Needs a write token: `hf auth login` (or HF_TOKEN in the environment).
The Space is a Docker Space built from the repository's Dockerfile; it pulls
the weights from the model repo at build time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_FILES = ["fusion_model_best.pth", "modality_classifier.pth", "multimodal_config.json"]
SPACE_IGNORE = [
    ".git/*", ".github/*", "venv/*", ".venv/*", "**/__pycache__/*", ".pytest_cache/*", ".ruff_cache/*",
    "frontend/node_modules/*", "frontend/dist/*", "kaggle_notebooks/*", "models/*", "tests/*", "README.md",
]


def _results_line(models_dir: Path) -> str:
    meta = json.loads((models_dir / "multimodal_config.json").read_text(encoding="utf-8"))
    r = meta.get("training_results", {}).get("multimodal", {})
    chest, skin, brain = r.get("chest", {}), r.get("skin", {}), r.get("brain", {})

    def f(x):
        return "–" if x is None else f"{x:.3f}"

    return (
        f"| multimodal | {f(chest.get('macro_auroc'))} | {f(skin.get('balanced_accuracy'))} "
        f"| {f(skin.get('macro_auroc'))} | {f(brain.get('accuracy'))} | {f(r.get('ece'))} |"
    )


def model_card(models_dir: Path, github: str) -> str:
    return f"""---
license: cc-by-nc-4.0
library_name: pytorch
tags: [medical-imaging, multimodal, chest-x-ray, dermatology, brain-mri, grad-cam]
---

# Multi-Modal Disease Detection (research prototype)

EfficientNet-B4 image encoder + BioBERT text encoder with bidirectional cross-attention fusion,
scoring 25 findings across chest X-rays (NIH ChestX-ray14), dermoscopy (HAM10000) and brain MRI.
The paired text is real recorded metadata (age, sex, X-ray view, lesion site), never the label.

Held-out test results (patient-, lesion- and duplicate-disjoint splits, scored once):

| Mode | Chest macro AUROC | Skin balanced acc. | Skin macro AUROC | Brain accuracy | ECE |
|---|---|---|---|---|---|
{_results_line(models_dir)}

Full per-mode results, confidence intervals and the training history are in `multimodal_config.json`.
Code, training pipeline and evaluation: {github}

**Not a medical device.** Trained on public research datasets and not clinically validated.
Licensed CC BY-NC 4.0 because HAM10000 is; check each dataset's own terms before reuse.
"""


def space_readme(model_repo: str, github: str) -> str:
    return f"""---
title: Multi-Modal Disease Detection
emoji: 🩻
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
short_description: Chest X-ray, skin and brain MRI findings with Grad-CAM
models: [{model_repo}]
---

Research prototype: image + patient details, 25 findings, Grad-CAM and text attention.
Weights: [{model_repo}](https://huggingface.co/{model_repo}). Source: {github}

Not a medical device. Do not use for diagnosis.
"""


def publish_weights(api: HfApi, repo: str, models_dir: Path, github: str) -> None:
    missing = [f for f in WEIGHT_FILES if not (models_dir / f).exists()]
    if missing or not (models_dir / "tokenizer").is_dir():
        raise SystemExit(f"Missing in {models_dir}: {missing or ['tokenizer/']}")
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo, folder_path=str(models_dir), repo_type="model",
        allow_patterns=WEIGHT_FILES + ["tokenizer/*"], commit_message="Upload trained weights",
    )
    api.upload_file(
        repo_id=repo, path_or_fileobj=model_card(models_dir, github).encode("utf-8"),
        path_in_repo="README.md", repo_type="model", commit_message="Model card",
    )
    print(f"Weights published: https://huggingface.co/{repo}")


def publish_space(api: HfApi, repo: str, model_repo: str, github: str) -> None:
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=repo, folder_path=str(PROJECT_ROOT), repo_type="space",
        ignore_patterns=SPACE_IGNORE, commit_message="Deploy from repository",
    )
    api.upload_file(
        repo_id=repo, path_or_fileobj=space_readme(model_repo, github).encode("utf-8"),
        path_in_repo="README.md", repo_type="space", commit_message="Space card",
    )
    print(f"Space deployed: https://huggingface.co/spaces/{repo}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=["weights", "space"])
    parser.add_argument("--repo", required=True, help="Hub repo id to create or update")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--model-repo", help="Weights repo the Space builds from (space only)")
    parser.add_argument(
        "--github", default="https://github.com/ARRY7686/Multi-Modal-Disease-Detection-System",
    )
    args = parser.parse_args()
    api = HfApi()
    if args.target == "weights":
        publish_weights(api, args.repo, args.models_dir, args.github)
    else:
        if not args.model_repo:
            parser.error("space needs --model-repo")
        publish_space(api, args.repo, args.model_repo, args.github)


if __name__ == "__main__":
    main()
