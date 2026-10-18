r"""
Download trained model weights from Kaggle to the local models/ directory.

Usage:
    python scripts/download_models.py --kaggle-dataset YOUR_USERNAME/multimodal-disease-weights

Requires the Kaggle API key to be configured:
    pip install kaggle
    Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

EXPECTED_FILES = [
    "fusion_model_best.pth",
    "modality_classifier.pth",
    "multimodal_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
]


def download_from_kaggle(dataset_slug: str, dest: Path):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        return False

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {dataset_slug}")
    print(f"Destination: {dest}")

    dest.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset_slug, path=str(dest), unzip=True)
    print("Download complete.")
    return True


def verify_files(dest: Path):
    print("\nVerifying model files:")
    missing = []
    for f in EXPECTED_FILES:
        fpath = dest / f
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  [OK] {f} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {f}")
            missing.append(f)
    if missing:
        print(f"\nWarning: {len(missing)} file(s) missing. The system may still work in degraded mode.")
    else:
        print("\nAll model files present.")


def main():
    parser = argparse.ArgumentParser(description="Download trained model weights from Kaggle.")
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        help="Kaggle dataset slug (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        help="Local directory containing model files (skip download)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(MODELS_DIR),
        help=f"Destination directory (default: {MODELS_DIR})",
    )
    args = parser.parse_args()
    dest = Path(args.dest)

    if args.local_dir:
        src = Path(args.local_dir)
        if not src.exists():
            print(f"ERROR: Local directory not found: {src}")
            return
        print(f"Copying from {src} to {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            dst_path = dest / item.name
            if item.is_dir():
                shutil.copytree(item, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst_path)
    elif args.kaggle_dataset:
        if not download_from_kaggle(args.kaggle_dataset, dest):
            return
    else:
        print("Provide --kaggle-dataset or --local-dir. Run with --help for details.")
        return

    verify_files(dest)


if __name__ == "__main__":
    main()
