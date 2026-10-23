r"""
Precompute the example gallery: real model outputs for a few held-out test cases.

    python -m scripts.build_gallery --images path/to/originals

`scripts/gallery_cases.json` lists the cases: one per class, drawn at random
(seed 0) from each dataset's held-out test split and not filtered by what the
model predicts. `--images` holds each case's original file named by its id
(e.g. `skin-mel.jpg`, `chest-effusion.png`).

For each case this writes a web-sized image and the three /predict responses
the frontend asks for (multimodal, image only, details only), produced by the
real app in-process from that same web-sized file, so a live run on the
example reproduces the cached one. Also writes /health and /model-info.
The frontend uses the gallery as one-click examples, and the static build
(`npm run build:static`) serves it with no backend at all.

Rerun after replacing the weights in models/.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from training.data import NO_DETAILS, chest_text, skin_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_SIDE = 1024
THUMB_SIDE = 160


def details_text(case: dict) -> str:
    """Same sentence the frontend builds from the structured fields."""
    d = case["details"]
    if case["modality"] == "chest":
        return chest_text(d.get("age"), d.get("sex"), d.get("view", ""))
    if case["modality"] == "skin":
        return skin_text(d.get("age"), d.get("sex"), d.get("site", ""))
    return NO_DETAILS


def web_image(src: Path, side: int) -> bytes:
    img = Image.open(src).convert("RGB")
    img.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90, optimize=True)
    return buf.getvalue()


def find_original(images: Path, case_id: str) -> Path:
    hits = [p for p in images.iterdir() if p.stem == case_id]
    if len(hits) != 1:
        raise SystemExit(f"Expected one image named {case_id}.* in {images}, found {len(hits)}")
    return hits[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="Folder of originals named <case id>.<ext>")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "scripts" / "gallery_cases.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "frontend" / "public" / "gallery")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    with TestClient(create_app()) as client:
        health = client.get("/health").json()
        if health["status"] != "ok":
            raise SystemExit(f"Trained weights are not loaded: {health}")
        index = []
        for case in cases:
            image = web_image(find_original(args.images, case["id"]), MAX_SIDE)
            (args.out / f"{case['id']}.jpg").write_bytes(image)
            (args.out / f"{case['id']}-thumb.jpg").write_bytes(
                web_image(find_original(args.images, case["id"]), THUMB_SIDE)
            )
            text = details_text(case)

            def predict(with_image: bool, with_text: bool, _image=image, _text=text, _case=case):
                files = {"image": (f"{_case['id']}.jpg", _image, "image/jpeg")} if with_image else None
                data = {"modality": _case["modality"], **({"symptoms": _text} if with_text else {})}
                r = client.post("/predict", files=files, data=data)
                r.raise_for_status()
                return r.json()

            results = {
                "multimodal": predict(True, True),
                "image_only": predict(True, False),
                "text_only": predict(False, True),
            }
            (args.out / f"{case['id']}.json").write_text(
                json.dumps({"text": text, "results": results}), encoding="utf-8",
            )
            index.append({**case, "text": text, "image": f"{case['id']}.jpg", "thumb": f"{case['id']}-thumb.jpg"})
            top = results["multimodal"]["predictions"][0]
            print(f"{case['id']:<24} label={'+'.join(case['label']) or 'none':<22} top={top['disease']} {top['confidence']:.3f}")

        (args.out / "index.json").write_text(json.dumps({"cases": index}, indent=1) + "\n", encoding="utf-8")
        (args.out / "health.json").write_text(json.dumps(health) + "\n", encoding="utf-8")
        (args.out / "model-info.json").write_text(json.dumps(client.get("/model-info").json()) + "\n", encoding="utf-8")
    print(f"Gallery written to {args.out}")


if __name__ == "__main__":
    main()
