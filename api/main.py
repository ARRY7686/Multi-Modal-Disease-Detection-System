"""
Multi-Modal Disease Detection — FastAPI Application

Loads trained models from disk and exposes prediction endpoints.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.config import Config, PROJECT_ROOT
from api.models import (
    CrossAttentionFusion,
    ImageEncoder,
    ModalityClassifier,
    MultiModalDetector,
    TextEncoder,
)
from api.explainability.gradcam import GradCAM
from api.explainability.text_attention import (
    extract_token_importance,
    merge_subword_importance,
    format_explanation,
)
from api.preprocessing.image_transforms import (
    preprocess_uploaded_image,
    preprocess_for_modality_classifier,
    inverse_normalize,
)
from api.preprocessing.text_processing import TextPreprocessor
from api.schemas import (
    DiseaseScore,
    Explanation,
    HealthResponse,
    LabelResponse,
    PredictionResponse,
    TokenImportance,
)

logger = logging.getLogger(__name__)

# ── Global state ─────────────────────────────────────────────────────────────

cfg: Optional[Config] = None
model: Optional[MultiModalDetector] = None
modality_clf_loaded: bool = False
modality_clf: Optional[ModalityClassifier] = None
text_preprocessor: Optional[TextPreprocessor] = None
grad_cam: Optional[GradCAM] = None
device: torch.device = torch.device("cpu")


def _load_state_dict(path: Path, map_location: str = "cpu") -> Dict:
    if not path.exists():
        logger.warning("Model file not found: %s", path)
        return {}
    return torch.load(path, map_location=map_location, weights_only=False)


def _load_models(cfg: Config):
    global model, modality_clf, text_preprocessor, grad_cam, device

    device = torch.device(cfg.device)
    model_dir = cfg.model_dir
    logger.info("Loading models from %s onto %s", model_dir, device)

    # ── MultiModal Detector ──────────────────────────────────────────
    model = MultiModalDetector.from_config(cfg)

    # Load the full model state dict (saved by multimodal training notebook)
    best_weights = _load_state_dict(model_dir / "fusion_model_best.pth")
    if best_weights:
        model.load_state_dict(best_weights, strict=False)
        logger.info("Full model loaded from fusion_model_best.pth")
    else:
        # Fallback: load individual component weights
        full_weights = _load_state_dict(model_dir / "fusion_model.pth")
        if full_weights:
            model.load_state_dict(full_weights, strict=False)
            logger.info("Full model loaded from fusion_model.pth")
        else:
            logger.warning("No fusion model checkpoint found; using random init.")

    model.to(device)
    model.eval()

    # ── Modality Classifier ──────────────────────────────────────────
    global modality_clf_loaded
    modality_clf = ModalityClassifier(num_classes=3, pretrained=False)
    mc_weights = _load_state_dict(model_dir / "modality_classifier.pth")
    if mc_weights:
        modality_clf.load_state_dict(mc_weights, strict=False)
        modality_clf_loaded = True
        logger.info("Modality classifier loaded.")
    else:
        logger.warning("Modality classifier weights not found — using heuristic fallback.")
    modality_clf.to(device)
    modality_clf.eval()

    # ── Text Preprocessor ────────────────────────────────────────────
    tokenizer_path = model_dir / "tokenizer"
    if not tokenizer_path.exists():
        tokenizer_path = None  # Falls back to HF hub
    text_preprocessor = TextPreprocessor(
        model_name=cfg.text_model_name,
        max_length=cfg.text_max_length,
        tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
    )
    logger.info("Text preprocessor ready.")

    # ── Grad-CAM ─────────────────────────────────────────────────────
    # features is nn.Sequential(backbone.features), so [0][-1] = last conv block
    target_layer = model.image_encoder.features[0][-1]
    grad_cam = GradCAM(
        model=model.image_encoder,
        target_layer=target_layer,
    )
    logger.info("Grad-CAM initialised.")


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cfg
    cfg = Config.get()
    _load_models(cfg)
    logger.info("All models loaded. API ready.")
    yield
    # Cleanup
    if grad_cam is not None:
        grad_cam.remove_hooks()
    logger.info("Shutdown complete.")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Modal Disease Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _detect_modality_heuristic(image_bytes: bytes) -> str:
    """Fallback modality detection using image color statistics."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32)

    # Mean saturation: skin lesions are colorful, X-rays and MRIs are grayscale
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    max_c = np.maximum(np.maximum(r, g, ), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c
    # Saturation = diff / max_c where max_c > 0
    mask = max_c > 0
    saturation = np.zeros_like(max_c)
    saturation[mask] = diff[mask] / max_c[mask]
    mean_sat = saturation.mean()

    if mean_sat > 0.15:
        return "skin"

    # Grayscale image → distinguish chest vs brain
    # Brain MRIs typically have dark (black) borders; X-rays are brighter overall
    mean_brightness = arr.mean() / 255.0
    # Check border darkness (brain MRIs have black background)
    h, w = arr.shape[:2]
    border = np.concatenate([
        arr[:h // 10, :].reshape(-1),     # top
        arr[-h // 10:, :].reshape(-1),    # bottom
        arr[:, :w // 10].reshape(-1),     # left
        arr[:, -w // 10:].reshape(-1),    # right
    ])
    border_brightness = border.mean() / 255.0

    # Brain MRI: dark borders, moderate center brightness
    if border_brightness < 0.12 and mean_brightness < 0.45:
        return "brain"

    return "chest"


def _detect_modality(image_bytes: bytes) -> str:
    if modality_clf_loaded:
        tensor = preprocess_for_modality_classifier(image_bytes)
        tensor = tensor.to(device)
        result = modality_clf.predict_modality(tensor)
        return result["modality"]
    return _detect_modality_heuristic(image_bytes)


def _filter_predictions(
    probs: np.ndarray,
    modality: str,
    top_k: int = 5,
    threshold: float = 0.5,
) -> list[DiseaseScore]:
    labels = cfg.all_labels
    start, end = cfg.modality_index_range(modality)

    # Filter to the relevant modality labels
    scores = []
    for i in range(start, end):
        scores.append(DiseaseScore(
            disease=labels[i],
            confidence=round(float(probs[i]), 4),
            label_index=i,
        ))
    scores.sort(key=lambda s: s.confidence, reverse=True)
    return scores[:top_k]


def _generate_image_explanation(
    image_bytes: bytes,
    input_tensor: torch.Tensor,
    top_class_idx: int,
) -> Optional[str]:
    try:
        original = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        heatmap_b64 = grad_cam.generate_base64(
            input_tensor, original, top_class_idx
        )
        return heatmap_b64
    except Exception as e:
        logger.warning("Grad-CAM failed: %s", e)
        return None


def _generate_text_explanation(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    original_text: str,
) -> tuple[Optional[list[TokenImportance]], Optional[str]]:
    try:
        attn = model.text_encoder.last_attention_weights
        if attn is None:
            return None, None
        tokens = text_preprocessor.tokenizer.convert_ids_to_tokens(
            input_ids[0].tolist()
        )
        raw_importance = extract_token_importance(
            attn, tokens, attention_mask, top_k=cfg.text_top_k_tokens
        )
        explanation = format_explanation(raw_importance, original_text, top_k=cfg.text_top_k_tokens)
        token_list = [
            TokenImportance(token=t["token"], importance=t["importance"])
            for t in explanation["important_tokens"]
        ]
        return token_list, explanation.get("highlighted_text")
    except Exception as e:
        logger.warning("Text explanation failed: %s", e)
        return None, None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        models_loaded={
            "image_encoder": model is not None,
            "text_encoder": model is not None,
            "fusion": model is not None,
            "modality_classifier": modality_clf is not None,
        },
    )


@app.get("/labels", response_model=LabelResponse)
async def labels():
    return LabelResponse(
        labels=cfg.all_labels,
        modality_groups={
            "chest": cfg.chest_labels,
            "skin": cfg.skin_labels,
            "brain": cfg.brain_labels,
        },
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: Optional[UploadFile] = File(None),
    symptoms: Optional[str] = Form(None),
    modality: Optional[str] = Form(None),
):
    if image is None and symptoms is None:
        raise HTTPException(400, "Provide at least an image or symptoms text.")

    image_bytes: Optional[bytes] = None
    if image is not None:
        image_bytes = await image.read()
        if len(image_bytes) > cfg.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(413, "Image file too large.")

    # Determine modality
    detected_modality = modality or "chest"
    if image_bytes and (modality is None or modality == "auto"):
        detected_modality = _detect_modality(image_bytes)

    # Determine mode
    has_image = image_bytes is not None
    has_text = symptoms is not None and symptoms.strip() != ""
    if has_image and has_text:
        mode = "multimodal"
    elif has_image:
        mode = "image_only"
    elif has_text:
        mode = "text_only"
    else:
        raise HTTPException(400, "No valid input provided.")

    # Prepare tensors
    img_tensor = None
    input_ids = None
    attention_mask = None

    if has_image:
        img_tensor = preprocess_uploaded_image(image_bytes, cfg.image_size).to(device)

    if has_text:
        encoded = text_preprocessor.tokenize(symptoms)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

    # Inference
    with torch.no_grad():
        outputs = model(
            image=img_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode=mode,
        )

    logits = outputs["logits"]
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Predictions
    predictions = _filter_predictions(
        probs, detected_modality, top_k=cfg.top_k_predictions
    )

    # Explainability
    heatmap_b64 = None
    important_tokens = None
    highlighted_text = None

    if has_image and predictions:
        top_idx = predictions[0].label_index
        heatmap_b64 = _generate_image_explanation(image_bytes, img_tensor, top_idx)

    if has_text and input_ids is not None:
        # Re-run forward with grad to capture attention (already stored during inference)
        important_tokens, highlighted_text = _generate_text_explanation(
            input_ids, attention_mask, symptoms
        )

    explanation = Explanation(
        heatmap_base64=heatmap_b64,
        important_tokens=important_tokens,
        highlighted_text=highlighted_text,
    )

    return PredictionResponse(
        predictions=predictions,
        modality=detected_modality,
        mode=mode,
        explanation=explanation,
    )


@app.post("/predict-multimodal", response_model=PredictionResponse)
async def predict_multimodal(
    image: UploadFile = File(...),
    symptoms: str = Form(...),
    modality: Optional[str] = Form(None),
):
    if not symptoms.strip():
        raise HTTPException(400, "Symptoms text is required for multimodal prediction.")

    image_bytes = await image.read()
    if len(image_bytes) > cfg.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(413, "Image file too large.")

    # Modality
    detected_modality = modality or "auto"
    if detected_modality == "auto":
        detected_modality = _detect_modality(image_bytes)

    # Prepare tensors
    img_tensor = preprocess_uploaded_image(image_bytes, cfg.image_size).to(device)
    encoded = text_preprocessor.tokenize(symptoms)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Inference
    with torch.no_grad():
        outputs = model(
            image=img_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode="multimodal",
        )

    logits = outputs["logits"]
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    predictions = _filter_predictions(probs, detected_modality, top_k=cfg.top_k_predictions)

    # Explainability
    heatmap_b64 = None
    if predictions:
        top_idx = predictions[0].label_index
        heatmap_b64 = _generate_image_explanation(image_bytes, img_tensor, top_idx)

    important_tokens, highlighted_text = _generate_text_explanation(
        input_ids, attention_mask, symptoms
    )

    return PredictionResponse(
        predictions=predictions,
        modality=detected_modality,
        mode="multimodal",
        explanation=Explanation(
            heatmap_base64=heatmap_b64,
            important_tokens=important_tokens,
            highlighted_text=highlighted_text,
        ),
    )


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    _cfg = Config.get()
    uvicorn.run(
        "api.main:app",
        host=_cfg.api_host,
        port=_cfg.api_port,
        reload=False,
    )
