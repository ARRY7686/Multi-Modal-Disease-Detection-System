"""
Inference service — owns the loaded models and turns raw request inputs
(image bytes, symptom text, modality hint) into a PredictionResponse.

Kept separate from the FastAPI app so it can be built from tiny models in
tests and reused outside HTTP (scripts, notebooks).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from api.config import Config
from api.explainability.gradcam import GradCAM
from api.explainability.text_attention import extract_token_importance, format_explanation
from api.models import ModalityClassifier, MultiModalDetector
from api.preprocessing.image_transforms import (
    decode_image,
    preprocess_for_modality_classifier,
    preprocess_image,
)
from api.preprocessing.text_processing import TextPreprocessor
from api.schemas import DiseaseScore, Explanation, PredictionResponse, TokenImportance

logger = logging.getLogger(__name__)

MODALITIES = ("chest", "skin", "brain")
MODES = ("multimodal", "image_only", "text_only")


class InvalidInput(ValueError):
    """A request the service cannot answer; carries the HTTP status to return."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# ── Weight loading ───────────────────────────────────────────────────────────

def load_state_dict(path: Path) -> Optional[Dict[str, torch.Tensor]]:
    if not path.exists():
        logger.warning("Model file not found: %s", path)
        return None
    # weights_only=True: a state dict is plain tensors, and this refuses to
    # unpickle arbitrary objects from a downloaded checkpoint.
    return torch.load(path, map_location="cpu", weights_only=True)


def load_weights(module: torch.nn.Module, state: Optional[Dict], name: str) -> bool:
    """Load `state` into `module`; True only if every parameter was covered."""
    if not state:
        return False
    result = module.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        logger.warning(
            "%s: ignored %d unexpected checkpoint keys (e.g. %s)",
            name, len(result.unexpected_keys), result.unexpected_keys[:3],
        )
    if result.missing_keys:
        logger.error(
            "%s: %d parameters missing from checkpoint (e.g. %s); "
            "treating weights as NOT loaded",
            name, len(result.missing_keys), result.missing_keys[:3],
        )
        return False
    logger.info("%s weights loaded.", name)
    return True


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_thresholds(metadata: Dict, num_classes: int, default: float) -> Dict[str, np.ndarray]:
    """
    Per-mode, per-class decision thresholds.

    `metadata["thresholds"]` may be a list (same thresholds for every mode)
    or a dict {mode: list}. Anything missing or malformed falls back to
    `default` for every class.
    """
    out = {mode: np.full(num_classes, default, dtype=np.float32) for mode in MODES}
    raw = metadata.get("thresholds")
    per_mode = {mode: raw for mode in MODES} if isinstance(raw, list) else (raw or {})
    for mode, values in per_mode.items():
        if mode not in out:
            continue
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape != (num_classes,):
            logger.warning("Ignoring thresholds for %s: expected %d values", mode, num_classes)
            continue
        out[mode] = arr
    return out


def detect_modality_heuristic(image: Image.Image) -> str:
    """Fallback modality detection from colour statistics."""
    arr = np.asarray(image, dtype=np.float32)

    # Mean saturation: skin lesions are colourful, X-rays and MRIs are grayscale
    max_c = arr.max(axis=-1)
    min_c = arr.min(axis=-1)
    saturation = np.where(max_c > 0, (max_c - min_c) / np.maximum(max_c, 1e-6), 0.0)
    if saturation.mean() > 0.15:
        return "skin"

    # Brain MRIs have a black background; X-rays are brighter overall
    h, w = arr.shape[:2]
    border = np.concatenate([
        arr[: h // 10].reshape(-1),
        arr[-(h // 10):].reshape(-1),
        arr[:, : w // 10].reshape(-1),
        arr[:, -(w // 10):].reshape(-1),
    ])
    if border.mean() / 255.0 < 0.12 and arr.mean() / 255.0 < 0.45:
        return "brain"
    return "chest"


# ── Service ──────────────────────────────────────────────────────────────────

class InferenceService:
    def __init__(
        self,
        cfg: Config,
        model: MultiModalDetector,
        modality_clf: Optional[ModalityClassifier],
        text_preprocessor: TextPreprocessor,
        *,
        weights_loaded: Dict[str, bool],
        metadata: Optional[Dict] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            modality_clf:   trained modality classifier, or None to use the
                            colour heuristic
            weights_loaded: {"fusion_model": bool, "modality_classifier": bool}
            metadata:       contents of models/multimodal_config.json
                            (per-class thresholds, validation results)
        """
        self.cfg = cfg
        self.model = model.to(device).eval()
        self.modality_clf = modality_clf.to(device).eval() if modality_clf is not None else None
        self.text_preprocessor = text_preprocessor
        self.weights_loaded = dict(weights_loaded)
        self.device = device

        metadata = metadata or {}
        self.thresholds = parse_thresholds(metadata, cfg.num_classes, cfg.confidence_threshold)
        self.mode_val_auc: Dict[str, float] = {
            mode: float(res["auc"])
            for mode, res in metadata.get("training_results", {}).items()
            if mode in MODES and isinstance(res, dict) and res.get("auc")
        }

        self.grad_cam = GradCAM(self.model.image_encoder.gradcam_layer())
        # The model keeps per-request state (attention maps, Grad-CAM
        # activations), so requests are served one at a time.
        self._lock = threading.Lock()

    @classmethod
    def from_disk(cls, cfg: Config) -> "InferenceService":
        device = torch.device(cfg.device)
        model_dir = cfg.model_dir
        logger.info("Loading models from %s onto %s", model_dir, device)

        checkpoint = load_state_dict(model_dir / "fusion_model_best.pth")
        if checkpoint is None:
            checkpoint = load_state_dict(model_dir / "fusion_model.pth")
        # Only download ImageNet / BioBERT weights when there is no trained
        # checkpoint to overwrite them with.
        model = MultiModalDetector.from_config(cfg, pretrained=checkpoint is None)
        fusion_loaded = load_weights(model, checkpoint, "Fusion model")
        if not fusion_loaded:
            logger.warning("No complete fusion checkpoint: predictions come from an untrained model.")

        modality_clf = ModalityClassifier(num_classes=3, pretrained=False)
        mc_loaded = load_weights(
            modality_clf, load_state_dict(model_dir / "modality_classifier.pth"), "Modality classifier",
        )
        if not mc_loaded:
            logger.warning("Modality classifier unavailable: using colour heuristic.")

        tokenizer_path = model_dir / "tokenizer"
        text_preprocessor = TextPreprocessor(
            model_name=cfg.text_model_name,
            max_length=cfg.text_max_length,
            tokenizer_path=str(tokenizer_path) if tokenizer_path.exists() else None,
        )

        return cls(
            cfg,
            model,
            modality_clf if mc_loaded else None,
            text_preprocessor,
            weights_loaded={"fusion_model": fusion_loaded, "modality_classifier": mc_loaded},
            metadata=_read_json(model_dir / "multimodal_config.json"),
            device=device,
        )

    def close(self) -> None:
        self.grad_cam.remove_hooks()

    @property
    def healthy(self) -> bool:
        return all(self.weights_loaded.values())

    # ── Public API ──────────────────────────────────────────────────────────

    def predict(
        self,
        image_bytes: Optional[bytes],
        symptoms: Optional[str],
        modality: Optional[str] = None,
        require_both: bool = False,
    ) -> PredictionResponse:
        start = time.perf_counter()

        modality = (modality or "auto").strip().lower()
        if modality not in ("auto", *MODALITIES):
            raise InvalidInput(f"Unknown modality '{modality}'. Use auto, chest, skin or brain.")

        text = (symptoms or "").strip()
        has_image, has_text = bool(image_bytes), bool(text)
        if require_both and not (has_image and has_text):
            raise InvalidInput("Multimodal prediction needs both an image and symptom text.")
        if not (has_image or has_text):
            raise InvalidInput("Provide at least an image or symptoms text.")

        image = None
        if has_image:
            try:
                image = decode_image(image_bytes)
            except ValueError as e:
                raise InvalidInput(str(e)) from e

        mode = "multimodal" if has_image and has_text else "image_only" if has_image else "text_only"

        with self._lock:
            response = self._predict(image, text, modality, mode)
        response.inference_time_ms = round((time.perf_counter() - start) * 1000, 1)
        return response

    # ── Internals ───────────────────────────────────────────────────────────

    def _predict(
        self,
        image: Optional[Image.Image],
        text: str,
        modality: str,
        mode: str,
    ) -> PredictionResponse:
        warnings = self._mode_warnings(mode)

        modality_confidence = None
        if modality != "auto":
            modality_source = "user"
        elif image is not None:
            modality, modality_confidence, modality_source = self._detect_image_modality(image)
        else:
            modality_source = "text"  # resolved from the text predictions below

        img_tensor = None
        if image is not None:
            img_tensor = preprocess_image(image, self.cfg.image_size).to(self.device)
        input_ids = attention_mask = None
        if text:
            encoded = self.text_preprocessor.tokenize(text)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

        def logit_fn() -> torch.Tensor:
            return self.model(
                image=img_tensor, input_ids=input_ids, attention_mask=attention_mask, mode=mode,
            )["logits"]

        with torch.no_grad():
            probs = torch.sigmoid(logit_fn())[0].cpu().numpy()

        if modality_source == "text":
            # No image and no hint: use the modality of the text model's top label.
            modality = self._modality_of_label(int(probs.argmax()))

        predictions = self._rank(probs, modality, mode)

        explanation = Explanation()
        if input_ids is not None:
            # Must run before Grad-CAM, which re-runs the forward pass.
            tokens, highlighted = self._text_explanation(input_ids, attention_mask, text)
            explanation.important_tokens, explanation.highlighted_text = tokens, highlighted
            if tokens is None:
                warnings.append("Text explanation could not be generated.")
        if image is not None and predictions:
            top = predictions[0]
            explanation.heatmap_base64 = self._heatmap(logit_fn, image, top.label_index)
            explanation.heatmap_class = top.disease
            if explanation.heatmap_base64 is None:
                warnings.append("Grad-CAM heatmap could not be generated.")

        return PredictionResponse(
            predictions=predictions,
            modality=modality,
            modality_source=modality_source,
            modality_confidence=modality_confidence,
            mode=mode,
            explanation=explanation,
            inference_time_ms=0.0,
            warnings=warnings,
        )

    def _mode_warnings(self, mode: str) -> List[str]:
        warnings = []
        if not self.weights_loaded.get("fusion_model", False):
            warnings.append("Trained weights were not found; predictions come from an untrained model.")
        auc = self.mode_val_auc.get(mode)
        if auc is not None and auc < self.cfg.min_mode_val_auc:
            warnings.append(
                f"The loaded checkpoint scored validation AUC {auc:.2f} in {mode} mode; "
                "treat these predictions as unreliable."
            )
        return warnings

    def _detect_image_modality(self, image: Image.Image) -> Tuple[str, Optional[float], str]:
        if self.modality_clf is None:
            return detect_modality_heuristic(image), None, "heuristic"

        tensor = preprocess_for_modality_classifier(image).to(self.device)
        result = self.modality_clf.predict_modality(tensor)
        modality, confidence = result["modality"], round(result["confidence"], 4)
        if confidence < self.cfg.min_modality_confidence:
            raise InvalidInput(
                "This image does not look like a chest X-ray, dermoscopy image or brain MRI "
                f"(best guess: {modality}, {confidence:.0%} confidence). "
                "Choose the modality manually if you are sure it is one of these.",
                status_code=422,
            )
        return modality, confidence, "classifier"

    def _modality_of_label(self, label_index: int) -> str:
        for modality in MODALITIES:
            start, end = self.cfg.modality_index_range(modality)
            if start <= label_index < end:
                return modality
        raise ValueError(f"Label index {label_index} belongs to no modality")

    def _rank(self, probs: np.ndarray, modality: str, mode: str) -> List[DiseaseScore]:
        labels = self.cfg.all_labels
        thresholds = self.thresholds[mode]
        start, end = self.cfg.modality_index_range(modality)
        scores = [
            DiseaseScore(
                disease=labels[i],
                confidence=round(float(probs[i]), 4),
                label_index=i,
                threshold=round(float(thresholds[i]), 4),
                positive=bool(probs[i] >= thresholds[i]),
            )
            for i in range(start, end)
        ]
        scores.sort(key=lambda s: s.confidence, reverse=True)
        return scores[: self.cfg.top_k_predictions]

    def _heatmap(
        self,
        logit_fn: Callable[[], torch.Tensor],
        image: Image.Image,
        class_idx: int,
    ) -> Optional[str]:
        try:
            return self.grad_cam.generate_base64(logit_fn, np.asarray(image), class_idx)
        except Exception:
            logger.exception("Grad-CAM failed")
            return None

    def _text_explanation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text: str,
    ) -> Tuple[Optional[List[TokenImportance]], Optional[str]]:
        try:
            attn = self.model.text_encoder.last_attention_weights
            if attn is None:
                return None, None
            tokens = self.text_preprocessor.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            importance = extract_token_importance(attn, tokens, attention_mask)
            explanation = format_explanation(importance, text, top_k=self.cfg.text_top_k_tokens)
            token_list = [
                TokenImportance(token=t["token"], importance=t["importance"])
                for t in explanation["important_tokens"]
            ]
            return token_list, explanation["highlighted_text"]
        except Exception:
            logger.exception("Text explanation failed")
            return None, None
