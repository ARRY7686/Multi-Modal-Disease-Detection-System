"""Pydantic schemas for the Multi-Modal Disease Detection API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Modality = Literal["chest", "skin", "brain"]
Mode = Literal["image_only", "text_only", "multimodal"]
ModalitySource = Literal["user", "classifier", "heuristic", "text"]


# ── Response schemas ─────────────────────────────────────────────────────────

class DiseaseScore(BaseModel):
    disease: str
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Sigmoid output; temperature-scaled when the response says calibrated=true",
    )
    label_index: int
    threshold: float = Field(..., description="Decision threshold for this class")
    positive: bool = Field(..., description="confidence >= threshold")


class TokenImportance(BaseModel):
    token: str
    importance: float


class Explanation(BaseModel):
    heatmap_base64: Optional[str] = Field(
        None, description="Grad-CAM overlay for the top prediction, base64-encoded PNG"
    )
    heatmap_class: Optional[str] = Field(
        None, description="Disease the Grad-CAM heatmap explains"
    )
    important_tokens: Optional[List[TokenImportance]] = Field(
        None, description="Top-k words with importance scores"
    )
    highlighted_text: Optional[str] = Field(
        None, description="Original text with important words wrapped in **double asterisks**"
    )


class PredictionResponse(BaseModel):
    predictions: List[DiseaseScore]
    modality: Modality
    modality_source: ModalitySource = Field(
        ..., description="How the modality was chosen"
    )
    modality_confidence: Optional[float] = Field(
        None, description="Modality-classifier softmax confidence, when it was used"
    )
    mode: Mode
    explanation: Optional[Explanation] = None
    calibrated: bool = Field(
        False, description="Scores were temperature-scaled on validation data for this mode"
    )
    inference_time_ms: float
    warnings: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: Dict[str, bool] = Field(
        ..., description="True only when trained weights were found and loaded"
    )


class LabelResponse(BaseModel):
    labels: List[str]
    modality_groups: Dict[str, List[str]]


class ModelInfoResponse(BaseModel):
    """What the loaded checkpoint is and how it scored (models/multimodal_config.json)."""
    weights_loaded: bool
    image_backbone: Optional[str] = None
    text_model_name: Optional[str] = None
    fusion_type: Optional[str] = None
    test_results: Dict[str, Any] = Field(
        default_factory=dict, description="Per-mode held-out test metrics, as written by training"
    )
    temperatures: Dict[str, float] = Field(default_factory=dict)
    modality_classifier_test_accuracy: Optional[float] = None
    evaluation_protocol: Dict[str, str] = Field(default_factory=dict)
    data_counts: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    training_history: List[Dict[str, Any]] = Field(default_factory=list)
    chest_train_fraction: Optional[float] = None
    epochs: Optional[int] = None
