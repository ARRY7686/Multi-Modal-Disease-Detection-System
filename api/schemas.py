"""Pydantic schemas for the Multi-Modal Disease Detection API."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Modality = Literal["chest", "skin", "brain"]
Mode = Literal["image_only", "text_only", "multimodal"]
ModalitySource = Literal["user", "classifier", "heuristic", "text"]


# ── Response schemas ─────────────────────────────────────────────────────────

class DiseaseScore(BaseModel):
    disease: str
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Sigmoid output of the model (not a calibrated probability)",
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
