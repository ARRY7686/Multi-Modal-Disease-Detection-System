"""Pydantic schemas for the Multi-Modal Disease Detection API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request schemas ──────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    symptoms: Optional[str] = Field(None, description="Patient symptom description")
    modality: Optional[str] = Field(
        None,
        description="Image modality: 'chest', 'skin', 'brain', or 'auto'",
    )


# ── Response schemas ─────────────────────────────────────────────────────────

class DiseaseScore(BaseModel):
    disease: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    label_index: int


class TokenImportance(BaseModel):
    token: str
    importance: float


class Explanation(BaseModel):
    heatmap_base64: Optional[str] = Field(
        None, description="Grad-CAM overlay as base64-encoded PNG"
    )
    important_tokens: Optional[List[TokenImportance]] = Field(
        None, description="Top-k tokens with importance scores"
    )
    highlighted_text: Optional[str] = Field(
        None, description="Original text with important tokens bolded"
    )


class PredictionResponse(BaseModel):
    predictions: List[DiseaseScore]
    modality: str
    mode: str = Field(..., description="'image_only', 'text_only', or 'multimodal'")
    explanation: Optional[Explanation] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]


class LabelResponse(BaseModel):
    labels: List[str]
    modality_groups: Dict[str, List[str]]
