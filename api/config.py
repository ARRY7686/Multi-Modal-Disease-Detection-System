"""
Configuration loader — reads configs/model_config.yaml and exposes typed helpers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# Project root: two levels up from this file  (api/config.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Config:
    """Singleton-style accessor for the full YAML configuration."""

    _instance: Optional["Config"] = None
    _data: Dict[str, Any]

    def __init__(self, path: Path = CONFIG_PATH):
        self._data = _load_yaml(path)

    @classmethod
    def get(cls, path: Path = CONFIG_PATH) -> "Config":
        if cls._instance is None:
            cls._instance = cls(path)
        return cls._instance

    # ------ raw access ------
    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

    # ------ project ------
    @property
    def project_name(self) -> str:
        return self._data["project"]["name"]

    # ------ model dims ------
    @property
    def projection_dim(self) -> int:
        return self._data["model"]["projection_dim"]

    @property
    def dropout(self) -> float:
        return self._data["model"]["dropout"]

    @property
    def num_classes(self) -> int:
        return self._data["model"]["num_classes"]

    # ------ image encoder ------
    @property
    def image_backbone(self) -> str:
        return self._data["model"]["image_encoder"]["backbone"]

    @property
    def image_pretrained(self) -> bool:
        return self._data["model"]["image_encoder"]["pretrained"]

    @property
    def image_feature_dim(self) -> int:
        return self._data["model"]["image_encoder"]["feature_dim"]

    @property
    def image_size(self) -> int:
        return self._data["model"]["image_encoder"]["image_size"]

    @property
    def spatial_size(self) -> int:
        return self._data["model"]["image_encoder"]["spatial_size"]

    # ------ text encoder ------
    @property
    def text_model_name(self) -> str:
        return self._data["model"]["text_encoder"]["model_name"]

    @property
    def text_hidden_dim(self) -> int:
        return self._data["model"]["text_encoder"]["hidden_dim"]

    @property
    def text_max_length(self) -> int:
        return self._data["model"]["text_encoder"]["max_length"]

    # ------ fusion ------
    @property
    def fusion_type(self) -> str:
        return self._data["model"]["fusion"]["type"]

    @property
    def fusion_num_heads(self) -> int:
        return self._data["model"]["fusion"]["num_heads"]

    @property
    def fusion_hidden_dim(self) -> int:
        return self._data["model"]["fusion"]["hidden_dim"]

    @property
    def fusion_num_cross_layers(self) -> int:
        return self._data["model"]["fusion"]["num_cross_layers"]

    # ------ labels ------
    @property
    def all_labels(self) -> List[str]:
        labels = self._data["labels"]
        return labels["chest"] + labels["skin"] + labels["brain"]

    @property
    def chest_labels(self) -> List[str]:
        return self._data["labels"]["chest"]

    @property
    def skin_labels(self) -> List[str]:
        return self._data["labels"]["skin"]

    @property
    def brain_labels(self) -> List[str]:
        return self._data["labels"]["brain"]

    def modality_index_range(self, modality: str) -> Tuple[int, int]:
        """Return (start, end) indices for a modality in the 25-class vector."""
        r = self._data["modality_index_ranges"][modality]
        return (r[0], r[1])

    @property
    def modality_names(self) -> List[str]:
        return self._data["modality_names"]

    # ------ training ------
    def training_cfg(self, section: str) -> Dict[str, Any]:
        return self._data["training"][section]

    @property
    def loss_cfg(self) -> Dict[str, Any]:
        return self._data["training"]["loss"]

    # ------ inference ------
    @property
    def device(self) -> str:
        return self._data["inference"]["device"]

    @property
    def model_dir(self) -> Path:
        return PROJECT_ROOT / self._data["inference"]["model_dir"]

    @property
    def confidence_threshold(self) -> float:
        return self._data["inference"]["confidence_threshold"]

    @property
    def top_k_predictions(self) -> int:
        return self._data["inference"]["top_k_predictions"]

    @property
    def gradcam_target_layer(self) -> str:
        return self._data["inference"]["gradcam_target_layer"]

    @property
    def text_top_k_tokens(self) -> int:
        return self._data["inference"]["text_top_k_tokens"]

    # ------ API ------
    @property
    def api_host(self) -> str:
        return self._data["api"]["host"]

    @property
    def api_port(self) -> int:
        return self._data["api"]["port"]

    @property
    def cors_origins(self) -> List[str]:
        return self._data["api"]["cors_origins"]

    @property
    def max_upload_size_mb(self) -> int:
        return self._data["api"]["max_upload_size_mb"]
