"""
Image preprocessing — wraps augmentation pipelines for datasets and inference.
"""

from __future__ import annotations

import io
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError

from utils.augmentation import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_inference_transform,
    get_modality_classifier_transforms,
    get_train_transforms,
    get_val_transforms,
)


class ImagePreprocessor:
    """Stateful image preprocessor that caches the transform pipeline."""

    def __init__(
        self,
        image_size: int = 380,
        modality: Optional[str] = None,
        is_training: bool = False,
    ):
        self.image_size = image_size
        self.modality = modality
        self.is_training = is_training

        if is_training:
            self.transform = get_train_transforms(image_size, modality)
        else:
            self.transform = get_val_transforms(image_size)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Args:
            image: PIL Image (RGB)
        Returns:
            tensor: (3, H, W) normalised
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)


def load_and_preprocess(
    image_path: str,
    image_size: int = 380,
) -> torch.Tensor:
    """
    Load an image from disk and apply inference transforms.

    Returns:
        tensor: (1, 3, H, W) batched, normalised
    """
    image = Image.open(image_path).convert("RGB")
    transform = get_inference_transform(image_size)
    return transform(image).unsqueeze(0)


def decode_image(image_bytes: bytes) -> Image.Image:
    """
    Decode uploaded bytes into an RGB PIL image.

    Raises:
        ValueError: if the bytes are not a readable image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ValueError(f"Could not read the uploaded file as an image: {e}") from e


def preprocess_image(
    image: Image.Image,
    image_size: int = 380,
) -> torch.Tensor:
    """
    Apply inference transforms to a decoded RGB image.

    Returns:
        tensor: (1, 3, H, W)
    """
    return get_inference_transform(image_size)(image).unsqueeze(0)


def preprocess_for_modality_classifier(
    image: Image.Image,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Preprocess for the modality classifier (MobileNetV3 @ 224x224).

    Returns:
        tensor: (1, 3, 224, 224)
    """
    return get_modality_classifier_transforms(image_size)(image).unsqueeze(0)


def inverse_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation for visualisation / Grad-CAM overlay.

    Args:
        tensor: (3, H, W) or (B, 3, H, W)
    Returns:
        tensor in [0, 1] range
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return (tensor * std + mean).clamp(0, 1)
