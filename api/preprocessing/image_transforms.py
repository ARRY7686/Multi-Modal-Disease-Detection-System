"""
Image preprocessing — wraps augmentation pipelines for datasets and inference.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image
import torch
import torchvision.transforms as T

from utils.augmentation import (
    get_train_transforms,
    get_val_transforms,
    get_inference_transform,
    get_modality_classifier_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
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


def preprocess_uploaded_image(
    image_bytes: bytes,
    image_size: int = 380,
) -> torch.Tensor:
    """
    Preprocess an in-memory image (e.g. from a FastAPI upload).

    Returns:
        tensor: (1, 3, H, W)
    """
    import io
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_inference_transform(image_size)
    return transform(image).unsqueeze(0)


def preprocess_for_modality_classifier(
    image_bytes: bytes,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Preprocess for the modality classifier (MobileNetV3 @ 224×224).

    Returns:
        tensor: (1, 3, 224, 224)
    """
    import io
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_modality_classifier_transforms(image_size)
    return transform(image).unsqueeze(0)


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
