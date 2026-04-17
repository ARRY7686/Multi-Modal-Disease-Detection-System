"""
Medical image augmentation pipelines.

Provides train-time augmentation and validation/inference transforms
for chest X-rays, skin lesions, and brain MRI scans.
"""

from __future__ import annotations

from typing import Optional

import torchvision.transforms as T

# ImageNet statistics (used for pretrained EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 380,
    modality: Optional[str] = None,
) -> T.Compose:
    """
    Training augmentation pipeline.

    Modality-aware:
      - chest:  no vertical flip (anatomy is orientation-dependent)
      - skin:   aggressive color jitter (lighting varies heavily)
      - brain:  minimal color augmentation (MRI is grayscale-like)
    """
    augmentations = [T.Resize((image_size, image_size))]

    # --- geometric ---
    if modality != "chest":
        augmentations.append(T.RandomVerticalFlip(p=0.3))
    augmentations.append(T.RandomHorizontalFlip(p=0.5))
    augmentations.append(T.RandomRotation(degrees=15))
    augmentations.append(T.RandomAffine(
        degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
    ))

    # --- photometric ---
    if modality == "skin":
        augmentations.append(T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ))
    elif modality == "brain":
        augmentations.append(T.ColorJitter(
            brightness=0.15, contrast=0.15
        ))
    else:
        augmentations.append(T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1
        ))

    # --- common tail ---
    augmentations.extend([
        T.RandomGrayscale(p=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    return T.Compose(augmentations)


def get_val_transforms(image_size: int = 380) -> T.Compose:
    """Deterministic val / test transforms — resize + normalize."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transform(image_size: int = 380) -> T.Compose:
    """Single-image inference transform (same as val)."""
    return get_val_transforms(image_size)


def get_modality_classifier_transforms(image_size: int = 224) -> T.Compose:
    """Transforms for the modality classifier (MobileNetV3 expects 224)."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
