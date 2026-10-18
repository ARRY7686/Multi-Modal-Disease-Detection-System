"""
Grad-CAM implementation for the image encoder.

Generates class-activation heatmaps that highlight which image regions
contributed most to a particular disease prediction.

The caller passes a zero-argument ``logit_fn`` that runs the *same* path that
produced the prediction (image head for image-only mode, cross-attention fusion
for multimodal mode) and returns logits of shape (1, num_classes).  Grad-CAM
then differentiates the chosen disease logit with respect to the target conv
layer's activations, so the heatmap explains that disease and nothing else.
"""

from __future__ import annotations

import base64
import io
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

LogitFn = Callable[[], torch.Tensor]


class GradCAM:
    """
    Grad-CAM for any CNN that exposes a target conv layer.

    Usage:
        cam = GradCAM(target_layer=model.image_encoder.gradcam_layer())
        heatmap = cam.generate(lambda: model(image=x, mode="image_only")["logits"], class_idx=6)
    """

    def __init__(self, target_layer: torch.nn.Module):
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        # Kept attached to the graph so torch.autograd.grad can reach it.
        self._activations = output

    def generate(
        self,
        logit_fn: LogitFn,
        class_idx: int,
        output_size: Tuple[int, int] = (380, 380),
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for one class.

        Args:
            logit_fn:    runs the prediction path, returns (1, num_classes) logits
            class_idx:   target class index into those logits
            output_size: (H, W) of the returned heatmap

        Returns:
            heatmap: (H, W) float32 in [0, 1]
        """
        self._activations = None
        with torch.enable_grad():
            logits = logit_fn()
            if logits.dim() != 2 or logits.shape[0] != 1:
                raise ValueError(f"logit_fn must return (1, C) logits, got {tuple(logits.shape)}")
            if not 0 <= class_idx < logits.shape[1]:
                raise ValueError(f"class_idx {class_idx} out of range for {logits.shape[1]} classes")
            if self._activations is None:
                raise RuntimeError("Target layer was not executed by logit_fn")
            activations = self._activations
            # Gradients w.r.t. the activation only: no parameter .grad buffers are
            # filled, so repeated requests do not accumulate state on the model.
            (gradients,) = torch.autograd.grad(logits[0, class_idx], activations)
        self._activations = None

        weights = gradients[0].mean(dim=(1, 2))                        # (C,)
        cam = F.relu((weights[:, None, None] * activations[0]).sum(0))  # (h, w)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.detach().cpu().numpy().astype(np.float32)
        return cv2.resize(cam_np, (output_size[1], output_size[0]))

    def generate_overlay(
        self,
        logit_fn: LogitFn,
        original_image: np.ndarray,
        class_idx: int,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap overlaid on the original image.

        Args:
            logit_fn:       see generate()
            original_image: (H, W, 3) uint8 RGB
            class_idx:      target class index
            alpha:          overlay transparency

        Returns:
            overlay: (H, W, 3) uint8 RGB
        """
        h, w = original_image.shape[:2]
        heatmap = self.generate(logit_fn, class_idx, output_size=(h, w))

        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = (
            alpha * heatmap_color.astype(np.float32)
            + (1 - alpha) * original_image.astype(np.float32)
        )
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def generate_base64(
        self,
        logit_fn: LogitFn,
        original_image: np.ndarray,
        class_idx: int,
        alpha: float = 0.5,
    ) -> str:
        """Generate the overlay and return it as a base64-encoded PNG string."""
        overlay = self.generate_overlay(logit_fn, original_image, class_idx, alpha)
        buffer = io.BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def remove_hooks(self):
        """Clean up hooks when done."""
        self._fwd_hook.remove()
