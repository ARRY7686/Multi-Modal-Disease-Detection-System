"""
Grad-CAM implementation for the image encoder.

Generates class-activation heatmaps that highlight which image regions
contributed most to a particular disease prediction.
"""

from __future__ import annotations

import base64
import io
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM for any CNN that exposes a target conv layer.

    Usage:
        cam = GradCAM(model.image_encoder, target_layer=model.image_encoder.features[-1])
        heatmap = cam.generate(image_tensor, class_idx=6)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        image_size: Tuple[int, int] = (380, 380),
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a specific class.

        Args:
            input_tensor: (1, 3, H, W)
            class_idx:    target class index
            image_size:   output heatmap resolution

        Returns:
            heatmap: (H, W) float32 in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        output = self.model(input_tensor)
        if isinstance(output, dict):
            output = output["logits"]

        # Backward for target class
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward(retain_graph=True)

        # Compute Grad-CAM
        gradients = self._gradients[0]     # (C, H, W)
        activations = self._activations[0] # (C, H, W)

        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        cam = F.relu(cam)   # Only positive contributions
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to image dimensions
        cam_np = cam.cpu().numpy()
        cam_resized = cv2.resize(cam_np, (image_size[1], image_size[0]))

        return cam_resized

    def generate_overlay(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_idx: int,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap overlaid on the original image.

        Args:
            input_tensor:   (1, 3, H, W)
            original_image: (H, W, 3) uint8 BGR or RGB
            class_idx:      target class index
            alpha:          overlay transparency

        Returns:
            overlay: (H, W, 3) uint8 RGB
        """
        h, w = original_image.shape[:2]
        heatmap = self.generate(input_tensor, class_idx, image_size=(h, w))

        # Convert heatmap to colour
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = (
            alpha * heatmap_color.astype(np.float32)
            + (1 - alpha) * original_image.astype(np.float32)
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay

    def generate_base64(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_idx: int,
        alpha: float = 0.5,
    ) -> str:
        """
        Generate overlay and return as base64-encoded PNG string.
        """
        overlay = self.generate_overlay(
            input_tensor, original_image, class_idx, alpha
        )
        pil_image = Image.fromarray(overlay)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def remove_hooks(self):
        """Clean up hooks when done."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
