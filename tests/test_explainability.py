import pytest
import torch
import torch.nn as nn

from api.explainability.gradcam import GradCAM
from api.explainability.text_attention import (
    extract_token_importance,
    format_explanation,
    merge_subword_importance,
)
from api.models import ImageEncoder


class MirrorFeatures(nn.Module):
    """Channel 0 is the image, channel 1 its left-right mirror."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.scale * torch.cat([x, x.flip(-1)], dim=1)


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

def test_gradcam_heatmap_follows_the_target_class():
    # Class 0 reads channel 0 (stripe on the left), class 1 reads channel 1
    # (the mirror, stripe on the right). The heatmap must move with the class.
    x = torch.zeros(1, 1, 8, 8)
    x[..., :2] = 1.0
    layer = MirrorFeatures()
    cam = GradCAM(layer)

    def logit_fn():
        return layer(x).mean(dim=(2, 3))  # (1, 2)

    left = cam.generate(logit_fn, class_idx=0, output_size=(8, 8))
    right = cam.generate(logit_fn, class_idx=1, output_size=(8, 8))

    assert left[:, :2].min() > 0.99 and left[:, 2:].max() == 0
    assert right[:, -2:].min() > 0.99 and right[:, :-2].max() == 0


def test_gradcam_rejects_out_of_range_class():
    layer = MirrorFeatures()
    cam = GradCAM(layer)
    with pytest.raises(ValueError):
        cam.generate(lambda: layer(torch.ones(1, 1, 4, 4)).mean(dim=(2, 3)), class_idx=5)


def test_gradcam_does_not_fill_parameter_grads():
    encoder = ImageEncoder("efficientnet_b0", pretrained=False, feature_dim=1280, projection_dim=16)
    encoder.eval()
    head = nn.Linear(16, 25)
    cam = GradCAM(encoder.gradcam_layer())

    heatmap = cam.generate(lambda: head(encoder(torch.rand(1, 3, 64, 64))), class_idx=20)

    assert heatmap.shape == (380, 380)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert all(p.grad is None for p in encoder.parameters())


# ── Text token importance ────────────────────────────────────────────────────

def test_subword_merges_only_into_the_adjacent_token():
    # '##pnea' at index 4 must not be glued onto 'fever' at index 1
    # (the old implementation produced 'feverpnea' after top-k filtering).
    items = [
        {"token": "fever", "importance": 0.9, "index": 1},
        {"token": "##pnea", "importance": 0.8, "index": 4},
    ]
    tokens = {m["token"] for m in merge_subword_importance(items)}
    assert tokens == {"fever", "pnea"}


def test_subwords_are_averaged_and_renormalised():
    items = [
        {"token": "fever", "importance": 0.3, "index": 1},
        {"token": "dys", "importance": 0.4, "index": 3},
        {"token": "##pnea", "importance": 0.8, "index": 4},
    ]
    merged = merge_subword_importance(items)
    assert merged[0] == {"token": "dyspnea", "importance": 1.0}
    assert merged[1] == {"token": "fever", "importance": 0.5}


def test_format_explanation_highlights_merged_words():
    tokens = ["[CLS]", "fever", "and", "dys", "##pnea", ".", "[SEP]", "[PAD]"]
    attn = torch.zeros(1, 8, 8)
    attn[0, 0] = torch.tensor([0.0, 0.2, 0.05, 0.5, 0.6, 0.9, 0.1, 0.7])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])

    importance = extract_token_importance(attn, tokens, mask)
    explanation = format_explanation(importance, "Fever and dyspnea.", top_k=2)

    # punctuation is dropped from the ranking, padding is masked out
    assert [t["token"] for t in explanation["important_tokens"]] == ["dyspnea", "fever"]
    assert explanation["highlighted_text"] == "**Fever** and **dyspnea.**"
