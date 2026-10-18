"""
Shared fixtures: a tiny, fully offline InferenceService.

Same architecture as production (EfficientNet → cross-attention ← BERT,
MobileNetV3 modality classifier) but with EfficientNet-B0 at 64 px and a
2-layer BERT with a local vocabulary, so tests need no downloads and run
in seconds on CPU.
"""

from __future__ import annotations

import inspect
import io

import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from PIL import Image
from transformers import BertConfig, BertTokenizerFast

from api.config import CONFIG_PATH, Config
from api.main import create_app
from api.models import ModalityClassifier, MultiModalDetector
from api.preprocessing.text_processing import TextPreprocessor
from api.service import InferenceService

VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "a", "patient", "with", "fever", "cough", "and", "dys", "##pnea",
    "mole", "irregular", "borders", "headache", "seizures", "year", "old",
    "-", ".", ",",
]


@pytest.fixture(scope="session")
def tiny_text_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_bert")
    vocab_file = path / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB), encoding="utf-8")
    # transformers 5 takes `vocab`; 4.x takes `vocab_file` (and would ignore `vocab`)
    if "vocab" in inspect.signature(BertTokenizerFast.__init__).parameters:
        tokenizer = BertTokenizerFast(vocab={tok: i for i, tok in enumerate(VOCAB)})
    else:
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
    assert tokenizer.tokenize("fever dyspnea") == ["fever", "dys", "##pnea"]
    tokenizer.save_pretrained(path)
    BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    ).save_pretrained(path)
    return path


@pytest.fixture
def tiny_cfg(tiny_text_dir) -> Config:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    model = data["model"]
    model["projection_dim"] = 32
    model["image_encoder"].update(
        backbone="efficientnet_b0", feature_dim=1280, image_size=64, pretrained=False,
    )
    model["text_encoder"].update(model_name=str(tiny_text_dir), hidden_dim=32, max_length=32)
    model["fusion"].update(num_heads=2, num_cross_layers=1)
    return Config.from_dict(data)


def set_logits(head: torch.nn.Sequential, logits) -> None:
    """Make the last Linear of `head` output constant `logits` whatever its input."""
    last = head[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.as_tensor(logits, dtype=torch.float32))


def build_service(
    cfg: Config,
    *,
    metadata=None,
    weights_loaded=None,
) -> InferenceService:
    torch.manual_seed(0)
    model = MultiModalDetector.from_config(cfg, pretrained=False)
    modality_clf = ModalityClassifier(num_classes=3, pretrained=False)
    set_logits(modality_clf.classifier, [5.0, 0.0, 0.0])  # confidently "chest"
    text_preprocessor = TextPreprocessor(
        model_name=cfg.text_model_name, max_length=cfg.text_max_length,
    )
    return InferenceService(
        cfg,
        model,
        modality_clf,
        text_preprocessor,
        weights_loaded=weights_loaded or {"fusion_model": True, "modality_classifier": True},
        metadata=metadata,
    )


@pytest.fixture
def service(tiny_cfg):
    svc = build_service(tiny_cfg)
    yield svc
    svc.close()


@pytest.fixture
def client(service, tiny_cfg):
    with TestClient(create_app(service=service, cfg=tiny_cfg)) as c:
        yield c


def png_bytes(size=(80, 60), color=(120, 120, 120)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()
