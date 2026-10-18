# Multi-Modal Disease Detection System

<div align="center">

**Multi-Modal Medical Diagnosis Prototype with Cross-Attention Fusion**

Combines **medical image analysis** (chest X-rays, dermoscopy, brain MRI) with **clinical symptom text understanding** (BioBERT) via **bidirectional cross-attention fusion** for 25-class multi-label diagnosis with full visual and textual explainability.

[![CI](https://github.com/ARRY7686/Multi-Modal-Disease-Detection-System/actions/workflows/ci.yml/badge.svg)](https://github.com/ARRY7686/Multi-Modal-Disease-Detection-System/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)

</div>

---

## Table of Contents

- [Highlights](#highlights)
- [System Architecture](#system-architecture)
- [Model Performance](#model-performance)
- [Supported Diseases (25 Classes)](#supported-diseases-25-classes)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training on Kaggle](#training-on-kaggle)
  - [Running Locally](#running-locally)
- [API Reference](#api-reference)
- [Frontend Features](#frontend-features)
- [Model Architecture Deep Dive](#model-architecture-deep-dive)
  - [Image Encoder](#image-encoder)
  - [Text Encoder](#text-encoder)
  - [Cross-Attention Fusion](#cross-attention-fusion)
  - [Modality Classifier](#modality-classifier)
- [Explainability](#explainability)
- [Training Pipeline](#training-pipeline)
- [Configuration](#configuration)
- [Technical Decisions](#technical-decisions)
- [Datasets](#datasets)
- [Disclaimer](#disclaimer)

---

## Highlights

- **Multi-modal inference** — Accepts medical images, clinical symptom text, or both simultaneously
- **25-class multi-label output** spanning 3 medical imaging modalities (chest, skin, brain)
- **Bidirectional cross-attention fusion** — Image and text representations attend to each other across 2 transformer layers with 8 attention heads
- **Automatic modality routing** — MobileNetV3-Small classifier auto-detects if an uploaded image is a chest X-ray, skin lesion photograph, or brain MRI scan (with a color-statistics heuristic fallback)
- **Dual explainability** — Grad-CAM heatmaps for image regions + attention-weighted token importance for symptom text
- **Three-phase Kaggle training** — Modular notebook pipeline (image → text → fusion) designed for Kaggle GPU instances
- **Production-ready API** — FastAPI backend with Pydantic validation, CORS, async file handling, and CPU-optimized inference
- **Modern frontend** — React 18 + Vite + Tailwind CSS with drag-and-drop upload, interactive heatmap overlay, and highlighted symptom tokens

---

## System Architecture

```mermaid
graph TB
    subgraph Input
        IMG[Medical Image] --> IE[EfficientNet-B4<br/>Image Encoder]
        TXT[Symptom Text] --> TE[BioBERT<br/>Text Encoder]
    end

    subgraph Modality Detection
        IMG --> MC[MobileNetV3-Small<br/>Modality Classifier]
        MC --> |chest / skin / brain| ROUTE[Route to Label Range]
    end

    subgraph Fusion
        IE --> |Spatial Features<br/>B×144×512| CA[Bidirectional Cross-Attention<br/>2 layers × 8 heads]
        TE --> |Token Features<br/>B×128×512| CA
        CA --> CONCAT[Concatenate Pooled<br/>1024-dim]
        CONCAT --> CLS[MLP Classifier<br/>1024→512→25]
    end

    subgraph Explainability
        IE --> GC[Grad-CAM<br/>Heatmap Overlay]
        TE --> ATT[Attention-based<br/>Token Importance]
    end

    CLS --> |sigmoid| PRED[Multi-label<br/>Predictions]
    GC --> EXP[Visual +<br/>Text Explanations]
    ATT --> EXP

    style CA fill:#4F46E5,color:#fff
    style PRED fill:#059669,color:#fff
    style EXP fill:#D97706,color:#fff
```

**Data flow summary:**

1. **Image path** — Input image (380×380) → EfficientNet-B4 backbone → spatial features (B, 144, 512) + pooled embedding (B, 512)
2. **Text path** — Symptom text → BioBERT tokenizer (max 128 tokens) → token features (B, seq, 512) + pooled embedding (B, 512)
3. **Fusion** — Bidirectional cross-attention (image attends to text; text attends to image) × 2 layers → mean-pooled → concatenated 1024-dim vector → MLP → 25 sigmoid outputs
4. **Modality routing** — MobileNetV3-Small classifies the image modality (chest/skin/brain) to filter the 25-class output to the relevant disease subset

---

## Model Performance

> [!WARNING]
> **The multimodal numbers below are inflated and are being replaced.** In the current
> training notebooks each image's symptom text is generated *from that image's ground-truth
> label*, in training and in validation, so the fusion model can read the diagnosis out of the
> text. This is why skin and brain score a perfect 1.000 while the image encoder alone reaches
> macro F1 0.42. The splits are also not patient-disjoint (NIH patients, HAM10000 `lesion_id`).
> A model retrained on real paired patient data with patient-level splits is in progress; until
> then, treat the image-encoder row as the only meaningful result.

### Multimodal Fusion (Cross-Attention)

| Metric | Score |
|--------|-------|
| **Macro F1** | **0.8564** |
| **Macro AUC-ROC** | **0.9889** |

### Per-Modality Breakdown

| Modality | F1 Score | AUC-ROC |
|----------|----------|---------|
| Skin (7 classes) | 1.0000 | 1.0000 |
| Brain (4 classes) | 1.0000 | 1.0000 |
| Chest (14 classes) | 0.7436 | 0.9058 |

### Individual Encoder Performance (Pre-Fusion)

| Encoder | Macro F1 (optimized thresholds) | Macro AUC-ROC |
|---------|--------------------------------|---------------|
| Image (EfficientNet-B4) | 0.4206 | 0.9216 |
| Text (BioBERT) | 0.9485 | 0.9986 |

### Training Convergence

Multimodal fusion training converged over 15 epochs (5 frozen + 10 unfrozen):

| Epoch | Train Loss | Train F1 | Val Loss | Val F1 | Val AUC |
|-------|-----------|----------|----------|--------|---------|
| 1 | 0.01060 | 0.4431 | 0.00339 | 0.8114 | 0.9827 |
| 5 | 0.00297 | 0.8468 | 0.00264 | 0.8475 | 0.9863 |
| 10 | 0.00261 | 0.8560 | 0.00247 | 0.8534 | 0.9883 |
| 15 | 0.00238 | 0.8679 | 0.00241 | 0.8564 | 0.9889 |

---

## Supported Diseases (25 Classes)

| Index Range | Modality | Diseases | Dataset |
|-------------|----------|----------|---------|
| 0–13 | **Chest X-ray** | Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia | NIH ChestX-ray14 |
| 14–20 | **Skin (Dermoscopy)** | Melanoma, Melanocytic Nevi, Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis, Dermatofibroma, Vascular Lesion | HAM10000 |
| 21–24 | **Brain MRI** | Glioma, Meningioma, Pituitary Tumor, No Tumor | Brain Tumor Classification (MRI) |

---

## Project Structure

```
multimodalAI/
├── api/                              # FastAPI backend
│   ├── __init__.py
│   ├── main.py                       # FastAPI app factory + endpoints (thin HTTP layer)
│   ├── service.py                    # InferenceService: model loading, routing, prediction, explanations
│   ├── schemas.py                    # Pydantic request/response models
│   ├── config.py                     # YAML config loader (singleton pattern)
│   ├── requirements.txt              # Python dependencies
│   ├── models/                       # PyTorch model definitions
│   │   ├── __init__.py
│   │   ├── image_encoder.py          # EfficientNet-B4 (1792→512 projection)
│   │   ├── text_encoder.py           # BioBERT (768→512 projection)
│   │   ├── fusion.py                 # Cross-attention blocks + classifier
│   │   ├── multimodal.py             # End-to-end MultiModalDetector
│   │   └── modality_classifier.py    # MobileNetV3-Small (3-class)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── image_transforms.py       # Resize, normalize, inference transforms
│   │   └── text_processing.py        # Medical text tokenization (BioBERT)
│   └── explainability/
│       ├── __init__.py
│       ├── gradcam.py                # Grad-CAM heatmap generation
│       └── text_attention.py         # Token importance via attention weights
│
├── frontend/                         # React + Vite + Tailwind CSS
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── App.jsx                   # Main app — state, routing, layout
│       ├── api.js                    # Axios API client
│       ├── main.jsx                  # React entry point
│       ├── index.css                 # Tailwind base styles
│       └── components/
│           ├── ImageUpload.jsx       # Drag-and-drop image upload with preview
│           ├── SymptomInput.jsx      # Symptom text area + quick examples
│           ├── PredictionResults.jsx # Confidence bar chart with color coding
│           ├── HeatmapViewer.jsx     # Grad-CAM overlay with toggle
│           └── ExplanationPanel.jsx  # Highlighted text + token importance
│
├── training/                         # Honest training pipeline (imports api/models)
│   ├── data.py                       # Loaders, patient-disjoint splits, metadata -> text
│   ├── evaluate.py                   # Per-modality metrics, bootstrap CIs, calibration, thresholds
│   └── train.py                      # python -m training.train: trains + writes every API artifact
│
├── kaggle_notebooks/
│   ├── train_v2.ipynb                # Kaggle runner for training/train.py
│   ├── image_training.ipynb          # Legacy v1 (label-leaking text, not patient-disjoint)
│   ├── text_training.ipynb           # Legacy v1
│   └── multimodal_training.ipynb     # Legacy v1
│
├── configs/
│   └── model_config.yaml             # Central config (architecture, labels, training)
│
├── models/                           # Trained model weights
│   ├── fusion_model_best.pth         # Full multimodal state dict (best F1)
│   ├── fusion_model.pth              # Final epoch state dict
│   ├── image_encoder_fused.pth       # Image encoder after fusion training
│   ├── text_encoder_fused.pth        # Text encoder after fusion training
│   ├── cross_attention_fusion.pth    # Fusion module weights
│   ├── image_head_fused.pth          # Image classification head
│   ├── text_head_fused.pth           # Text classification head
│   ├── modality_classifier.pth       # MobileNetV3 modality classifier
│   └── multimodal_config.json        # Architecture config + training results
│
├── utils/                            # Shared utilities (used by notebooks)
│   ├── __init__.py
│   ├── losses.py                     # Focal loss, weighted BCE
│   ├── metrics.py                    # F1, AUC-ROC, per-class threshold optimization
│   ├── augmentation.py               # Modality-aware image transforms
│   └── synthetic_symptoms.py         # Clinical symptom text generation
│
├── scripts/
│   ├── download_models.py            # Download weights from Kaggle
│   └── setup.sh                      # One-command project setup
│
├── tests/                            # pytest suite on tiny offline models (no downloads)
├── .github/workflows/ci.yml          # Lint + tests + frontend build on every push / PR
├── pyproject.toml                    # pytest + ruff configuration
├── requirements-dev.txt              # Test / lint dependencies
│
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- ~**2 GB disk space** for model weights + cached HuggingFace models
- No GPU required for inference (CPU-only)

### Installation

```bash
# Clone the repository
git clone <repo-url> multimodalAI
cd multimodalAI

# Backend setup
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows

pip install -r api/requirements.txt

# Frontend setup
cd frontend
npm install
cd ..
```

### Training on Kaggle

Training lives in the `training/` package and imports the model code from `api/models`, so the
weights it writes always load into the API. `kaggle_notebooks/train_v2.ipynb` is a thin runner:
it clones this repository and calls the training script.

1. Create a Kaggle notebook from `kaggle_notebooks/train_v2.ipynb`, with **GPU T4** and **Internet on**.
2. Add the input datasets `nih-chest-xrays/data`, `kmader/skin-cancer-mnist-ham10000` and
   `sartajbhuvaji/brain-tumor-classification-mri`.
3. Run all (about 2.5 h with the defaults). Download `models.zip` and unzip it into `models/`.

The same script runs anywhere the datasets are on disk:

```bash
python -m training.train --data-root /path/to/datasets --out models \
    --chest-train-fraction 0.3 --epochs 8
python -m training.train --help   # every setting is a flag
```

Outputs: `fusion_model_best.pth`, `modality_classifier.pth`, `tokenizer/`,
`multimodal_config.json` (per-mode thresholds and temperatures, test results with confidence
intervals, training history) and `results.md` (the tables for this README).

The original three-phase notebooks (`image_training.ipynb`, `text_training.ipynb`,
`multimodal_training.ipynb`) are kept for reference only. They pair each image with symptom
text generated from its own label and split without patient grouping, so their reported
multimodal scores are not valid.

### Running Locally

```bash
# Terminal 1 — Start the API server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Start the frontend dev server
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

### Running Tests

The test suite builds the same architecture at toy size (EfficientNet-B0 at 64 px, a 2-layer
BERT with a local vocabulary), so it needs no model downloads or trained weights:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
ruff check api utils tests scripts training
pytest
```

CI runs the same commands, plus `npm run build` for the frontend, on every push and pull request.

On first startup, the API will download **EfficientNet-B4** (~75 MB) and **BioBERT** (~433 MB) pretrained weights from PyTorch Hub and HuggingFace respectively. These are cached for subsequent runs.

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns model load status |
| `GET` | `/labels` | All 25 disease labels grouped by modality |
| `POST` | `/predict` | Flexible prediction — auto-detects mode (image-only, text-only, or multimodal) |
| `POST` | `/predict-multimodal` | Full multimodal prediction (requires both image + symptom text) |

### `POST /predict`

Accepts any combination of image and/or symptoms. Auto-detects the inference mode.

**Form parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File | No | Medical image (JPEG, PNG) |
| `symptoms` | string | No | Clinical symptom description |
| `modality` | string | No | `"chest"`, `"skin"`, `"brain"`, or `"auto"` (default: auto-detect) |

**How the modality is chosen** (reported as `modality_source`):

| Inputs | Source | Behaviour |
|--------|--------|-----------|
| `modality` given | `user` | Used as-is |
| Image, auto | `classifier` | MobileNetV3 modality classifier. Below `min_modality_confidence` (0.6) the request is rejected with **422**, so out-of-scope images do not get a diagnosis |
| Image, auto, no classifier weights | `heuristic` | Colour-statistics fallback |
| Text only, auto | `text` | Modality of the text model's top-scoring label |

**Errors:** `400` missing or invalid input or unreadable image, `413` upload over `max_upload_size_mb`, `422` image not recognised as a supported modality.

### `POST /predict-multimodal`

Full cross-attention multimodal inference. Requires both inputs.

**Form parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File | Yes | Medical image (JPEG, PNG) |
| `symptoms` | string | Yes | Clinical symptom description |
| `modality` | string | No | Override modality detection |

### Response Schema

```json
{
  "predictions": [
    {
      "disease": "Infiltration",
      "confidence": 0.7512,
      "label_index": 3,
      "threshold": 0.35,
      "positive": true
    },
    {
      "disease": "Pneumonia",
      "confidence": 0.2341,
      "label_index": 6,
      "threshold": 0.4,
      "positive": false
    }
  ],
  "modality": "chest",
  "modality_source": "classifier",
  "modality_confidence": 0.9931,
  "mode": "multimodal",
  "explanation": {
    "heatmap_base64": "iVBORw0KGgo...",
    "heatmap_class": "Infiltration",
    "important_tokens": [
      { "token": "consolidation", "importance": 1.0 },
      { "token": "infection", "importance": 0.87 }
    ],
    "highlighted_text": "Patient presents with bilateral **consolidation**..."
  },
  "calibrated": true,
  "inference_time_ms": 1840.2,
  "warnings": []
}
```

- `confidence` is the sigmoid output, temperature-scaled per mode when `calibrated` is true (temperatures are fitted on validation by `training/train.py` and saved as `"temperatures"` in `multimodal_config.json`). `positive` compares it with the class's decision threshold, read from `"thresholds"` in `models/multimodal_config.json` when training saved them (a list, or `{mode: list}`), else `inference.confidence_threshold`.
- `warnings` flags results to distrust: no trained weights found, or a mode whose recorded validation AUC in `multimodal_config.json` is below `min_mode_val_auc` (the current checkpoint's image-only and text-only heads score AUC 0.46 and 0.58).
- `/health` returns `"degraded"`, with `false` entries in `models_loaded`, when checkpoint files are missing or do not cover every parameter.

### Example: cURL

```bash
# Image only (auto-detect modality)
curl -X POST http://localhost:8000/predict \
  -F "image=@chest_xray.png"

# Text only
curl -X POST http://localhost:8000/predict \
  -F "symptoms=A 65-year-old male with productive cough, fever, and dyspnea"

# Multimodal with manual modality override
curl -X POST http://localhost:8000/predict-multimodal \
  -F "image=@skin_lesion.jpg" \
  -F "symptoms=A 58-year-old male with a gradually enlarging pigmented lesion" \
  -F "modality=skin"
```

---

## Frontend Features

| Feature | Description |
|---------|-------------|
| **Drag-and-drop image upload** | Upload medical images with instant preview and file size validation |
| **Symptom text input** | Free-text clinical description with pre-filled quick examples |
| **Modality selector** | Manual override buttons (Auto-detect / Chest / Skin / Brain) |
| **Prediction results** | Top-5 predictions with color-coded confidence bars (red ≥80%, orange ≥50%, yellow ≥30%, green <30%) |
| **Grad-CAM heatmap** | Interactive overlay with Show Original / Heatmap Overlay toggle |
| **Text explanation** | Highlighted symptom text with the most important tokens bolded + token importance chips |
| **API status indicator** | Live health check indicator showing API Online/Offline state |
| **Auto mode detection** | Displays current mode (Image only / Text only / Multimodal) based on provided inputs |

---

## Model Architecture Deep Dive

### Image Encoder

```
EfficientNet-B4 (pretrained on ImageNet)
  └── Backbone features (1792 channels, 12×12 spatial)
      ├── Global average pool → Linear(1792, 512) → Pooled embedding (B, 512)
      └── Spatial projection → Linear(1792, 512) → Spatial features (B, 144, 512)
```

- **Backbone:** `torchvision.models.efficientnet_b4` with ImageNet weights
- **Input:** 380×380 RGB image (ImageNet-normalized)
- **Pooled output:** 512-dim vector (used for single-modality image head)
- **Spatial output:** 144 spatial tokens × 512-dim (used for cross-attention with text)

### Text Encoder

```
BioBERT v1.1 (dmis-lab/biobert-v1.1)
  └── BERT encoder → last_hidden_state
      ├── [CLS] token → Linear(768, 512) → Pooled embedding (B, 512)
      └── All tokens → Linear(768, 512) → Token features (B, seq, 512)
```

- **Backbone:** BERT-base architecture (12 layers, 768 hidden dim) pre-trained on PubMed abstracts + PMC full-text articles
- **Tokenizer:** WordPiece tokenizer with 28,996 vocab tokens
- **Max length:** 128 tokens
- **Attention weights** are captured for text explainability

### Cross-Attention Fusion

The fusion module performs **bidirectional cross-attention** — image features attend to text tokens and vice versa — followed by classification.

```
CrossAttentionBlock × 2 layers:
  ├── Image→Text cross-attention:  Q=image, K=V=text  (8 heads)
  │   └── LayerNorm + Residual → FFN → LayerNorm + Residual
  └── Text→Image cross-attention:  Q=text, K=V=image  (8 heads)
      └── LayerNorm + Residual → FFN → LayerNorm + Residual

Mean-pool image features → 512-dim
Mean-pool text features  → 512-dim
Concatenate → 1024-dim

Classifier: Linear(1024, 512) → ReLU → Dropout(0.3) → Linear(512, 25)
```

- **Attention mechanism:** `nn.MultiheadAttention` with 8 heads, 512-dim, 0.1 attention dropout
- **FFN:** Two-layer MLP (512→2048→512) with GELU activation
- **Mask handling:** Attention mask (1=real token, 0=padding) is passed and inverted internally for PyTorch's key_padding_mask

### Modality Classifier

```
MobileNetV3-Small (pretrained on ImageNet)
  └── Features → AdaptiveAvgPool → Linear(576, 256) → HardSwish → Dropout → Linear(256, 3)
```

- **Input:** 224×224 resized image
- **Output:** softmax over 3 classes [chest, skin, brain]
- **Fallback:** If weights are unavailable, a color-statistics heuristic detects modality:
  - High saturation → skin
  - Grayscale + dark borders → brain MRI
  - Otherwise → chest X-ray

---

## Explainability

### Grad-CAM (Image)

Gradient-weighted Class Activation Mapping applied to the last convolutional block of EfficientNet-B4.

- **Target layer:** Last convolutional block (`ImageEncoder.gradcam_layer()`, i.e. `features[0][-1]`)
- **Process:** Re-run the *same* path that produced the prediction (image head in image-only mode, cross-attention fusion in multimodal mode) → differentiate the top predicted disease's logit with respect to the target layer's activations → weight feature maps by mean gradients → ReLU → upsample to the original image size
- Gradients are taken with `torch.autograd.grad` on the activation only, so no parameter `.grad` buffers build up across requests
- **Output:** Semi-transparent heatmap overlaid on the original image, encoded as base64 PNG

### Token Importance (Text)

Attention-based importance scoring from BioBERT's self-attention layers.

- **Process:** Extract attention weights from the last BERT layer → average across heads → take [CLS] token's attention over all tokens → normalize to [0, 1]
- **Subword merging:** WordPiece pieces are merged over the full sequence (a `##` piece only joins the token directly before it) and scored by their mean; punctuation is dropped, then scores are renormalised and cut to the top-k words
- **Output:** Top-K important tokens with scores + full text with important tokens bolded in HTML

---

## Training Pipeline

```
NIH ChestX-ray14 ─┐  official patient-disjoint test list; validation split by Patient ID;
                  │  text = age, sex, PA/AP view; "No Finding" X-rays kept
HAM10000 ─────────┤  split by lesion_id; text = age, sex, lesion site (dx_type unused: it encodes the label)
Brain Tumor MRI ──┘  official Testing folder minus duplicates of Training; text = "No clinical details provided."
        │
        ▼
EfficientNet-B4 ──┬── image head ──────────────► image-only logits      ┐
                  └── spatial tokens ─┐                                 │  masked BCE on each
                          cross-attention fusion ──► multimodal logits  ├─ sample's own modality,
BioBERT ──────────┬── token features ─┘                                 │  all three heads every step
                  └── text head ───────────────► text-only logits       ┘
        │
        ▼
best epoch by validation AUROC → temperatures + per-class thresholds fitted on validation
        → test set evaluated once (bootstrap 95% CIs, ECE) → artifacts the API loads
```

- **No label leakage:** the paired text is recorded patient metadata, never derived from the
  diagnosis. A test (`tests/test_training.py`) checks that no label word appears in any paired text.
- **Missing-modality robustness:** in training, the text is replaced by "No clinical details
  provided." 20% of the time, so the model works without it and the placeholder is not a
  give-away for brain scans.
- **Sampling:** modality share proportional to the square root of its size (`--sampling-power 0.5`).
- **Schedule:** 1 epoch with frozen backbones, then full fine-tuning with AdamW and warmup + cosine
  (head 3e-4, EfficientNet 1e-4, BioBERT 2e-5), mixed precision on GPU.

---

## Configuration

All architecture, training, and inference settings are centralized in `configs/model_config.yaml`:

| Section | Key Parameters |
|---------|---------------|
| **Model** | `projection_dim: 512`, `dropout: 0.3`, `num_classes: 25` |
| **Image encoder** | `backbone: efficientnet_b4`, `image_size: 380`, `feature_dim: 1792` |
| **Text encoder** | `model_name: dmis-lab/biobert-v1.1`, `max_length: 128`, `hidden_dim: 768` |
| **Fusion** | `type: cross_attention`, `num_heads: 8`, `num_cross_layers: 2` |
| **Labels** | 25 diseases mapped to [0,14), [14,21), [21,25) index ranges |
| **Inference** | `device: cpu`, `top_k_predictions: 5`, `confidence_threshold: 0.5`, `min_modality_confidence: 0.6`, `min_mode_val_auc: 0.7` |
| **API** | `port: 8000`, `cors_origins: [localhost:5173, localhost:3000]`, `max_upload_size_mb: 50` |

---

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Image backbone | EfficientNet-B4 | 1792-dim features with compound scaling — best accuracy/efficiency tradeoff for medical imaging |
| Text encoder | BioBERT v1.1 | Pre-trained on 18B tokens from PubMed abstracts + PMC articles — understands medical terminology |
| Fusion strategy | Bidirectional cross-attention (2 layers, 8 heads) | Richer interaction than simple concatenation — text attends to spatial image regions and vice versa |
| Loss function | Focal Loss (α=0.75, γ=2.0) | Downweights easy negatives; critical for severe class imbalance (14 chest diseases vs. 4 brain) |
| Training strategy | Frozen → Unfrozen with differential LR | Prevents catastrophic forgetting; encoder LR (1e-6) is 100× lower than fusion LR (1e-4) |
| Modality detection | MobileNetV3-Small + heuristic fallback | Lightweight (2.5M params); heuristic provides graceful degradation if weights unavailable |
| Inference device | CPU-only | Eliminates GPU dependency for deployment; inference takes ~2-5s per prediction |
| Explainability | Grad-CAM + attention weights | Clinically important — doctors need to see *why* the model predicts what it does |
| Frontend | React 18 + Vite + Tailwind | Fast HMR development, utility-first styling, modern component architecture |
| API framework | FastAPI | Async support, auto-generated OpenAPI docs, Pydantic validation, native file upload handling |

---

## Datasets

| Dataset | Size | Classes | Modality | Source |
|---------|------|---------|----------|--------|
| [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) | 45.1 GB (30% used) | 14 | Chest X-ray | NIH Clinical Center |
| [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | 2.9 GB | 7 | Dermoscopy | ISIC Archive |
| [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) | ~90 MB | 4 | Brain MRI | Kaggle |
| Synthetic Symptoms | Generated at runtime | 25 | Clinical text | GPT-generated templates × 200/disease |

---

## Disclaimer

> **This system is for research and educational purposes only.**
>
> It is **not** a medical device and has **not** been validated for clinical diagnosis. Predictions should not be used to make medical decisions. Always consult a qualified healthcare professional.
>
> Medical image datasets are subject to their respective licenses. See each dataset page on Kaggle for terms of use.
