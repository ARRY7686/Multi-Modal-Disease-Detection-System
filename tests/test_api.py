import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from tests.conftest import build_service, png_bytes, set_logits

MELANOMA = 14
SKIN = range(14, 21)
CHEST = range(0, 14)
BRAIN = range(21, 25)


def image_file(**kwargs):
    return {"image": ("scan.png", png_bytes(**kwargs), "image/png")}


def label_indices(body):
    return {p["label_index"] for p in body["predictions"]}


# ── Health / labels ──────────────────────────────────────────────────────────

def test_health_ok_when_weights_loaded(client):
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "models_loaded": {"fusion_model": True, "modality_classifier": True},
    }


def test_health_degraded_and_warns_without_weights(tiny_cfg):
    svc = build_service(tiny_cfg, weights_loaded={"fusion_model": False, "modality_classifier": True})
    with TestClient(create_app(service=svc, cfg=tiny_cfg)) as client:
        assert client.get("/health").json()["status"] == "degraded"
        body = client.post("/predict", data={"symptoms": "fever and cough"}).json()
    assert any("untrained" in w for w in body["warnings"])


def test_labels_lists_25_classes(client):
    body = client.get("/labels").json()
    assert len(body["labels"]) == 25
    assert set(body["modality_groups"]) == {"chest", "skin", "brain"}


# ── Modality routing ─────────────────────────────────────────────────────────

def test_text_only_is_routed_by_text_model_not_defaulted_to_chest(client, service):
    logits = [-5.0] * 25
    logits[MELANOMA] = 5.0
    set_logits(service.model.text_head, logits)

    body = client.post("/predict", data={"symptoms": "a mole with irregular borders"}).json()

    assert body["mode"] == "text_only"
    assert body["modality"] == "skin"
    assert body["modality_source"] == "text"
    assert body["predictions"][0]["disease"] == "Melanoma"
    assert label_indices(body) <= set(SKIN)


def test_explicit_modality_overrides_detection(client):
    body = client.post("/predict", data={"symptoms": "headache and seizures", "modality": "brain"}).json()
    assert body["modality"] == "brain"
    assert body["modality_source"] == "user"
    assert label_indices(body) <= set(BRAIN)


def test_image_modality_comes_from_classifier(client):
    body = client.post("/predict", files=image_file()).json()
    assert body["mode"] == "image_only"
    assert body["modality"] == "chest"
    assert body["modality_source"] == "classifier"
    assert body["modality_confidence"] > 0.9
    assert label_indices(body) <= set(CHEST)


def test_low_modality_confidence_is_rejected_unless_forced(client, service):
    set_logits(service.modality_clf.classifier, [0.0, 0.0, 0.0])  # 1/3 each

    rejected = client.post("/predict", files=image_file())
    assert rejected.status_code == 422
    assert "does not look like" in rejected.json()["detail"]

    forced = client.post("/predict", files=image_file(), data={"modality": "skin"})
    assert forced.status_code == 200
    assert forced.json()["modality"] == "skin"


# ── Explanations ─────────────────────────────────────────────────────────────

def test_image_prediction_has_heatmap_for_top_class(client):
    body = client.post("/predict", files=image_file(size=(80, 60))).json()
    explanation = body["explanation"]

    assert explanation["heatmap_class"] == body["predictions"][0]["disease"]
    heatmap = Image.open(io.BytesIO(base64.b64decode(explanation["heatmap_base64"])))
    assert heatmap.size == (80, 60)  # overlay is at the original resolution
    assert body["warnings"] == []


def test_multimodal_returns_heatmap_and_token_importance(client):
    body = client.post(
        "/predict-multimodal",
        files=image_file(),
        data={"symptoms": "a patient with fever and dyspnea"},
    ).json()

    assert body["mode"] == "multimodal"
    explanation = body["explanation"]
    assert explanation["heatmap_base64"]
    tokens = [t["token"] for t in explanation["important_tokens"]]
    assert "dyspnea" in tokens  # WordPiece pieces merged back into the word
    assert "**" in explanation["highlighted_text"]
    assert body["inference_time_ms"] > 0


# ── Thresholds and warnings from training metadata ───────────────────────────

def test_thresholds_from_training_metadata(tiny_cfg):
    metadata = {"thresholds": {"text_only": [0.0] * 25}}
    svc = build_service(tiny_cfg, metadata=metadata)
    with TestClient(create_app(service=svc, cfg=tiny_cfg)) as client:
        text = client.post("/predict", data={"symptoms": "fever"}).json()
        image = client.post("/predict", files=image_file()).json()

    assert all(p["threshold"] == 0.0 and p["positive"] for p in text["predictions"])
    assert all(p["threshold"] == tiny_cfg.confidence_threshold for p in image["predictions"])


def test_warns_when_checkpoint_scored_badly_in_mode(tiny_cfg):
    metadata = {"training_results": {"image_only": {"auc": 0.46}, "multimodal": {"auc": 0.99}}}
    svc = build_service(tiny_cfg, metadata=metadata)
    with TestClient(create_app(service=svc, cfg=tiny_cfg)) as client:
        image_only = client.post("/predict", files=image_file()).json()
        multimodal = client.post("/predict", files=image_file(), data={"symptoms": "fever"}).json()

    assert any("AUC 0.46" in w for w in image_only["warnings"])
    assert multimodal["warnings"] == []


# ── Input validation ─────────────────────────────────────────────────────────

def test_missing_inputs_is_400(client):
    assert client.post("/predict").status_code == 400
    assert client.post("/predict", data={"symptoms": "   "}).status_code == 400


def test_unknown_modality_is_400(client):
    response = client.post("/predict", data={"symptoms": "fever", "modality": "kidney"})
    assert response.status_code == 400


def test_non_image_upload_is_400(client):
    response = client.post("/predict", files={"image": ("notes.txt", b"not an image", "text/plain")})
    assert response.status_code == 400
    assert "image" in response.json()["detail"]


def test_multimodal_endpoint_requires_text(client):
    response = client.post("/predict-multimodal", files=image_file(), data={"symptoms": "  "})
    assert response.status_code == 400


def test_upload_size_limit(tiny_cfg):
    tiny_cfg.raw["api"]["max_upload_size_mb"] = 1
    svc = build_service(tiny_cfg)
    with TestClient(create_app(service=svc, cfg=tiny_cfg)) as client:
        big = {"image": ("big.png", b"0" * (1024 * 1024 + 1), "image/png")}
        assert client.post("/predict", files=big).status_code == 413
