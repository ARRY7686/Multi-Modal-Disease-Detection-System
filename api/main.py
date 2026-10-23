"""
Multi-Modal Disease Detection — FastAPI Application

Thin HTTP layer over api.service.InferenceService, which loads the trained
models and does the actual prediction and explanation work.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.config import PROJECT_ROOT, Config
from api.schemas import HealthResponse, LabelResponse, ModelInfoResponse, PredictionResponse
from api.service import InferenceService, InvalidInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _read_upload(upload: Optional[UploadFile], max_bytes: int) -> Optional[bytes]:
    if upload is None:
        return None
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"Image file too large (limit {max_bytes // (1024 * 1024)} MB).")
    return data


def create_app(
    service: Optional[InferenceService] = None,
    cfg: Optional[Config] = None,
    static_dir: Optional[Path] = None,
) -> FastAPI:
    """
    Build the FastAPI app.

    Args:
        service:    a ready InferenceService (tests pass one built from tiny
                    models); when None, one is loaded from disk at startup.
        cfg:        configuration; defaults to configs/model_config.yaml
        static_dir: built frontend (frontend/dist) to serve at "/"; nothing
                    is served when it has no index.html
    """
    cfg = cfg or Config.get()
    max_upload_bytes = cfg.max_upload_size_mb * 1024 * 1024

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_service = app.state.service is None
        if owns_service:
            app.state.service = InferenceService.from_disk(cfg)
            logger.info("All models loaded. API ready.")
        yield
        if owns_service:
            app.state.service.close()
            app.state.service = None
            logger.info("Shutdown complete.")

    app = FastAPI(
        title="Multi-Modal Disease Detection API",
        version=cfg.raw["project"]["version"],
        lifespan=lifespan,
    )
    app.state.service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(InvalidInput)
    async def invalid_input_handler(request: Request, exc: InvalidInput):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    # Endpoints are plain `def` so FastAPI runs CPU-bound inference in its
    # thread pool instead of blocking the event loop.

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        svc: InferenceService = request.app.state.service
        return HealthResponse(
            status="ok" if svc.healthy else "degraded",
            models_loaded=svc.weights_loaded,
        )

    @app.get("/labels", response_model=LabelResponse)
    def labels():
        return LabelResponse(
            labels=cfg.all_labels,
            modality_groups={
                "chest": cfg.chest_labels,
                "skin": cfg.skin_labels,
                "brain": cfg.brain_labels,
            },
        )

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(request: Request):
        svc: InferenceService = request.app.state.service
        meta = svc.metadata if svc.weights_loaded.get("fusion_model") else {}
        settings = meta.get("settings") or {}
        return ModelInfoResponse(
            weights_loaded=bool(svc.weights_loaded.get("fusion_model")),
            image_backbone=meta.get("image_backbone"),
            text_model_name=meta.get("text_model_name"),
            fusion_type=meta.get("fusion_type"),
            test_results=meta.get("training_results") or {},
            temperatures=meta.get("temperatures") or {},
            modality_classifier_test_accuracy=meta.get("modality_classifier_test_accuracy"),
            evaluation_protocol=meta.get("evaluation_protocol") or {},
            data_counts=meta.get("data_counts") or {},
            training_history=meta.get("training_history") or [],
            chest_train_fraction=settings.get("chest_train_fraction"),
            epochs=settings.get("epochs"),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        request: Request,
        image: Optional[UploadFile] = File(None),
        symptoms: Optional[str] = Form(None),
        modality: Optional[str] = Form(None),
    ):
        """Image only, text only, or both — the mode follows the inputs given."""
        return request.app.state.service.predict(
            _read_upload(image, max_upload_bytes), symptoms, modality,
        )

    @app.post("/predict-multimodal", response_model=PredictionResponse)
    def predict_multimodal(
        request: Request,
        image: UploadFile = File(...),
        symptoms: str = Form(...),
        modality: Optional[str] = Form(None),
    ):
        """Cross-attention prediction; requires both an image and symptom text."""
        return request.app.state.service.predict(
            _read_upload(image, max_upload_bytes), symptoms, modality, require_both=True,
        )

    # Single-container deployment (Docker / HF Space): serve the built React
    # app from the same origin. Mounted last so the API routes above win.
    if static_dir is not None and (static_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
        logger.info("Serving frontend from %s", static_dir)

    return app


app = create_app(static_dir=Path(os.environ.get("FRONTEND_DIST", PROJECT_ROOT / "frontend" / "dist")))


if __name__ == "__main__":
    import uvicorn

    _cfg = Config.get()
    uvicorn.run("api.main:app", host=_cfg.api_host, port=_cfg.api_port, reload=False)
