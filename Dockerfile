# Single container: FastAPI serves the API and the built React app on one port.
# Runs as a Hugging Face Docker Space (port 7860, user 1000) or anywhere else:
#   docker build -t mmdd . && docker run -p 7860:7860 mmdd

# ── Frontend ─────────────────────────────────────────────────────────────────
FROM node:20.18.0-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# Empty API base URL: the app calls the API on its own origin.
ENV VITE_API_URL=""
RUN npm run build

# ── API ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt huggingface_hub

COPY --chown=user . .
COPY --from=web --chown=user /web/dist frontend/dist
USER user

# Trained weights come from a Hugging Face model repo at build time
# (scripts/publish_hf.py weights). Leave empty to build without them.
ARG MODEL_REPO=Arry7868/multimodal-disease-detection
RUN if [ -n "$MODEL_REPO" ]; then python scripts/download_models.py --hf-repo "$MODEL_REPO"; fi

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
