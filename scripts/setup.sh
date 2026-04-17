#!/usr/bin/env bash
# setup.sh — One-command project setup
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Multi-Modal Disease Detection — Setup ==="
echo "Project root: $PROJECT_ROOT"

# ── Python backend ────────────────────────────────────────────────
echo ""
echo "--- Setting up Python backend ---"
cd "$PROJECT_ROOT"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r api/requirements.txt -q

echo "Python backend ready."

# ── Frontend ──────────────────────────────────────────────────────
echo ""
echo "--- Setting up React frontend ---"
cd "$PROJECT_ROOT/frontend"

if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
else
    echo "node_modules already exists, skipping."
fi

echo "Frontend ready."

# ── Model directory ───────────────────────────────────────────────
echo ""
echo "--- Checking model directory ---"
cd "$PROJECT_ROOT"
mkdir -p models

MODEL_FILES=(fusion_model.pth image_encoder.pth text_encoder.pth modality_classifier.pth)
MISSING=0
for f in "${MODEL_FILES[@]}"; do
    if [ ! -f "models/$f" ]; then
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo "WARNING: $MISSING model weight file(s) not found in models/"
    echo "  Run: python scripts/download_models.py --kaggle-dataset <your-dataset>"
    echo "  Or:  python scripts/download_models.py --local-dir /path/to/weights"
else
    echo "All model weights found."
fi

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start the API:     cd $PROJECT_ROOT && python -m uvicorn api.main:app --reload"
echo "To start the frontend: cd $PROJECT_ROOT/frontend && npm run dev"
