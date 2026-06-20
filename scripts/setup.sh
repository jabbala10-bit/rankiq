#!/usr/bin/env bash
# First-time setup helper for RankIQ.
# Usage: bash scripts/setup.sh
set -euo pipefail

echo "=== RankIQ setup ==="

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.11+ first." >&2
    exit 1
fi

echo "-> Installing Python dependencies..."
pip install -r requirements.txt

if [ ! -f .env ]; then
    echo "-> Creating .env from .env.example..."
    cp .env.example .env
else
    echo "-> .env already exists, leaving it untouched."
fi

echo "-> Creating local data directories..."
mkdir -p data/catalog data/indexes/faiss

echo ""
echo "Setup complete. Default VECTOR_BACKEND=faiss needs no extra service."
echo "Next steps:"
echo "  1. Review and edit .env"
echo "  2. make run-api          # starts FastAPI on :8000"
echo "  3. make index-sample     # indexes the sample catalog"
echo "  4. make run-ui           # starts the search UI on :7860"
echo ""
echo "To try a different vector backend (ADR-001):"
echo "  make docker-up-qdrant    # then set VECTOR_BACKEND=qdrant in .env"
echo "  make docker-up-pgvector  # then set VECTOR_BACKEND=pgvector in .env"
