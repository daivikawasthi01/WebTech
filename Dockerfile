# Dockerfile — Unified Framework Backend & Research Environment
#
# This image supports:
#   1. FastAPI Backend (api/main.py)
#   2. Research CLI tools (main.py, scripts/research.sh)
#
# Usage:
#   docker build -t webtech-backend .
#
#   # Run as a background service (API)
#   docker run -p 8000:8000 webtech-backend
#
#   # Run a research experiment
#   docker run --rm -v $(pwd)/data:/app/data webtech-backend ./scripts/research.sh --replicate-paper

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for build-time (gcc/g++ for native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Setup virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies (layered for better caching)
COPY requirements.txt .
COPY requirements_api.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_api.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# git and libgomp1 are required at runtime for mining and XGBoost/Torch stability
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Native Stability Fixes ───────────────────────────────────────────────────
# Prevents segmentation faults between PyTorch and XGBoost library conflicts.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Copy source code and scripts
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY main.py ./

# Ensure research script is executable
RUN chmod +x scripts/research.sh

# Persistent data directories
RUN mkdir -p data/results test_repos

# Exports
EXPOSE 8000

# Default: Launch FastAPI Backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]