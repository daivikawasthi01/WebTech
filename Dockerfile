# Dockerfile — Neuro-Genetic Maintainability Framework
#
# Two stages:
#   builder — installs Python dependencies into a venv (layer-cacheable)
#   runtime — copies only what's needed; keeps final image lean
#
# Usage:
#   docker build -t neuro-ga-maint .
#
#   # Run the Streamlit dashboard
#   docker run -p 8501:8501 -v $(pwd)/data:/app/data neuro-ga-maint
#
#   # Run the full CLI pipeline
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/test_repos:/app/test_repos \
#     neuro-ga-maint python main.py --repo test_repos/flask --run-all
#
#   # Run the Jupyter notebook server
#   docker run -p 8888:8888 -v $(pwd)/data:/app/data neuro-ga-maint \
#     jupyter notebook --ip=0.0.0.0 --no-browser --allow-root notebook.ipynb

FROM python:3.11-slim AS builder

WORKDIR /app

# System deps: git (for GitPython), build tools for C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a venv for clean copying
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────

FROM python:3.11-slim AS runtime

WORKDIR /app

# git must be present at runtime for GitPython repo operations
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source
COPY src/       ./src/
COPY main.py    .
COPY app.py     .
COPY notebook.ipynb .

# Persistent data directories (mount these as volumes in production)
RUN mkdir -p data/results test_repos

# Streamlit config — disable telemetry and set server defaults
RUN mkdir -p /root/.streamlit
COPY streamlit_config.toml /root/.streamlit/config.toml

# Default: launch Streamlit dashboard
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]