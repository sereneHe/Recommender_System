# Use Astral UV base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Expose port (set via env at runtime if needed)
EXPOSE ${PORT:-8000}

# Install build tools (needed for some ML libraries)
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the project name as a build argument or env var
ARG PROJECT_NAME=hei_project
ENV PROJECT_NAME=${PROJECT_NAME}

# Set working directory
WORKDIR /app

# -------------------------
# Dependencies
# -------------------------
# Copy only the API requirements file
COPY dockerfiles/api_requirements.txt ./requirements.txt

# Use uv for faster installation of dependencies
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    -r requirements.txt

# -------------------------
# Project metadata
# -------------------------
COPY pyproject.toml README.md* ./

# -------------------------
# Source code
# -------------------------
# Ensure package directory exists
RUN mkdir -p src/${PROJECT_NAME}

# Copy source files
COPY src/${PROJECT_NAME}/__init__.py ./src/${PROJECT_NAME}/__init__.py
COPY src/${PROJECT_NAME}/api.py ./src/${PROJECT_NAME}/api.py
COPY src/${PROJECT_NAME}/guardrails.py ./src/${PROJECT_NAME}/guardrails.py
COPY src/${PROJECT_NAME}/model.py ./src/${PROJECT_NAME}/model.py

# -------------------------
# Model & preprocessing artifacts
# -------------------------
COPY models/model.pth ./models/model.pth
COPY data/processed/scaler.joblib ./data/processed/scaler.joblib
COPY data/processed/feature_columns.json ./data/processed/feature_columns.json
COPY data/processed/label_encoder.joblib ./data/processed/label_encoder.joblib

# -------------------------
# Runtime environment
# -------------------------
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# -------------------------
# Entrypoint: uv script
# -------------------------
# This assumes pyproject.toml contains:
# [project.scripts]
# api = "hei_project.api:main"
ENTRYPOINT ["uv", "run", "api"]
