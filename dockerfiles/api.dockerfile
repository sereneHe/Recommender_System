# Use Astral UV base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

EXPOSE $PORT

# Install build tools (needed for some ML libraries)
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the project name as a build argument or env var
ARG PROJECT_NAME=mlo_group_project
ENV PROJECT_NAME=${PROJECT_NAME}

# Set working directory
WORKDIR /app

# Copy only the requirements file
COPY dockerfiles/api_requirements.txt ./requirements.txt

# Use uv for faster installation of dependencies
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    -r requirements.txt

# Copy metadata first (helps with caching)
COPY pyproject.toml README.md* ./
# Copy source files
COPY src/${PROJECT_NAME}/__init__.py ./src/${PROJECT_NAME}/__init__.py
COPY src/${PROJECT_NAME}/api.py ./src/${PROJECT_NAME}/api.py
COPY src/${PROJECT_NAME}/guardrails.py ./src/${PROJECT_NAME}/guardrails.py
COPY src/${PROJECT_NAME}/model.py ./src/${PROJECT_NAME}/model.py
# Copy model
COPY models/model.pth ./models/model.pth
# Copy processed data files
COPY data/processed/scaler.joblib ./data/processed/scaler.joblib
COPY data/processed/feature_columns.json ./data/processed/feature_columns.json
COPY data/processed/label_encoder.joblib ./data/processed/label_encoder.joblib

# Ensure the package directory exists
RUN mkdir -p src/${PROJECT_NAME} && touch src/${PROJECT_NAME}/__init__.py
# Install the project in editable mode
ENTRYPOINT ["uv", "run", "uvicorn"]
CMD ["mlo_group_project.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
