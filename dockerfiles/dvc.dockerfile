# Use Astral UV base image with Python 3.12 on Debian Bookworm slim
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
# Install build tools
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY ../uv.lock ./uv.lock
COPY ../pyproject.toml ./pyproject.toml
COPY ../README.md ./README.md
COPY ../tasks.py ./tasks.py
COPY ../src ./src
COPY ../data/raw/bcw.csv ./data/raw/bcw.csv
COPY ../.git ./.git
COPY ../.gitignore ./.gitignore

# DVC metadata
COPY ../.dvc ./.dvc
COPY ../dvc.yaml ./dvc.yaml
COPY ../dvc.lock ./dvc.lock
COPY ../*.dvc ./

# Set working directory to root
RUN uv sync --locked --no-cache --no-install-project --all-groups

# Set the entrypoint to run the training script
ENTRYPOINT ["uv", "run", "invoke"]
CMD ["data-pull"]
