# Use Astral UV base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

EXPOSE $PORT

# Install build tools (needed for some ML libraries)
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the project name as a build argument or env var
ARG PROJECT_NAME=mlo_group_project
ENV PROJECT_NAME=${PROJECT_NAME}

# Set working directory
WORKDIR /app

# Copy only the requirements file
COPY dockerfiles/streamlit_app_requirements.txt ./requirements.txt

# Use uv for faster installation of dependencies
RUN uv pip install --system --no-cache \
    --index-strategy unsafe-best-match \
    -r requirements.txt

# Copy metadata first (helps with caching)
COPY pyproject.toml README.md* ./
# Copy source files
COPY src/${PROJECT_NAME}/__init__.py ./src/${PROJECT_NAME}/__init__.py
COPY src/${PROJECT_NAME}/streamlit_app.py ./src/${PROJECT_NAME}/streamlit_app.py
COPY src/${PROJECT_NAME}/styles/ ./src/${PROJECT_NAME}/styles/

# Ensure the package directory exists
RUN mkdir -p src/${PROJECT_NAME} && touch src/${PROJECT_NAME}/__init__.py
# Install the project in editable mode

ENTRYPOINT streamlit run src/${PROJECT_NAME}/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
