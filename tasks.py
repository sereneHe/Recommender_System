import os
import sys
from invoke import task
from invoke.context import Context
import subprocess
from loguru import logger
from pathlib import Path
import tomllib
from typing import cast

WINDOWS = os.name == "nt"
PYTHON_VERSION = "3.12"


def get_project_name() -> str:
    """Reads the project name from pyproject.toml."""
    try:
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        return cast(str, data["project"]["name"])
    except Exception as e:
        logger.warning(f"Could not read project name from pyproject.toml: {e}. Falling back to folder name.")
        return Path.cwd().name


PROJECT_NAME = get_project_name()


# Utility functions
def kill_port(port: int):
    try:
        if sys.platform.startswith("win"):
            cmd = f"for /f \"tokens=5\" %a in ('netstat -aon ^| findstr :{port}') do taskkill /F /PID %a"
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(
                f"lsof -ti:{port} | xargs kill -9",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def train_profile(ctx: Context) -> None:
    """Profile training."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py --profile", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py", echo=True, pty=not WINDOWS)


@task
def visualize(ctx: Context) -> None:
    """Visualize model results."""
    ctx.run(f"uv run src/{PROJECT_NAME}/visualize.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", env={"PYTHONPATH": "."}, pty=not WINDOWS, echo=True)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


@task
def serve_api(ctx: Context, port: int = 8000) -> None:
    """Serve FastAPI backend (opens in a new terminal on Windows)."""

    kill_port(port)

    cmd = f"uv run uvicorn {PROJECT_NAME}.api:app --host 127.0.0.1 --port {port} --reload"

    if WINDOWS:
        full = f"{cmd} & echo. & echo API process exited. & pause"
        create_new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(
            ["cmd.exe", "/k", full],
            creationflags=create_new_console,
        )
    else:
        ctx.run(cmd, echo=True, pty=True)


@task
def serve_ui(ctx: Context, port: int = 8501) -> None:
    """Serve Streamlit UI (opens in a new terminal on Windows)."""

    kill_port(port)

    cmd = f"uv run streamlit run src/{PROJECT_NAME}/streamlit_app.py --server.port {port}"

    if WINDOWS:
        # /k keeps it open; pause shows errors if the command fails instantly
        full = f"{cmd} & echo. & echo UI process exited. & pause"
        create_new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(
            ["cmd.exe", "/k", full],
            creationflags=create_new_console,
        )
    else:
        ctx.run(cmd, echo=True, pty=True)


@task
def serve_api_ui(ctx, api_port=8000, ui_port=8501):
    """Start API + UI (API in background, UI in foreground)."""

    kill_port(api_port)
    kill_port(ui_port)

    # Use 0.0.0.0 if in Docker, else 127.0.0.1 for local safety
    IS_DOCKER = os.environ.get("IS_DOCKER", "0") == "1"
    logger.debug(f"IS_DOCKER={IS_DOCKER}")
    host = "0.0.0.0" if IS_DOCKER else "127.0.0.1"
    logger.debug(f"Using host address: {host}")

    # Updated commands
    api_cmd = f"uv run uvicorn {PROJECT_NAME}.api:app --host 0.0.0.0 --port {api_port}"

    # Streamlit needs address 0.0.0.0 and headless mode for Docker
    ui_cmd = (
        f"uv run streamlit run src/{PROJECT_NAME}/streamlit_app.py "
        f"--server.port {ui_port} "
        f"--server.address 0.0.0.0 --server.headless true"
    )

    if WINDOWS:
        # start both in separate consoles
        create_new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(
            ["cmd.exe", "/k", f"{api_cmd} & echo. & echo API exited. & pause"], creationflags=create_new_console
        )
        subprocess.Popen(
            ["cmd.exe", "/k", f"{ui_cmd} & echo. & echo UI exited. & pause"], creationflags=create_new_console
        )
    else:
        # start API in background
        api_proc = subprocess.Popen(api_cmd, shell=True)
        try:
            # run UI in foreground so Ctrl+C stops everything
            ctx.run(ui_cmd, echo=True, pty=True)
        finally:
            api_proc.terminate()


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def data_pull(ctx):
    """Pull data from GCS remote."""
    logger.debug("Pulling latest artifacts from Google Cloud Storage...")
    ctx.run("dvc pull")


@task
def repro(ctx, force = False):
    """Run the DVC pipeline. Only runs stages if code or data changed."""
    logger.debug("Checking pipeline lineage and reproducing...")
    flag = "-f" if force else ""
    ctx.run(f"dvc repro {flag}", echo=True)
    logger.info(f"Repro force flag on: {force}")
    ctx.run("git add dvc.lock")
    logger.success("Pipeline reproduced. dvc.lock updated.")


@task
def promote(ctx: Context) -> None:
    """Push results to Cloud."""
    logger.debug("Starting Model Promotion to Registry...")
    try:
        ctx.run("dvc commit -f", echo=True)
        ctx.run("dvc push", echo=True)
        ctx.run("git add dvc.lock")
        logger.success("Model promoted! Run 'git commit' and 'git push' to trigger CI evaluation.")
    except Exception as e:
        logger.error(f"Could not promote model: {e}")
        return

