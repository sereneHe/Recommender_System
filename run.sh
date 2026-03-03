#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

cd "${ROOT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

if command -v uv >/dev/null 2>&1; then
  uv pip install -e .
else
  "${PYTHON_BIN}" -m pip install -e .
fi

bash "${ROOT_DIR}/scripts/build_prep_data.sh"

PYTHONPATH="${ROOT_DIR}/src" python -m hei_project.train "$@"
