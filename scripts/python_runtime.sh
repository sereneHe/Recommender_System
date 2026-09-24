#!/bin/sh
# Resolve the only supported Python runtime for project scripts.
#
# Priority deliberately mirrors cluster_computing/run_metacentrum.pbs:
#   1. an explicit executable PYTHON_BIN;
#   2. the MetaCentrum Linux environment;
#   3. a project-local virtual environment (macOS/local development).
#
# Do not silently fall back to the host's python3.  On MetaCentrum that
# interpreter lacks pandas/PyYAML, so a fallback either fails halfway through
# an experiment or, worse, refreshes only part of the evidence dashboard.

if [ -n "${_RECOMMENDER_PYTHON_RUNTIME_LOADED:-}" ]; then
  return 0
fi
_RECOMMENDER_PYTHON_RUNTIME_LOADED=1

project_python_require() {
  repo_root="${1:?project_python_require needs the repository root}"
  explicit_python="${PYTHON_BIN:-}"

  # A bare python/python3 is commonly a legacy default, not an intentional
  # environment selection.  Treat it as unsafe unless the caller explicitly
  # opts in below.
  case "${explicit_python}" in
    ""|python|python3) explicit_python="" ;;
  esac

  for candidate in \
    "${explicit_python}" \
    "/storage/praha1/home/hexiaoyu/codiet311_linux/bin/python" \
    "${repo_root}/codiet311/bin/python" \
    "${repo_root}/.venv/bin/python"; do
    [ -n "${candidate}" ] && [ -x "${candidate}" ] || continue
    if "${candidate}" -c 'import pandas, yaml' >/dev/null 2>&1; then
      export PYTHON_BIN="${candidate}"
      export PROJECT_PYTHON="${candidate}"
      return 0
    fi
  done

  if [ "${ALLOW_SYSTEM_PYTHON:-0}" = "1" ]; then
    candidate="$(command -v python3 2>/dev/null || true)"
    if [ -n "${candidate}" ] && "${candidate}" -c 'import pandas, yaml' >/dev/null 2>&1; then
      export PYTHON_BIN="${candidate}"
      export PROJECT_PYTHON="${candidate}"
      return 0
    fi
  fi

  cat >&2 <<EOF
ERROR: No supported project Python runtime is available.
Expected pandas and PyYAML in one of:
  - an explicit executable PYTHON_BIN
  - /storage/praha1/home/hexiaoyu/codiet311_linux/bin/python (PBS Linux venv)
  - ${repo_root}/codiet311/bin/python or ${repo_root}/.venv/bin/python
The system python3 is intentionally not used.  Install/activate a project
environment, or set PYTHON_BIN to its executable.
EOF
  return 127
}
