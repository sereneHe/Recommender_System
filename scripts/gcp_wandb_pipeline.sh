#!/usr/bin/env bash
set -euo pipefail

# Project defaults
PROJECT_ID="${PROJECT_ID:-recommender-system-hei}"
BUCKET_NAME="${BUCKET_NAME:-serenehe_bucket_1}"
WANDB_ENTITY="${WANDB_ENTITY:-hexiaoyu-czech-technical-university-in-prague}"
WANDB_PROJECT="${WANDB_PROJECT:-Recommender System}"

echo "==> GCP project: ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}" >/dev/null

echo "==> Ensure DVC is initialized"
if [[ ! -d ".dvc" ]]; then
  uv run dvc init
fi

echo "==> Configure DVC remote: gs://${BUCKET_NAME}/dvc"
uv run dvc remote remove gcs_remote >/dev/null 2>&1 || true
uv run dvc remote add -d gcs_remote "gs://${BUCKET_NAME}/dvc"
if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  uv run dvc remote modify gcs_remote credentialpath "${GOOGLE_APPLICATION_CREDENTIALS}"
fi

echo "==> Bucket access check: gsutil ls gs://${BUCKET_NAME}"
set +e
GSUTIL_OUT="$(gsutil ls "gs://${BUCKET_NAME}" 2>&1)"
GSUTIL_RC=$?
set -e
echo "${GSUTIL_OUT}"
if [[ ${GSUTIL_RC} -ne 0 ]]; then
  echo "WARN: bucket list failed (check IAM permissions)."
fi

echo "==> W&B login check"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  uv run wandb login "${WANDB_API_KEY}" --relogin >/dev/null
  echo "W&B login configured for entity=${WANDB_ENTITY}, project=${WANDB_PROJECT}"
else
  echo "WARN: WANDB_API_KEY not set, skip login."
fi

echo "==> Done"
