#!/usr/bin/env python3
"""Create an auditable, validation-only schema-v2 configuration freeze receipt.

This command intentionally does not select a winner and does not accept any
holdout metric.  The researcher supplies the selected config hash AFTER the
validation report is closed and a human review has accepted it.  The resulting
receipt is the only input a future F0 locked-holdout run may accept.

Schema v2 adds the fields a locked holdout needs to be reproducible and
non-gameable:

* ``protocol_version``        - registry version the decision was made under;
* ``selection_rule``          - the pre-declared rule that picked the arm;
* ``candidate_set``           - every arm that was eligible;
* ``selected_config_hash``    - the frozen winner;
* ``validation_report_hash``  - hash of the closed validation artifact;
* ``code_commit``             - git commit of the selection;
* ``reserved_holdout_seed_table`` + hash - the untouched holdout seeds;
* ``registry_hash``           - hash of experiment_registry.yaml at freeze time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "experiment_registry.yaml"


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt_hash(payload: dict) -> str:
    core = {k: v for k, v in payload.items() if k != "receipt_hash"}
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--scope", required=True, help="ER, SF, FRED, or CoDiet")
    parser.add_argument("--selected-config-hash", required=True)
    parser.add_argument("--candidate-config-hashes", nargs="+", required=True)
    parser.add_argument("--validation-artifact", required=True,
                        help="Path to the closed validation summary; no holdout metrics allowed")
    parser.add_argument("--selection-metric", required=True)
    parser.add_argument("--selection-rule", required=True)
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    validation = (root / args.validation_artifact).resolve()
    if not validation.exists():
        raise SystemExit(f"validation artifact does not exist: {validation}")
    selected = str(args.selected_config_hash)
    candidates = [str(x) for x in args.candidate_config_hashes]
    if selected not in candidates:
        raise SystemExit("selected config hash must be one of candidate hashes")
    if any(token.lower() in validation.name.lower() for token in ("holdout", "test")):
        raise SystemExit("validation artifact name suggests a holdout/test result; refusing to freeze")

    registry_path = Path(args.registry)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    f0 = (registry.get("hypotheses") or {}).get("F0") or {}
    reserved = f0.get("reserved_holdout_seed_table")
    if not reserved:
        raise SystemExit("registry has no F0.reserved_holdout_seed_table; cannot bind holdout seeds")
    sys.path.insert(0, str(root / "scripts"))
    from experiment_registry import seed_table_hash

    payload = {
        "schema_version": 2,
        "scope": str(args.scope),
        "protocol_version": registry.get("registry_version"),
        "selected_config_hash": selected,
        "candidate_set": candidates,
        "validation_artifact": str(validation.relative_to(root)),
        "validation_report_hash": _sha256(validation),
        "selection_metric": str(args.selection_metric),
        "selection_rule": str(args.selection_rule),
        "code_commit": git_commit(root),
        "reserved_holdout_seed_table": reserved,
        "reserved_holdout_seed_table_hash": seed_table_hash(reserved),
        "registry_hash": _sha256(registry_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_metrics_present": False,
    }
    payload["receipt_hash"] = receipt_hash(payload)
    out = (root / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"wrote frozen selection receipt: {out}")
    print(f"receipt_hash={payload['receipt_hash']}")


if __name__ == "__main__":
    main()
