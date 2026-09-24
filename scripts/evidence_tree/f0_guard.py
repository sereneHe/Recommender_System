#!/usr/bin/env python3
"""Guard the F0 locked-holdout gate.

F0 is the final, confirmatory evaluation of a frozen configuration.  It may run
ONLY when:

* a schema-v2 frozen-selection receipt exists, is hash-intact, was produced from
  a validation-only artifact, and its bound reserved-holdout seed table and
  registry hash match the current ``experiment_registry.yaml``;
* every graph/noise seed the runner uses lies inside that reserved table;
* the runner is a plain executable file path (never an arbitrary shell string).

Anything else exits non-zero.  This module is importable so the rules can be
unit-tested without a cluster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "experiment_registry.yaml"
# Allow ``import experiment_registry`` regardless of the caller's cwd.
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

_INTERPRETER_BASENAMES = {
    "bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh",
    "python", "python3", "perl", "ruby", "node", "env", "eval", "exec",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def receipt_hash(payload: dict) -> str:
    core = {k: v for k, v in payload.items() if k != "receipt_hash"}
    return _sha256_bytes(json.dumps(core, sort_keys=True, separators=(",", ":")).encode())


def validate_receipt(receipt: dict, registry: dict, registry_path: Path) -> list[str]:
    problems: list[str] = []
    if receipt.get("schema_version") != 2:
        problems.append("schema_version must be 2")
    if receipt.get("holdout_metrics_present") is not False:
        problems.append("receipt must be validation-only (holdout_metrics_present=false)")
    selected = receipt.get("selected_config_hash")
    candidates = receipt.get("candidate_set") or receipt.get("candidate_config_hashes") or []
    if selected not in candidates:
        problems.append("selected_config_hash must be one of candidate_set")
    table = receipt.get("reserved_holdout_seed_table")
    if not table:
        problems.append("reserved_holdout_seed_table is missing")
    else:
        try:
            from experiment_registry import seed_table_hash
            claimed = receipt.get("reserved_holdout_seed_table_hash")
            if claimed != seed_table_hash(table):
                problems.append("reserved_holdout_seed_table_hash does not match the table")
            f0 = (registry.get("hypotheses") or {}).get("F0") or {}
            if f0.get("reserved_holdout_seed_table") and table != f0["reserved_holdout_seed_table"]:
                problems.append("receipt reserved seed table differs from the registry F0 table")
        except Exception as exc:  # pragma: no cover - defensive
            problems.append(f"could not verify reserved seed table: {exc}")
    if receipt.get("registry_hash") and registry_path.exists():
        if receipt["registry_hash"] != file_hash(registry_path):
            problems.append("registry_hash does not match the current experiment_registry.yaml")
    elif registry_path.exists() and not receipt.get("registry_hash"):
        problems.append("registry_hash is missing from the receipt")
    claimed_hash = receipt.get("receipt_hash")
    if claimed_hash != receipt_hash(receipt):
        problems.append("receipt_hash mismatch")
    return problems


def check_seed_table(seed_table: dict, reserved: dict) -> list[str]:
    """The holdout run must use the full reserved table, exactly once."""
    from experiment_registry import normalize_seed_table, seed_combinations

    used = set(seed_combinations(seed_table))
    allowed = set(seed_combinations(reserved))
    outside = sorted(used - allowed)
    missing = sorted(allowed - used)
    if outside or missing:
        problems = []
        if outside:
            problems.append(f"seeds outside the reserved holdout table: {outside}")
        if missing:
            problems.append(f"reserved holdout seeds not included: {missing}")
        return problems
    return []


def check_command(command: str) -> list[str]:
    """A runner must be a plain executable path, never a shell pipeline.

    Interpreters and shells are rejected outright: ``/bin/bash -c '...'`` is an
    arbitrary command in disguise, not a reviewed holdout runner.
    """
    if not command:
        return ["LOCKED_HOLDOUT_COMMAND is empty"]
    if any(tok in command for tok in (";", "&", "|", "`", "$", ">", "<", "(", ")", "\n")):
        return ["LOCKED_HOLDOUT_COMMAND must be a plain script path, not a shell pipeline"]
    first = command.split()[0]
    p = Path(first)
    if not p.is_absolute():
        p = ROOT / p
    if p.name in _INTERPRETER_BASENAMES:
        return [f"runner {first!r} is an interpreter/shell, not a reviewed holdout runner"]
    if not p.is_file():
        return [f"runner {first!r} is not an executable file"]
    if not (p.stat().st_mode & 0o111):
        return [f"runner {first!r} is not executable"]
    return []


def _load_registry(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rc = sub.add_parser("check-receipts")
    p_rc.add_argument("--dir", required=True)
    p_rc.add_argument("--registry", default=str(REGISTRY_PATH))

    p_r = sub.add_parser("check-receipt")
    p_r.add_argument("--receipt", required=True)
    p_r.add_argument("--registry", default=str(REGISTRY_PATH))

    p_s = sub.add_parser("check-seeds")
    p_s.add_argument("--graph", required=True, help="space-separated graph seeds")
    p_s.add_argument("--noise", required=True, help="space-separated noise seeds")
    p_s.add_argument("--registry", default=str(REGISTRY_PATH))

    p_c = sub.add_parser("check-command")
    p_c.add_argument("--command", required=True)

    args = parser.parse_args(argv)
    registry_path = Path(getattr(args, "registry", REGISTRY_PATH))
    registry = _load_registry(registry_path)

    if args.cmd in ("check-receipts", "check-receipt"):
        if args.cmd == "check-receipts":
            d = Path(args.dir)
            receipts = sorted(d.glob("*.yaml")) if d.is_dir() else []
            if not receipts:
                print(f"F0 blocked: no frozen selection receipt in {d}", file=sys.stderr)
                return 2
        else:
            receipts = [Path(args.receipt)]
        problems: list[str] = []
        for rp in receipts:
            try:
                receipt = yaml.safe_load(rp.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                problems.append(f"{rp.name}: unreadable ({exc})")
                continue
            problems.extend(f"{rp.name}: {p}" for p in validate_receipt(receipt, registry, registry_path))
        if problems:
            for p in problems:
                print(f"F0 blocked: {p}", file=sys.stderr)
            return 2
        print(f"frozen selection receipt(s) valid: {[r.name for r in receipts]}")
        return 0

    if args.cmd == "check-seeds":
        f0 = (registry.get("hypotheses") or {}).get("F0") or {}
        reserved = f0.get("reserved_holdout_seed_table")
        if not reserved:
            print("F0 blocked: registry has no reserved_holdout_seed_table", file=sys.stderr)
            return 2
        used = {
            "graph_seeds": [int(x) for x in args.graph.split()],
            "noise_seeds": [int(x) for x in args.noise.split()],
        }
        problems = check_seed_table(used, reserved)
        if problems:
            for p in problems:
                print(f"F0 blocked: {p}", file=sys.stderr)
            return 2
        print("holdout seeds are inside the reserved table")
        return 0

    if args.cmd == "check-command":
        problems = check_command(args.command)
        if problems:
            for p in problems:
                print(f"F0 blocked: {p}", file=sys.stderr)
            return 2
        print("runner command accepted")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
