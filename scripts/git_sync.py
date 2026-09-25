#!/usr/bin/env python3
"""Manifest-scoped git sync with preview, allowlist, guards and receipt.

Unlike ``git add -A``, this stages ONLY the paths listed in an explicit manifest
and always refuses data / runtime artifacts.  It is the single implementation
behind BOTH dashboards' "sync to git" buttons.

Guards
------
* a path is uploaded only if it is listed in the manifest AND not forbidden;
* sync refuses to commit if the index already holds staged files OUTSIDE the
  manifest (they would otherwise be swept into this commit);
* the receipt is written to a separate audit directory, never into the code
  commit, so "method" commits stay clean.

CLI
---
    python scripts/git_sync.py preview --manifest M [--target method]
    python scripts/git_sync.py status  --manifest M
    python scripts/git_sync.py sync    --manifest M [--target method] [--message ...]
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE = "github"
DEFAULT_BRANCH = "current-experiments"
DEFAULT_RECEIPT_DIR = DEFAULT_ROOT / "reports" / "audit" / "git_sync"
VALID_TARGETS = ("method", "dashboard")
# Never upload these, even if a manifest accidentally lists them.
FORBIDDEN = (
    "data/", "metacentrum_runs/", "mlruns/", "multirun/", "results/",
    "outputs/", "output/", "tmp/", "__pycache__/", ".git/",
    "reports/", "_quarantine_",
)
FORBIDDEN_GLOBS = ("*.pkl", "*.o[0-9]*", "*.e[0-9]*", "*.log", ".DS_Store",
                   "*.sqlite3", "*.sqlite3-wal", "*.sqlite3-shm", "*.bak.*",
                   "*.jsonl", "git_sync_receipt.json")


def _git(root: Path, args: list[str], timeout: int = 120):
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True, timeout=timeout)


def load_manifest(path: Path) -> list[str]:
    specs: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            specs.append(line)
    if not specs:
        raise ValueError(f"manifest is empty: {path}")
    return specs


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_forbidden(path: str) -> bool:
    if any(path.startswith(prefix) or f"/{prefix}" in path for prefix in FORBIDDEN):
        return True
    return any(fnmatch.fnmatch(Path(path).name, g) or fnmatch.fnmatch(path, g)
               for g in FORBIDDEN_GLOBS)


def changed_under(root: Path, specs: list[str]) -> list[str]:
    """Files (tracked or untracked) with pending changes under the given paths."""
    result = _git(root, ["status", "--porcelain", "-uall", "--", *specs])
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()}")
    files: list[str] = []
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        entry = line[3:].strip()
        if " -> " in entry:          # rename: take the new path
            entry = entry.split(" -> ", 1)[1]
        files.append(entry.strip('"'))
    return files


def staged_outside_manifest(root: Path, specs: list[str]) -> list[str]:
    """Already-staged files that the manifest does NOT allow."""
    total = [l.strip() for l in _git(root, ["diff", "--cached", "--name-only"]).stdout.splitlines() if l.strip()]
    within = set(l.strip() for l in _git(root, ["diff", "--cached", "--name-only", "--", *specs]).stdout.splitlines() if l.strip())
    return sorted(p for p in total if p not in within)


def remote_commit(root: Path, remote: str, branch: str) -> str | None:
    result = _git(root, ["ls-remote", remote, f"refs/heads/{branch}"], timeout=60)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.split()[0]


def local_commit(root: Path) -> str:
    return _git(root, ["rev-parse", "HEAD"]).stdout.strip()


def preview(root: Path, manifest: Path, remote: str, branch: str) -> dict:
    specs = load_manifest(manifest)
    files = changed_under(root, specs)
    upload = sorted(p for p in files if not _is_forbidden(p))
    excluded = sorted(p for p in files if _is_forbidden(p))
    everything = set(changed_under(root, []))
    skipped = sorted(everything - set(files))
    outside = staged_outside_manifest(root, specs)
    return {
        "root": str(root),
        "manifest": str(manifest),
        "manifest_hash": manifest_hash(manifest),
        "remote": remote,
        "branch": branch,
        "local_commit": local_commit(root),
        "remote_commit": remote_commit(root, remote, branch),
        "dirty": bool(everything),
        "upload_files": upload,
        "excluded_files": excluded,
        "skipped_files": skipped,
        "staged_outside_manifest": outside,
        "upload_count": len(upload),
        "skipped_count": len(skipped),
    }


def sync(root: Path, manifest: Path, remote: str, branch: str,
         message: str | None, receipt_dir: Path | None = None,
         target: str = "method") -> tuple[int, dict]:
    target = str(target).strip().lower()
    if target not in VALID_TARGETS:
        return 2, {"status": "invalid_target", "target": target,
                   "error": f"target must be one of {VALID_TARGETS}"}
    info = preview(root, manifest, remote, branch)
    info["target"] = target
    info["repository"] = str(root)
    info["pre_sync_commit"] = info["local_commit"]
    info["remote_commit_before"] = info["remote_commit"]
    if info["staged_outside_manifest"]:
        info["status"] = "refused_staged_outside_manifest"
        _write_receipt(receipt_dir, target, info)
        return 1, info
    if not info["upload_files"]:
        info["status"] = "no_changes"
        info["post_sync_commit"] = info["local_commit"]
        info["remote_commit_after"] = info["remote_commit"]
        _write_receipt(receipt_dir, target, info)
        return 0, info
    add = _git(root, ["add", "--", *info["upload_files"]])
    if add.returncode != 0:
        info["status"] = "add_failed"
        info["error"] = add.stderr.strip()
        _write_receipt(receipt_dir, target, info)
        return 1, info
    msg = message or f"sync({manifest.name}/{target}) {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    commit = _git(root, ["commit", "-m", msg])
    info["commit_output"] = (commit.stdout + commit.stderr).strip()
    if commit.returncode != 0:
        info["status"] = "commit_failed"
        _write_receipt(receipt_dir, target, info)
        return 1, info
    push = _git(root, ["push", remote, f"HEAD:{branch}"], timeout=300)
    info["push_output"] = (push.stdout + push.stderr).strip()
    info["status"] = "pushed" if push.returncode == 0 else "push_failed"
    info["post_sync_commit"] = local_commit(root)
    info["remote_commit_after"] = remote_commit(root, remote, branch)
    _write_receipt(receipt_dir, target, info)
    return (0 if push.returncode == 0 else 1), info


def _write_receipt(receipt_dir: Path | None, target: str, info: dict) -> None:
    if receipt_dir is None:
        return
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "target": target,
        "repository": info.get("repository") or info.get("root"),
        "branch": info.get("branch"),
        "manifest": info.get("manifest"),
        "manifest_hash": info.get("manifest_hash"),
        "pre_sync_commit": info.get("pre_sync_commit"),
        "post_sync_commit": info.get("post_sync_commit"),
        "remote_commit_before": info.get("remote_commit_before"),
        "remote_commit_after": info.get("remote_commit_after"),
        "uploaded": info.get("upload_files", []),
        "excluded": info.get("excluded_files", []),
        "skipped_outside_manifest": info.get("skipped_files", []),
        "staged_outside_manifest": info.get("staged_outside_manifest", []),
        "status": info.get("status"),
        "synced_at_utc": now.isoformat(),
    }
    receipt_dir = Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    (receipt_dir / f"{stamp}-{target}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (receipt_dir / f"latest-{target}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (receipt_dir / "log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("preview", "status", "sync"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--target", default="method", choices=list(VALID_TARGETS))
        if name == "sync":
            p.add_argument("--message", default=None)
            p.add_argument("--receipt-dir", default=str(DEFAULT_RECEIPT_DIR))
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    if args.cmd in ("preview", "status"):
        print(json.dumps(preview(root, manifest, args.remote, args.branch), indent=2, sort_keys=True))
        return 0
    code, info = sync(root, manifest, args.remote, args.branch, args.message,
                      Path(args.receipt_dir), args.target)
    print(json.dumps(info, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
