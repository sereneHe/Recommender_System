#!/usr/bin/env python3
"""Manifest-scoped git sync with preview, allowlist, guards and receipts.

Unlike ``git add -A``, this stages ONLY the paths listed in an explicit manifest
and always refuses data / runtime artifacts.  It is the single implementation
behind BOTH dashboards' "sync to git" buttons.

Guards
------
* a path is uploaded only if it is listed in the manifest AND not forbidden;
* sync refuses to commit if the index already holds staged files OUTSIDE the
  manifest (they would otherwise be swept into this commit);
* every target takes its own lock, so two processes can never push the same
  target concurrently;
* the receipt is written to a separate audit directory, never into the code
  commit, so "method" commits stay clean.

Receipts (audit trail)
----------------------
Each sync attempt writes ONE immutable receipt named
``<utc-stamp>_<target>_<attempt-id>.json`` under the audit directory.  Existing
receipts are never modified; failed attempts are kept too.  ``latest_<target>.json``
is only a convenience pointer, never the authoritative record.

Receipts record repository-relative paths only (no absolute machine paths, no
tokens/credentials).  A receipt is not proof by itself: use ``verify`` to
re-query ``git ls-remote`` and compare the remote commit and manifest hash.

CLI
---
    python scripts/git_sync.py preview --manifest M [--target method]
    python scripts/git_sync.py status  --manifest M
    python scripts/git_sync.py verify  --manifest M [--target method]
    python scripts/git_sync.py sync    --manifest M [--target method] [--message ...]
                                       [--receipt-dir D] [--audit-branch B]
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:  # POSIX only; the sync tool is used on macOS / Linux
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE = "github"
DEFAULT_BRANCH = "current-experiments"
DEFAULT_RECEIPT_DIR = DEFAULT_ROOT / "reports" / "audit" / "git_sync"
# Remote audit branch.  Defaults to a DEDICATED branch (never a code branch);
# set GIT_SYNC_AUDIT_BRANCH="" to keep receipts local-only.
DEFAULT_AUDIT_BRANCH = os.environ.get("GIT_SYNC_AUDIT_BRANCH", "audit/git-sync") or None
VALID_TARGETS = ("method", "dashboard")
# Terminal statuses written into a receipt.
STATUS_VERIFIED = "verified"
STATUS_PUSH = "push_succeeded"
STATUS_PUSH_FAILED = "push_failed"
STATUS_VERIFY_FAILED = "verification_failed"
STATUS_NO_CHANGES = "no_changes"
STATUS_REFUSED = "refused_staged_outside_manifest"
STATUS_LOCKED = "locked"
STATUS_INVALID_TARGET = "invalid_target"
# Never upload these, even if a manifest accidentally lists them.
FORBIDDEN = (
    "data/", "metacentrum_runs/", "mlruns/", "multirun/", "results/",
    "outputs/", "output/", "tmp/", "__pycache__/", ".git/",
    "reports/", "_quarantine_",
)
FORBIDDEN_GLOBS = ("*.pkl", "*.o[0-9]*", "*.e[0-9]*", "*.log", ".DS_Store",
                   "*.sqlite3", "*.sqlite3-wal", "*.sqlite3-shm", "*.bak.*",
                   "*.jsonl", "git_sync_receipt.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _relative_to(path, root: Path) -> str:
    """Repository-relative path, or just the basename if it lives elsewhere."""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except (ValueError, OSError):
        return Path(path).name


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


# --- receipts -------------------------------------------------------------

@contextlib.contextmanager
def _target_lock(receipt_dir: Path | None, target: str):
    """One lock file per target; yields False when another sync holds it."""
    if receipt_dir is None or fcntl is None:
        yield True
        return
    receipt_dir = Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    handle = (receipt_dir / f".lock-{target}").open("w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True
    finally:
        handle.close()


def _atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _receipt_payload(target: str, info: dict, attempt: dict, status: str,
                     error: str | None = None) -> dict:
    root = Path(info.get("root") or ".")
    manifest = info.get("manifest")
    return {
        "schema_version": 1,
        "attempt_id": attempt["id"],
        "target": target,
        # repository-relative / non-identifying only: no absolute machine paths
        "repository": root.name,
        "remote": info.get("remote"),
        "branch": info.get("branch"),
        "manifest": _relative_to(manifest, root) if manifest else None,
        "manifest_hash": info.get("manifest_hash"),
        "pre_sync_commit": info.get("pre_sync_commit"),
        "created_commit": info.get("post_sync_commit"),
        "post_sync_commit": info.get("post_sync_commit"),
        "remote_commit_before": info.get("remote_commit_before"),
        "remote_commit_after": info.get("remote_commit_after"),
        "uploaded_files": list(info.get("upload_files", [])),
        "excluded_files": list(info.get("excluded_files", [])),
        "skipped_outside_manifest": list(info.get("skipped_files", [])),
        "staged_outside_manifest": list(info.get("staged_outside_manifest", [])),
        "status": status,
        "started_at_utc": attempt["started"],
        "finished_at_utc": _now(),
        "error": error,
    }


def _write_receipt(receipt_dir: Path | None, target: str, info: dict,
                   attempt: dict, status: str, error: str | None = None) -> Path | None:
    """Write one immutable receipt plus a non-authoritative latest pointer."""
    if receipt_dir is None:
        return None
    payload = _receipt_payload(target, info, attempt, status, error)
    receipt_dir = Path(receipt_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt = receipt_dir / f"{stamp}_{target}_{attempt['id']}.json"
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_write(receipt, body)                       # immutable record
    _atomic_write(receipt_dir / f"latest_{target}.json", body)  # convenience
    return receipt


def latest_receipt(receipt_dir: Path | None, target: str) -> dict | None:
    if receipt_dir is None:
        return None
    path = Path(receipt_dir) / f"latest_{target}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def verify(root: Path, manifest: Path, remote: str, branch: str,
           receipt_dir: Path | None, target: str = "method") -> dict:
    """Re-check the latest receipt against the live remote and current manifest.

    A receipt is only trustworthy when BOTH the remote commit and the manifest
    hash still match.  Returns ``status='verified'`` only then.
    """
    receipt = latest_receipt(receipt_dir, target) or {}
    actual = remote_commit(root, remote, branch)
    current_hash = manifest_hash(manifest)
    remote_ok = bool(receipt) and actual is not None and actual == receipt.get("remote_commit_after")
    hash_ok = bool(receipt) and current_hash == receipt.get("manifest_hash")
    verified = remote_ok and hash_ok
    return {
        "target": target,
        "repository": root.name,
        "verified": verified,
        "status": "verified" if verified else "push_succeeded_but_unverified",
        "receipt_status": receipt.get("status"),
        "attempt_id": receipt.get("attempt_id"),
        "actual_remote_commit": actual,
        "receipt_remote_commit_after": receipt.get("remote_commit_after"),
        "current_manifest_hash": current_hash,
        "receipt_manifest_hash": receipt.get("manifest_hash"),
        "checks": {"remote_commit_matches": remote_ok, "manifest_hash_matches": hash_ok},
    }


def publish_receipts(root: Path, receipts: list[Path], remote: str,
                     audit_branch: str, audit_remote: str | None = None,
                     timeout: int = 180) -> dict:
    """Publish receipt files to a DEDICATED audit branch (never a code branch).

    Uses a throwaway clone and an orphan/fetched audit branch, so the target's
    working tree, index and code branches are never touched.
    """
    url = _git(root, ["remote", "get-url", audit_remote or remote]).stdout.strip()
    if not url:
        return {"published": False, "reason": f"unknown remote: {audit_remote or remote}"}
    receipts = [Path(p) for p in receipts]
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        _git(staging, ["init", "-q"])
        _git(staging, ["remote", "add", "audit", url])
        fetched = _git(staging, ["fetch", "-q", "audit", audit_branch])
        if fetched.returncode == 0:
            _git(staging, ["checkout", "-q", "-b", audit_branch, "FETCH_HEAD"])
        else:
            _git(staging, ["checkout", "-q", "--orphan", audit_branch])
        dest = staging / "receipts"
        dest.mkdir(exist_ok=True)
        for item in receipts:
            shutil.copy2(item, dest / item.name)
        _git(staging, ["add", "--", "receipts"])
        if not _git(staging, ["status", "--porcelain"]).stdout.strip():
            return {"published": True, "reason": "no change", "audit_branch": audit_branch}
        _git(staging, ["-c", "user.email=sync@localhost", "-c", "user.name=git_sync",
                       "commit", "-q", "-m", "audit: sync receipts"])
        push = _git(staging, ["push", "audit", f"HEAD:{audit_branch}"], timeout=timeout)
        return {
            "published": push.returncode == 0,
            "audit_remote": audit_remote or remote,
            "audit_branch": audit_branch,
            "output": (push.stdout + push.stderr).strip(),
        }


# --- sync -----------------------------------------------------------------

def sync(root: Path, manifest: Path, remote: str, branch: str,
         message: str | None, receipt_dir: Path | None = None,
         target: str = "method", audit_branch: str | None = None,
         audit_remote: str | None = None) -> tuple[int, dict]:
    target = str(target).strip().lower()
    attempt = {"id": uuid.uuid4().hex[:12], "started": _now()}
    if target not in VALID_TARGETS:
        return 2, {"status": STATUS_INVALID_TARGET, "target": target,
                   "error": f"target must be one of {VALID_TARGETS}"}
    with _target_lock(receipt_dir, target) as locked:
        if not locked:
            return 1, {"status": STATUS_LOCKED, "target": target,
                       "error": "another sync holds this target lock"}
        return _sync_locked(root, manifest, remote, branch, message,
                            receipt_dir, target, attempt,
                            audit_branch if audit_branch is not None else DEFAULT_AUDIT_BRANCH,
                            audit_remote)


def _sync_locked(root, manifest, remote, branch, message, receipt_dir, target,
                 attempt, audit_branch, audit_remote) -> tuple[int, dict]:
    info = preview(root, manifest, remote, branch)
    info["target"] = target
    info["repository"] = root.name
    info["pre_sync_commit"] = info["local_commit"]
    info["remote_commit_before"] = info["remote_commit"]
    info["attempt_id"] = attempt["id"]

    def finish(status: str, code: int, error: str | None = None) -> tuple[int, dict]:
        info["status"] = status
        # Exactly ONE immutable receipt per attempt; never rewritten afterwards.
        receipt = _write_receipt(receipt_dir, target, info, attempt, status, error)
        if receipt and audit_branch:
            info["audit_publish"] = publish_receipts(root, [receipt], remote,
                                                     audit_branch, audit_remote)
        return code, info

    if info["staged_outside_manifest"]:
        return finish(STATUS_REFUSED, 1,
                      "index already holds staged files outside the manifest")

    if not info["upload_files"]:
        info["post_sync_commit"] = info["local_commit"]
        info["remote_commit_after"] = info["remote_commit"]
        return finish(STATUS_NO_CHANGES, 0)

    add = _git(root, ["add", "--", *info["upload_files"]])
    if add.returncode != 0:
        return finish("add_failed", 1, add.stderr.strip())

    msg = message or f"sync({manifest.name}/{target}) {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    commit = _git(root, ["commit", "-m", msg])
    info["commit_output"] = (commit.stdout + commit.stderr).strip()
    if commit.returncode != 0:
        return finish("commit_failed", 1, info["commit_output"])

    info["post_sync_commit"] = local_commit(root)
    push = _git(root, ["push", remote, f"HEAD:{branch}"], timeout=300)
    info["push_output"] = (push.stdout + push.stderr).strip()
    if push.returncode != 0:
        info["remote_commit_after"] = remote_commit(root, remote, branch)
        return finish(STATUS_PUSH_FAILED, 1, info["push_output"])

    actual = remote_commit(root, remote, branch)
    info["remote_commit_after"] = actual
    if actual is None:
        return finish(STATUS_PUSH, 0)                # pushed, could not re-read
    if actual != info["post_sync_commit"]:
        return finish(STATUS_VERIFY_FAILED, 1,
                      f"remote {actual} != local {info['post_sync_commit']}")
    return finish(STATUS_VERIFIED, 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("preview", "status", "verify", "sync"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--target", default="method", choices=list(VALID_TARGETS))
        if name in ("verify", "sync"):
            p.add_argument("--receipt-dir", default=str(DEFAULT_RECEIPT_DIR))
        if name == "sync":
            p.add_argument("--message", default=None)
            p.add_argument("--audit-branch", default=DEFAULT_AUDIT_BRANCH)
            p.add_argument("--audit-remote", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    if args.cmd in ("preview", "status"):
        print(json.dumps(preview(root, manifest, args.remote, args.branch), indent=2, sort_keys=True))
        return 0
    if args.cmd == "verify":
        info = verify(root, manifest, args.remote, args.branch,
                      Path(args.receipt_dir), args.target)
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0 if info["verified"] else 1
    code, info = sync(root, manifest, args.remote, args.branch, args.message,
                      Path(args.receipt_dir), args.target,
                      args.audit_branch, args.audit_remote)
    print(json.dumps(info, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
