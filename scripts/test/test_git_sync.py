#!/usr/bin/env python3
"""Tests for the manifest-scoped git sync tool (scripts/git_sync.py)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import git_sync  # noqa: E402


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)


class TestForbidden(unittest.TestCase):
    def test_data_and_runtime_are_forbidden(self):
        for p in ("data/energy_expenditure/expenditure.pkl", ".DS_Store",
                  "metacentrum_runs/x/y.yaml", "reports/evidence_ci/a.csv",
                  "scripts/__pycache__/x.pyc", "a/b.log", "model.pkl"):
            self.assertTrue(git_sync._is_forbidden(p), p)

    def test_code_is_allowed(self):
        for p in ("scripts/evidence_tree/gurobi_slot.sh", "hc_predictor.py",
                  "experiment_registry.yaml", "experiments_conf/problem/synthetic_er.yaml"):
            self.assertFalse(git_sync._is_forbidden(p), p)


class TestManifestSync(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.remote = base / "remote.git"
        self.work = base / "work"
        subprocess.run(["git", "init", "--bare", "-q", str(self.remote)], check=True)
        self.work.mkdir()
        _git(self.work, "init", "-q")
        _git(self.work, "config", "user.email", "t@t")
        _git(self.work, "config", "user.name", "t")
        (self.work / "keep.py").write_text("x")
        _git(self.work, "add", "-A")
        _git(self.work, "commit", "-qm", "init")
        _git(self.work, "remote", "add", "github", str(self.remote))
        self.manifest = self.work / "m.txt"
        self.manifest.write_text("*.py\nscripts\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_preview_lists_only_manifest_paths(self):
        (self.work / "code.py").write_text("a")
        (self.work / "scripts").mkdir()
        (self.work / "scripts" / "run.sh").write_text("b")
        (self.work / "data").mkdir()
        (self.work / "data" / "x.pkl").write_text("c")   # outside manifest
        info = git_sync.preview(self.work, self.manifest, "github", "current-experiments")
        joined = "\n".join(info["upload_files"])
        self.assertIn("code.py", joined)
        self.assertIn("scripts/run.sh", joined)
        self.assertNotIn("data/x.pkl", joined)
        self.assertIn("data/x.pkl", "\n".join(info["skipped_files"]))

    def test_sync_commits_and_pushes_only_manifest(self):
        (self.work / "code.py").write_text("a")
        (self.work / "data").mkdir()
        (self.work / "data" / "x.pkl").write_text("c")
        code, info = git_sync.sync(self.work, self.manifest, "github",
                                   "current-experiments", "manifest test", None)
        self.assertEqual(code, 0)
        self.assertEqual(info["status"], "pushed")
        status = _git(self.work, "status", "--porcelain").stdout
        self.assertIn("data/", status)          # data NOT committed
        self.assertNotIn("code.py", status)     # code committed
        branches = subprocess.run(["git", "--git-dir", str(self.remote), "branch", "--list"],
                                  capture_output=True, text=True).stdout
        self.assertIn("current-experiments", branches)

    def test_invalid_target_is_rejected(self):
        code, info = git_sync.sync(self.work, self.manifest, "github",
                                   "current-experiments", None, None, target="nonsense")
        self.assertEqual(code, 2)
        self.assertEqual(info["status"], "invalid_target")

    def test_refuses_when_staged_outside_manifest(self):
        (self.work / "outside.txt").write_text("x")
        _git(self.work, "add", "--", "outside.txt")   # staged, but not in manifest
        (self.work / "code.py").write_text("a")
        code, info = git_sync.sync(self.work, self.manifest, "github",
                                   "current-experiments", None, None)
        self.assertEqual(code, 1)
        self.assertEqual(info["status"], "refused_staged_outside_manifest")
        self.assertIn("outside.txt", info["staged_outside_manifest"])
        # nothing was pushed
        status = _git(self.work, "status", "--porcelain").stdout
        self.assertIn("code.py", status)

    def test_receipt_has_unified_schema(self):
        import tempfile as _tf
        with _tf.TemporaryDirectory() as d:
            (self.work / "code.py").write_text("a")
            code, _ = git_sync.sync(self.work, self.manifest, "github",
                                    "current-experiments", "m", Path(d), target="method")
            self.assertEqual(code, 0)
            receipt = json.loads((Path(d) / "latest-method.json").read_text())
            for key in ("target", "repository", "branch", "manifest_hash",
                        "pre_sync_commit", "post_sync_commit",
                        "remote_commit_before", "remote_commit_after",
                        "uploaded", "excluded", "synced_at_utc"):
                self.assertIn(key, receipt)
            self.assertEqual(receipt["target"], "method")


if __name__ == "__main__":
    unittest.main(verbosity=2)
