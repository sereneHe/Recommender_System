#!/usr/bin/env python3
"""Unit tests for the frozen comparison contract and the F0/G0 gates.

Run with the project interpreter, e.g.::

    ./codiet311/bin/python -m unittest discover -s scripts/test -p 'test_*.py'

The tests use only the standard library plus pandas/yaml (already required by
the evidence builder) so they run anywhere the builder runs.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "evidence_tree"))

import build_evidence_index as bei  # noqa: E402
import experiment_registry as reg  # noqa: E402
import f0_guard  # noqa: E402
import g0_check  # noqa: E402

REGISTRY_PATH = ROOT / "experiment_registry.yaml"


def run(split="s1", w="w1"):
    return SimpleNamespace(split_hash=split, fold_w_cache_hash=w)


class TestFoldWContract(unittest.TestCase):
    """Acceptance: the four fold-W contracts each have a unit test."""

    def test_same_requires_equal_hashes(self):
        ok = bei.pair_contract(run(w="same"), run(w="same"), "same", strict=True)
        self.assertTrue(ok["fold_w_contract_valid"])
        self.assertTrue(ok["pair_contract_valid"])
        bad = bei.pair_contract(run(w="a"), run(w="b"), "same", strict=True)
        self.assertFalse(bad["fold_w_contract_valid"])
        self.assertFalse(bad["pair_contract_valid"])

    def test_candidate_requires_only_candidate_receipt(self):
        ok = bei.pair_contract(run(w=""), run(w="cand"), "candidate", strict=True)
        self.assertTrue(ok["pair_contract_valid"])
        bad = bei.pair_contract(run(w=""), run(w=""), "candidate", strict=True)
        self.assertFalse(bad["pair_contract_valid"])

    def test_both_requires_two_receipts_but_not_equal(self):
        ok = bei.pair_contract(run(w="a"), run(w="b"), "both", strict=True)
        self.assertTrue(ok["pair_contract_valid"])
        self.assertFalse(ok["fold_w_cache_hash_equal"])
        bad = bei.pair_contract(run(w="a"), run(w=""), "both", strict=True)
        self.assertFalse(bad["pair_contract_valid"])

    def test_none_never_requires_a_receipt(self):
        ok = bei.pair_contract(run(w=""), run(w=""), "none", strict=True)
        self.assertTrue(ok["pair_contract_valid"])
        self.assertFalse(ok["fold_w_cache_applicable"])

    def test_strict_false_is_non_blocking(self):
        c = bei.pair_contract(run(w=""), run(w=""), "same", strict=False)
        self.assertFalse(c["fold_w_contract_valid"])
        self.assertTrue(c["pair_contract_valid"])


class TestDuplicateRejection(unittest.TestCase):
    """Acceptance: a repeated arm in the same cohort must be rejected."""

    def test_single_pair_is_allowed(self):
        self.assertIsNone(bei.duplicate_reason([run()], [run()]))

    def test_duplicate_reference_is_rejected(self):
        reason = bei.duplicate_reason([run(), run()], [run()])
        self.assertEqual(reason, "invalid_pairing:duplicate_within_cohort")

    def test_duplicate_candidate_is_rejected(self):
        self.assertEqual(
            bei.duplicate_reason([run()], [run(), run()]),
            "invalid_pairing:duplicate_within_cohort",
        )


class TestA4MipTime(unittest.TestCase):
    """Acceptance: only mip_time_300 may enter the strict A4 time comparison."""

    def _candidate(self):
        for entry in bei.COMPARISONS:
            if entry[0] == "A4.mip_time.er":
                return entry[4]
        self.fail("A4.mip_time.er not defined")

    def test_exact_300_arm_only(self):
        self.assertEqual(self._candidate(), "EV:A4.mip_time:mip_time_300")
        self.assertNotIn("*", self._candidate())


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = reg.load(REGISTRY_PATH)

    def test_registry_valid(self):
        self.assertEqual(reg.validate(self.registry), [])

    def test_h1_has_twenty_units(self):
        h1 = reg.hypothesis(self.registry, "H1")
        self.assertEqual(reg.seed_table_units(h1["seed_table"]), 20)
        self.assertEqual(reg.seed_table_units(h1["seed_table"]), h1["min_independent_units"])

    def test_h1_and_f0_seeds_disjoint(self):
        h1 = reg.hypothesis(self.registry, "H1")
        f0 = reg.hypothesis(self.registry, "F0")
        overlap = set(reg.seed_combinations(h1["seed_table"])) & set(
            reg.seed_combinations(f0["reserved_holdout_seed_table"])
        )
        self.assertEqual(overlap, set())

    def test_duplicate_cohort_reservation_fails(self):
        with tempfile.TemporaryDirectory() as d:
            reg_dir = Path(d) / "cohort_registry"
            reg.reserve_cohort("H1", "et_h1_test", registry_path=REGISTRY_PATH,
                               registry_dir=reg_dir)
            with self.assertRaises(RuntimeError):
                reg.reserve_cohort("H1", "et_h1_test", registry_path=REGISTRY_PATH,
                                   registry_dir=reg_dir)

    def test_concurrent_reservation_wins_exactly_once(self):
        import threading
        with tempfile.TemporaryDirectory() as d:
            reg_dir = Path(d) / "cohort_registry"
            wins, errors = [], []
            def attempt():
                try:
                    reg.reserve_cohort("H1", "et_h1_race", registry_path=REGISTRY_PATH,
                                       registry_dir=reg_dir)
                    wins.append(1)
                except RuntimeError:
                    errors.append(1)
            threads = [threading.Thread(target=attempt) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(len(wins), 1)
            self.assertEqual(len(errors), 7)

    def test_remote_duplicate_and_unreachable_fail_closed(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            reg_dir = Path(d) / "cohort_registry"
            dup_runner = lambda cmd: SimpleNamespace(returncode=1, stderr="mkdir: exists")
            with self.assertRaises(RuntimeError):
                reg.reserve_cohort("H1", "et_h1_remote", registry_path=REGISTRY_PATH,
                                   registry_dir=reg_dir, remote=True,
                                   _ssh_runner=dup_runner)
            self.assertFalse((reg_dir / "et_h1_remote").exists())
            down_runner = lambda cmd: SimpleNamespace(returncode=255, stderr="ssh: connect")
            with self.assertRaises(RuntimeError):
                reg.reserve_cohort("H1", "et_h1_remote", registry_path=REGISTRY_PATH,
                                   registry_dir=reg_dir, remote=True,
                                   _ssh_runner=down_runner)

    def test_retire_marks_cohort_ineligible(self):
        with tempfile.TemporaryDirectory() as d:
            reg_dir = Path(d) / "cohort_registry"
            reg.reserve_cohort("H1", "et_h1_old", registry_path=REGISTRY_PATH,
                               registry_dir=reg_dir)
            self.assertTrue(reg.is_cohort_eligible_for_strict(
                "et_h1_old", self.registry, registry_dir=reg_dir))
            reg.retire_cohort("et_h1_old", "interrupted",
                              superseded_by="et_h1_new", registry_dir=reg_dir)
            self.assertFalse(reg.is_cohort_eligible_for_strict(
                "et_h1_old", self.registry, registry_dir=reg_dir))
            self.assertFalse(reg.is_cohort_eligible_for_strict(
                "priority_ce_20200101", self.registry, registry_dir=reg_dir))
            self.assertFalse(reg.is_cohort_eligible_for_strict(
                "legacy_unknown_cohort", self.registry, registry_dir=reg_dir))


class TestF0Guard(unittest.TestCase):
    def setUp(self):
        self.registry = yaml.safe_load(REGISTRY_PATH.read_text())

    def _valid_receipt(self):
        f0 = self.registry["hypotheses"]["F0"]
        payload = {
            "schema_version": 2,
            "scope": "ER",
            "protocol_version": self.registry["registry_version"],
            "selected_config_hash": "abc",
            "candidate_set": ["abc", "def"],
            "validation_artifact": "reports/validation.yaml",
            "validation_report_hash": "deadbeef",
            "selection_metric": "nmse",
            "selection_rule": "lowest validation nmse",
            "code_commit": "unknown",
            "reserved_holdout_seed_table": f0["reserved_holdout_seed_table"],
            "reserved_holdout_seed_table_hash": reg.seed_table_hash(
                f0["reserved_holdout_seed_table"]
            ),
            "registry_hash": f0_guard.file_hash(REGISTRY_PATH),
            "holdout_metrics_present": False,
        }
        payload["receipt_hash"] = f0_guard.receipt_hash(payload)
        return payload

    def test_valid_receipt_passes(self):
        self.assertEqual(
            f0_guard.validate_receipt(self._valid_receipt(), self.registry, REGISTRY_PATH), []
        )

    def test_schema_v1_receipt_rejected(self):
        r = self._valid_receipt()
        r["schema_version"] = 1
        self.assertTrue(f0_guard.validate_receipt(r, self.registry, REGISTRY_PATH))

    def test_tampered_hash_rejected(self):
        r = self._valid_receipt()
        r["selected_config_hash"] = "tampered"
        self.assertIn(
            "receipt_hash mismatch",
            f0_guard.validate_receipt(r, self.registry, REGISTRY_PATH),
        )

    def test_holdout_metric_receipt_rejected(self):
        r = self._valid_receipt()
        r["holdout_metrics_present"] = True
        r["receipt_hash"] = f0_guard.receipt_hash(r)
        self.assertTrue(f0_guard.validate_receipt(r, self.registry, REGISTRY_PATH))

    def test_shell_command_rejected(self):
        self.assertTrue(f0_guard.check_command("bash -c 'rm -rf /'; echo hi"))

    def test_interpreter_runner_rejected(self):
        self.assertTrue(f0_guard.check_command("/bin/bash -c 'true'"))

    def test_missing_runner_rejected(self):
        self.assertTrue(f0_guard.check_command("/nonexistent/runner.sh"))

    def test_plain_executable_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "runner.sh"
            p.write_text("#!/usr/bin/env bash\ntrue\n")
            p.chmod(0o755)
            self.assertEqual(f0_guard.check_command(f"{p} --frozen x"), [])

    def test_seed_outside_reserved_rejected(self):
        f0 = self.registry["hypotheses"]["F0"]
        reserved = f0["reserved_holdout_seed_table"]
        ok = {"graph_seeds": reserved["graph_seeds"], "noise_seeds": reserved["noise_seeds"]}
        self.assertEqual(f0_guard.check_seed_table(ok, reserved), [])
        partial = {"graph_seeds": reserved["graph_seeds"][:2], "noise_seeds": reserved["noise_seeds"]}
        self.assertTrue(f0_guard.check_seed_table(partial, reserved))
        bad = {"graph_seeds": [9999], "noise_seeds": [reserved["noise_seeds"][0]]}
        self.assertTrue(f0_guard.check_seed_table(bad, reserved))


class TestG0Check(unittest.TestCase):
    def setUp(self):
        self.registry = yaml.safe_load(REGISTRY_PATH.read_text())

    def _frozen_row(self, **overrides):
        row = {
            "comparison_id": "H1.ce_vs_nn.ER",
            "n_clusters_used": 20,
            "n_clusters_expected": 20,
            "n_excluded": 0,
            "rel_improvement": 0.08,
            "ci_lo": 0.02,
            "ci_hi": 0.14,
            "status": "supported",
            "split_hash_equal": True,
            "split_receipts_present": True,
            "nuisance_hash_equal": True,
            "fold_w_cache_hash_equal": True,
            "fold_w_cache_applicable": True,
            "fold_w_receipts_present": True,
        }
        row.update(overrides)
        return row

    def test_complete_frozen_row_passes(self):
        failures, _ = g0_check.audit_index(pd.DataFrame([self._frozen_row()]), self.registry)
        self.assertEqual(failures, [])

    def test_missing_columns_fail(self):
        failures, _ = g0_check.audit_index(pd.DataFrame([{"comparison_id": "H1.ce_vs_nn.ER"}]), self.registry)
        self.assertTrue(any("G0.1" in f for f in failures))

    def test_incomplete_units_fail(self):
        failures, _ = g0_check.audit_index(
            pd.DataFrame([self._frozen_row(n_clusters_used=7, n_clusters_expected=20)]),
            self.registry,
        )
        self.assertTrue(any("G0.2" in f for f in failures))

    def test_missing_split_receipt_fails(self):
        failures, _ = g0_check.audit_index(
            pd.DataFrame([self._frozen_row(split_receipts_present=False)]), self.registry
        )
        self.assertTrue(any("G0.3" in f for f in failures))

    def test_duplicate_pairing_fails(self):
        failures, _ = g0_check.audit_index(
            pd.DataFrame([self._frozen_row(n_excluded=2)]), self.registry
        )
        self.assertTrue(any("unique_pairing" in f for f in failures))


class TestH1Protocol(unittest.TestCase):
    def test_h1_is_frozen_with_twenty_units(self):
        proto = bei.protocol_for("H1.ce_vs_nn.ER")
        self.assertTrue(proto["frozen"])
        self.assertEqual(proto["min_units"], 20)
        self.assertEqual(proto["practical"], 0.05)
        self.assertTrue(proto["strict_contract"])

    def test_legacy_comparison_uses_defaults(self):
        proto = bei.protocol_for("C0b.ce_vs_nn.repair_ER")
        self.assertFalse(proto["frozen"])
        self.assertEqual(proto["min_units"], bei.MIN_CLUSTERS)

    def test_h1_script_pins_exact_seed_table(self):
        text = (ROOT / "scripts" / "evidence_tree" / "H1_mechanism.sh").read_text()
        self.assertIn("42 43 44 45 46 47 48 49 50 51", text)
        self.assertIn("101 102", text)
        # A subset or superset must be rejected: exact match against registry.
        self.assertIn("must exactly match", text)


class TestPipelineState(unittest.TestCase):
    def test_registered_before_h1_cohort(self):
        sys.path.insert(0, str(ROOT / "scripts" / "evidence_tree"))
        import pipeline_state
        payload = pipeline_state.build()
        # A clean checkout has no cohort manifest, so the machine must sit at
        # "registered".  On an operator machine that already submitted H1 this
        # is "submitted"/"running"; either way NO method verdict may appear.
        if not (ROOT / "reports" / "cohorts").exists() or not list(
                (ROOT / "reports" / "cohorts").glob("*.yaml")):
            self.assertEqual(payload["state"], "registered")
        self.assertNotIn(payload["state"], ("validated", "frozen", "holdout confirmed"))
        self.assertEqual(payload["h1"]["used"], 0)
        self.assertEqual(payload["h1"]["expected"], 20)

    def test_dashboard_state_exposes_gates(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import serve_evidence_dashboard as dash
        s = dash.state()
        for key in ("pipeline", "gates", "nodes", "comparisons", "exclusions",
                    "cohort_registry", "cohort_manifests", "holdout_receipts",
                    "sync", "frozen_selection", "pairs"):
            self.assertIn(key, s)
        self.assertIn("G0_H1", s["gates"])

    def test_dashboard_api_matches_on_disk_layout(self):
        """The API keys must read the SAME paths the writers use."""
        import json as _json
        import tempfile as _tf
        sys.path.insert(0, str(ROOT / "scripts"))
        import serve_evidence_dashboard as dash
        with _tf.TemporaryDirectory() as d:
            base = Path(d)
            (base / "cohort_registry" / "et_h1_x").mkdir(parents=True)
            (base / "cohort_registry" / "et_h1_x" / "record.json").write_text(
                _json.dumps({"cohort_id": "et_h1_x", "status": "active"}))
            (base / "cohorts").mkdir()
            (base / "cohorts" / "et_h1_x.yaml").write_text(
                "cohort_id: et_h1_x\npbs_job_id: '12345'\nstatus: active\n")
            (base / "holdout_receipts").mkdir()
            (base / "holdout_receipts" / "et_f0_x.json").write_text(
                _json.dumps({"gate": "F0", "cohort_id": "et_f0_x"}))
            self.assertEqual(dash.read_json_dir(base / "cohort_registry")["et_h1_x"]["status"], "active")
            self.assertEqual(dash.read_yaml_dir(base / "cohorts")["et_h1_x"]["pbs_job_id"], "12345")
            self.assertEqual(dash.read_json_dir(base / "holdout_receipts")["et_f0_x"]["gate"], "F0")

    def test_frontend_declares_pipeline_state_machine(self):
        html = (ROOT / "reports" / "progress" / "dashboard.html").read_text(encoding="utf-8")
        for state in ("registered", "submitted", "running", "synced",
                      "gate_failed", "validated", "frozen",
                      "holdout_running", "holdout_confirmed"):
            self.assertIn(state, html)
        self.assertIn("renderPipeline", html)
        self.assertIn("renderGateFailures", html)


class TestSyncFreshness(unittest.TestCase):
    def _write_receipt(self, path: Path, **over):
        from datetime import datetime, timezone
        payload = {
            "schema_version": 1, "origin": "remote", "status": "ok",
            "refresh_id": "r1",
            "synced_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_hash": "abc", "file_count": 0,
        }
        payload.update(over)
        path.write_text(json.dumps(payload))
        return payload

    def test_missing_receipt_fails(self):
        with tempfile.TemporaryDirectory() as d:
            problems, _ = g0_check.validate_sync_receipt(Path(d) / "none.json")
            self.assertTrue(any("missing" in p for p in problems))

    def test_stale_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rp = Path(d) / "sync.json"
            self._write_receipt(rp, synced_at_utc="2000-01-01T00:00:00+00:00")
            problems, _ = g0_check.validate_sync_receipt(rp, max_age_sec=60)
            self.assertTrue(any("age" in p for p in problems))

    def test_refresh_id_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rp = Path(d) / "sync.json"
            self._write_receipt(rp, refresh_id="old")
            problems, _ = g0_check.validate_sync_receipt(rp, expect_refresh_id="new")
            self.assertTrue(any("refresh_id" in p for p in problems))

    def test_failed_status_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rp = Path(d) / "sync.json"
            self._write_receipt(rp, status="failed")
            problems, _ = g0_check.validate_sync_receipt(rp)
            self.assertTrue(any("status" in p for p in problems))

    def test_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            mirror = Path(d) / "mirror"
            mirror.mkdir()
            (mirror / "a.yaml").write_text("x: 1\n")
            rp = Path(d) / "sync.json"
            self._write_receipt(rp, manifest_hash="deadbeef")
            problems, _ = g0_check.validate_sync_receipt(rp, mirror_dir=mirror)
            self.assertTrue(any("changed after sync" in p for p in problems))

    def test_fresh_receipt_passes(self):
        with tempfile.TemporaryDirectory() as d:
            mirror = Path(d) / "mirror"
            mirror.mkdir()
            (mirror / "a.yaml").write_text("x: 1\n")
            digest, count, _ = g0_check.manifest_hash(mirror)
            rp = Path(d) / "sync.json"
            self._write_receipt(rp, manifest_hash=digest, file_count=count)
            problems, _ = g0_check.validate_sync_receipt(
                rp, expect_refresh_id="r1", mirror_dir=mirror)
            self.assertEqual(problems, [])


class TestCanonicalAttempt(unittest.TestCase):
    def test_duplicate_execution_keys_detects_same_layout_rerun(self):
        df = pd.DataFrame([
            {"cohort": "c1", "experiment": "ET_H1_ER_nn_graph42_noise101",
             "graph_seed": 42, "noise_seed": 101, "target": "X1", "seed": 1, "layout": "multirun"},
            {"cohort": "c1", "experiment": "ET_H1_ER_nn_graph42_noise101",
             "graph_seed": 42, "noise_seed": 101, "target": "X1", "seed": 1, "layout": "multirun"},
        ])
        keys = bei._duplicate_execution_keys(df)
        self.assertEqual(len(keys), 1)

    def test_mirror_layout_copies_are_not_flagged(self):
        df = pd.DataFrame([
            {"cohort": "c1", "experiment": "ET_H1_ER_nn_graph42_noise101",
             "graph_seed": 42, "noise_seed": 101, "target": "X1", "seed": 1, "layout": "multirun"},
            {"cohort": "c1", "experiment": "ET_H1_ER_nn_graph42_noise101",
             "graph_seed": 42, "noise_seed": 101, "target": "X1", "seed": 1, "layout": "mlruns"},
        ])
        self.assertEqual(bei._duplicate_execution_keys(df), set())

    def test_manifest_declares_attempt_and_status(self):
        text = (ROOT / "scripts" / "evidence_tree" / "submit_cohort.sh").read_text(encoding="utf-8")
        self.assertIn('"attempt": 1', text)
        self.assertIn('"status": "active"', text)
        self.assertIn("--supersedes", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
