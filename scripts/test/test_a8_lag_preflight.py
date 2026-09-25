#!/usr/bin/env python3
"""Contract tests for the A8 lag preflight skeleton and the W adapter.

These pin the parts that must be agreed *before* the W alignment is
implemented: the lag-column classification, the refusal behaviour of the
adapter, the manifest schema, and the "one shared row mask / one shared split"
invariant of the runner.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evidence_tree"))

import synthetic_utils as su  # noqa: E402
import w_adapter  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "a8_lag_preflight_runner",
    ROOT / "scripts" / "evidence_tree" / "a8_lag_preflight_runner.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


class TestWAdapterContract(unittest.TestCase):
    """Contract points 1-4 must hold before W alignment is implemented."""

    def test_lag_column_is_a_predictor_not_a_node(self):
        c = w_adapter.classify_columns(["X0", "X1", "Y"], ["X0", "X0_lag1", "Y"], "Y")
        self.assertEqual(c["node_columns"], ["X0"])
        self.assertEqual(c["lag_columns"], ["X0_lag1"])
        self.assertNotIn("X0_lag1", c["node_columns"])

    def test_unknown_column_is_refused(self):
        with self.assertRaises(ValueError):
            w_adapter.classify_columns(["X0", "Y"], ["X0", "X9", "Y"], "Y")

    def test_lag_column_of_an_unknown_node_is_refused(self):
        with self.assertRaises(ValueError):
            w_adapter.classify_columns(["X0", "Y"], ["X0", "X9_lag1", "Y"], "Y")

    def test_build_augmented_w_is_interface_only(self):
        """Failure test: the alignment must not silently exist yet."""
        with self.assertRaises(NotImplementedError):
            w_adapter.build_augmented_w(np.eye(3), ["X0", "X1", "Y"], ["X0", "X1", "Y"], "Y")

    def test_build_augmented_w_validates_before_refusing(self):
        """A mis-specified request must fail loudly, not with the placeholder."""
        with self.assertRaises(ValueError):
            w_adapter.build_augmented_w(np.eye(3), ["X0", "X1", "Y"], ["X0", "NOPE", "Y"], "Y")
        with self.assertRaises(ValueError):
            w_adapter.build_augmented_w(np.eye(2), ["X0", "X1", "Y"], ["X0", "X1", "Y"], "Y")
        with self.assertRaises(ValueError):
            w_adapter.build_augmented_w(np.eye(3), ["X0", "X1", "Y"], ["X0", "X1", "Y"], "Z")

    def test_alignment_manifest_schema(self):
        good = {
            "w_shape": [4, 4],
            "row_and_col_names": ["X0", "X0_lag1", "X1", "Y"],
            "w_source": "supplied_true_w_zero_extended",
            "node_columns": ["X0", "X1"],
            "lag_columns": ["X0_lag1"],
            "target": "Y",
        }
        self.assertEqual(w_adapter.validate_alignment_manifest(good), [])

        missing = {k: v for k, v in good.items() if k != "w_source"}
        self.assertTrue(w_adapter.validate_alignment_manifest(missing))

        bad_source = dict(good, w_source="guessed")
        self.assertTrue(w_adapter.validate_alignment_manifest(bad_source))

        bad_shape = dict(good, w_shape=[3, 3])
        self.assertTrue(w_adapter.validate_alignment_manifest(bad_shape))

        clashing = dict(good, node_columns=["X0", "X0_lag1"])
        self.assertTrue(any("both" in p
                            for p in w_adapter.validate_alignment_manifest(clashing)))


class TestPreflightSkeleton(unittest.TestCase):
    """The runner must fix the row mask and the split before anything else."""

    @classmethod
    def setUpClass(cls):
        cls.frame, cls.W, cls.meta = runner.build_frame(42, 101, 3000, 10, 15, 0.5)
        cls.target = str(cls.meta["target_node"])
        cls.tiers, cls.lagged = runner.build_tiers(cls.frame, cls.meta, cls.target)
        cls.rows = runner.shared_valid_rows(cls.lagged, cls.tiers, cls.target)
        cls.splits = runner.expanding_window_splits(cls.rows, 3, 500, 0)

    def test_every_tier_is_complete_on_the_shared_rows(self):
        for tier, spec in self.tiers.items():
            block = self.lagged[spec["columns"]].to_numpy(float)[self.rows]
            self.assertTrue(np.all(np.isfinite(block)), tier)
        y = self.lagged[self.target].to_numpy(float)[self.rows]
        self.assertTrue(np.all(np.isfinite(y)))

    def test_one_shared_row_mask(self):
        """t2 (parents only) must not use a wider or narrower row mask."""
        r1 = runner.shared_valid_rows(self.lagged, {"t1": self.tiers["t1"]}, self.target)
        r2 = runner.shared_valid_rows(self.lagged, {"t2": self.tiers["t2"]}, self.target)
        self.assertTrue(set(self.rows) <= set(r1))
        self.assertTrue(set(self.rows) <= set(r2))
        # the shared mask is exactly the intersection
        self.assertEqual(set(self.rows), set(r1) & set(r2))

    def test_expanding_window_splits_are_time_ordered(self):
        for train_idx, test_idx in self.splits:
            self.assertLess(int(max(train_idx)), int(min(test_idx)))
        sizes = [len(train_idx) for train_idx, _ in self.splits]
        self.assertEqual(sizes, sorted(sizes))

    def test_tiers_differ_only_in_their_feature_columns(self):
        self.assertTrue(all("_lag" not in c for c in self.tiers["t0"]["columns"]))
        self.assertTrue(all(not c.endswith("_lag1") for c in self.tiers["t0"]["columns"]))
        self.assertTrue(any(c.endswith("_lag1") for c in self.tiers["t1"]["columns"]))
        self.assertEqual(set(self.tiers["t2"]["columns"]),
                         {p for p in self.meta["true_parents_of_target"]}
                         | {f"{p}_lag1" for p in self.meta["true_parents_of_target"]})

    def test_oracle_is_reference_only_and_finite(self):
        node_block = self.frame.to_numpy(float)
        _cur, lag = su.temporal_parent_blocks(self.W, node_block, int(self.target[1:]))
        oracle = su.structural_conditional_mean(
            self.W, node_block, "temporal_smooth", int(self.target[1:]),
            temporal_parents=lag, require_lag=True)
        y = self.lagged[self.target].to_numpy(float)
        values = runner.nmse_folds(y, oracle, self.splits)
        self.assertTrue(np.all(np.isfinite(values)))

    def test_nmse_uses_the_repository_fold_normaliser(self):
        y = self.lagged[self.target].to_numpy(float)
        perfect = y.copy()
        values = runner.nmse_folds(y, perfect, self.splits)
        np.testing.assert_allclose(values, 0.0, atol=1e-12)

    def test_split_budget_guards_the_masked_row_count(self):
        """The masked row count, not n_samples, sets the split budget."""
        with self.assertRaises(ValueError):
            runner.expanding_window_splits(self.rows, 5, 2000, 0)
        with self.assertRaises(ValueError):
            runner.expanding_window_splits(self.rows, 1, 100, 0)

    def test_unsorted_rows_are_refused(self):
        """The repo's time_series contract refuses unsorted input; so must we."""
        runner.assert_time_ordered(self.lagged.index)  # sorted input passes
        with self.assertRaises(ValueError):
            runner.assert_time_ordered(self.lagged.iloc[::-1].index)

    def test_inner_validation_is_the_last_time_block(self):
        train, val = runner.last_block_validation_split(100, 0.2)
        self.assertEqual(len(val), 20)
        self.assertEqual(int(val[0]), 80)
        self.assertEqual(int(val[-1]), 99)
        self.assertTrue(np.all(np.diff(train) == 1))
        self.assertLess(int(train.max()), int(val.min()))

    def test_inner_validation_matches_the_estimator_ceil_rule(self):
        # recommender_estimator uses ceil(n * fraction) and needs >=2 fit rows.
        train, val = runner.last_block_validation_split(7, 0.2)
        self.assertEqual(len(val), 2)   # ceil(1.4) = 2
        self.assertEqual(len(train), 5)
        with self.assertRaises(ValueError):
            runner.last_block_validation_split(3, 0.5)   # would leave 1 train row
        with self.assertRaises(ValueError):
            runner.last_block_validation_split(10, 1.0)

    def test_dry_run_does_not_fit_models(self):
        """--dry-run must emit a plan, never a model number."""
        import contextlib
        import io
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dry.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = runner.main([
                    "--dry-run", "--graph-seeds", "42", "--noise-seeds", "101",
                    "--n-samples", "3000", "--n-nodes", "10", "--expected-edges", "15",
                    "--n-splits", "3", "--test-size", "500", "--out", str(out),
                ])
            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text())
            report = payload["reports"][0]
        self.assertFalse(payload["strict_inference_allowed"])
        xgb = [r for r in report["rows"] if r["model"] == "xgb100"]
        self.assertTrue(xgb)
        self.assertTrue(all(np.isnan(r["nmse"]) for r in xgb))
        self.assertTrue(all(r.get("status") == "dry-run" for r in xgb))
        self.assertEqual(report["model_status"]["xgb100"], "dry-run/skipped")
        # the oracle reference is still computed (it is not a fitted model)
        self.assertEqual(report["oracle_metrics"]["evaluated_on"],
                         "t2 features (P_t + P_{t-1})")

    def test_nn0_is_blocked_by_the_w_adapter(self):
        with self.assertRaises(NotImplementedError):
            runner.fit_predict_nn0(
                None, None, None,
                base_names=["X0", "X1", "Y"],
                feature_names=["X0", "X0_lag1", "X1", "Y"],
                base_w=np.eye(3),
                target="Y")


if __name__ == "__main__":
    unittest.main(verbosity=2)
