#!/usr/bin/env python3
"""Oracle-audit wiring tests for the estimator (no training required).

The estimator's ``_synthetic_oracle_prediction_normalized`` only needs
``w_est``, ``row_and_col_names``, ``target_col`` and a few normalisation
statistics, so we drive the *real* method with a lightweight stub instead of
fitting a model.  This verifies stage A: E[Y|X] is the true generator oracle.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import synthetic_utils as su  # noqa: E402
import recommender_estimator as re  # noqa: E402


def _stub(mechanism, oracle="synthetic_generator_oracle"):
    df, w, oracle_fn, meta = su.simulate_synthetic_problem(
        graph_type="ER", n_samples=400, n_nodes=8, expected_edges=12,
        mechanism=mechanism, graph_seed=42, noise_seed=101)
    target = meta["target_node"]
    features = [c for c in df.columns if c != target]
    s = SimpleNamespace()
    s.cfg = SimpleNamespace(
        constraint_audit_oracle=oracle,
        constraint_audit_oracle_mechanism=mechanism,
    )
    s.w_est = w
    s.row_and_col_names = [str(c) for c in df.columns]
    s.target_col = target
    s._y_mean = 0.0
    s._y_std = 1.0
    # Bind the real helper methods so the stub exercises the actual code path.
    s.get_current_column_names = MethodType(
        re.RecommenderBaseEstimator.get_current_column_names, s)
    s._synthetic_true_w_for_names = MethodType(
        re.RecommenderBaseEstimator._synthetic_true_w_for_names, s)
    return s, df, w, oracle_fn, target, features


class TestGeneratorOracle(unittest.TestCase):
    def test_static_mechanisms_use_true_conditional_mean(self):
        for mech in ("smooth_additive", "compositional", "highdim_smooth", "periodic"):
            with self.subTest(mechanism=mech):
                s, df, w, oracle_fn, target, features = _stub(mech)
                X = df[features]
                out = re.HCRecommenderPredictor._synthetic_oracle_prediction_normalized(s, X)
                expected = oracle_fn(df.to_numpy())
                np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-9)

    def test_linear_mechanism_matches_linear_path(self):
        s, df, w, oracle_fn, target, features = _stub(
            "linear", oracle="synthetic_linear_sem")
        X = df[features]
        out = re.HCRecommenderPredictor._synthetic_oracle_prediction_normalized(s, X)
        expected = df.to_numpy() @ w[:, -1]
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-9)

    def test_temporal_refused_from_static_matrix(self):
        s, df, w, oracle_fn, target, features = _stub("temporal_smooth")
        with self.assertRaises(ValueError):
            re.HCRecommenderPredictor._synthetic_oracle_prediction_normalized(s, df[features])

    def test_unknown_oracle_mode_rejected(self):
        s, df, w, oracle_fn, target, features = _stub(
            "smooth_additive", oracle="totally_unknown")
        with self.assertRaises(ValueError):
            re.HCRecommenderPredictor._synthetic_oracle_prediction_normalized(s, df[features])

    def test_none_oracle_returns_none(self):
        s, df, w, oracle_fn, target, features = _stub("smooth_additive", oracle="none")
        self.assertIsNone(
            re.HCRecommenderPredictor._synthetic_oracle_prediction_normalized(s, df[features]))

    def test_audit_metadata_fields_present(self):
        src = (ROOT / "recommender_estimator.py").read_text(encoding="utf-8")
        for field in ('"constraint_source"', '"oracle_type"', '"statistic_validity"'):
            self.assertIn(field, src)
        # generator oracle must call the true conditional mean helper
        self.assertIn("structural_conditional_mean", src)
        # statistic_validity is set only after the oracle violation exists
        self.assertIn('metadata_rows[-1]["statistic_validity"]', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
