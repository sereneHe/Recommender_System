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


class TestTemporalLagChannel(unittest.TestCase):
    """The lag-aware oracle contract for the A8.5 preflight."""

    @staticmethod
    def _problem():
        return su.simulate_synthetic_problem(
            graph_type="ER", n_samples=300, n_nodes=6, expected_edges=8,
            mechanism="temporal_smooth", graph_seed=42, noise_seed=101,
            target_node=5)

    def test_static_temporal_oracle_is_refused(self):
        df, w, _oracle, _meta = self._problem()
        with self.assertRaises(ValueError):
            su.structural_conditional_mean(
                w, df.to_numpy(float), "temporal_smooth", 5, require_lag=True)

    def test_lag_oracle_reproduces_the_generator_mean(self):
        df, w, _oracle, meta = self._problem()
        X = df.to_numpy(float)
        _cur, lag = su.temporal_parent_blocks(w, X, 5)
        rho = float(meta["temporal_rho"])
        parents = np.flatnonzero(w[:, 5])
        got = su.structural_conditional_mean(
            w, X, "temporal_smooth", 5, temporal_parents=lag,
            temporal_rho=rho, require_lag=True)
        truth = np.tanh(X[:, parents] @ w[parents, 5]
                        + rho * (lag @ w[parents, 5]))
        np.testing.assert_allclose(got, truth, atol=1e-12)

    def test_add_lag_features_adds_one_column_per_predictor(self):
        df, _w, _oracle, _meta = self._problem()
        lagged = su.add_lag_features(df, 1)
        predictors = [c for c in df.columns[:-1]]
        self.assertEqual(len(lagged.columns), len(df.columns) + len(predictors))
        for col in predictors:
            self.assertIn(f"{col}_lag1", lagged.columns)
        # row 0 has no predecessor, so the lag column starts undefined
        self.assertTrue(bool(lagged[f"{predictors[0]}_lag1"].isna().iloc[0]))

    def test_temporal_generator_row0_is_all_zero(self):
        """Documents the initialisation artefact a burn-in must remove.

        The sequential generator writes rows ``1..n-1`` only, so row 0 stays
        all-zero: an impossible observation that is currently fed to models.
        """
        df, _w, _oracle, _meta = self._problem()
        self.assertTrue(bool(np.all(df.to_numpy(float)[0] == 0.0)))


class TestBurnInMixingDiagnostic(unittest.TestCase):
    """The non-training mixing diagnostic that must freeze any burn-in."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = ROOT / "scripts" / "evidence_tree" / "burnin_mixing_diagnostic.py"
        spec = importlib.util.spec_from_file_location("burnin_diag", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_steady_series_needs_no_burn_in(self):
        rng = np.random.default_rng(0)
        steady = rng.normal(0.0, 1.0, size=(5000, 3))
        self.assertEqual(self.mod.mixing_time(steady, stride=10)["burn_in"], 0)

    def test_known_transient_is_detected(self):
        rng = np.random.default_rng(1)
        steady = rng.normal(0.0, 1.0, size=(5000, 3))
        transient = np.full((100, 3), 50.0)
        res = self.mod.mixing_time(np.vstack([transient, steady]), stride=10)
        self.assertTrue(res["converged"])
        self.assertGreaterEqual(res["burn_in"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
