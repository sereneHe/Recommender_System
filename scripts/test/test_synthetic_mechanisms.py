#!/usr/bin/env python3
"""Tests for the A8 NN-favourable synthetic mechanisms.

Verify that each mechanism is deterministic, acyclic, and — critically —
that ``oracle_fn`` really returns the TRUE structural conditional mean of the
generating mechanism (not a linear ``X @ W`` surrogate).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import synthetic_utils as su  # noqa: E402


class TestMechanisms(unittest.TestCase):
    def _gen(self, mechanism, **kw):
        params = dict(graph_type="ER", n_samples=800, n_nodes=12, expected_edges=20,
                      mechanism=mechanism, graph_seed=42, noise_seed=101)
        params.update(kw)
        return su.simulate_synthetic_problem(**params)

    def test_all_mechanisms_run_and_are_finite(self):
        for mech in su.MECHANISMS:
            with self.subTest(mechanism=mech):
                df, w, oracle, meta = self._gen(mech)
                self.assertEqual(df.shape, (800, 12))
                self.assertTrue(np.isfinite(df.to_numpy()).all())
                self.assertEqual(meta["mechanism"], mech)
                self.assertEqual(meta["scope"], su.MECHANISM_SCOPE[mech])
                self.assertTrue(np.isfinite(oracle(df.to_numpy())).all())

    def test_ground_truth_dag_is_acyclic_and_reaches_target(self):
        for mech in su.MECHANISMS:
            with self.subTest(mechanism=mech):
                _, w, _, meta = self._gen(mech)
                self.assertTrue(np.allclose(w, np.triu(w)), f"{mech}: W must be upper-triangular")
                self.assertTrue(meta["true_parents_of_target"], f"{mech}: target needs parents")

    def test_determinism_graph_vs_noise(self):
        a, wa, _, _ = self._gen("smooth_additive")
        b, wb, _, _ = self._gen("smooth_additive")
        pd.testing.assert_frame_equal(a, b)
        np.testing.assert_array_equal(wa, wb)
        c, wc, _, _ = self._gen("smooth_additive", noise_seed=202)
        np.testing.assert_array_equal(wa, wc)          # same graph
        self.assertFalse(a.equals(c))                   # different innovations

    def test_oracle_is_true_structural_mean_smooth_additive(self):
        df, w, oracle, meta = self._gen("smooth_additive")
        target = int(meta["target_node"][1:])
        parents = np.flatnonzero(w[:, target])
        link = su._SMOOTH_LINKS[target % len(su._SMOOTH_LINKS)]
        expected = su._link(link, df.to_numpy()[:, parents] @ w[parents, target])
        np.testing.assert_allclose(oracle(df.to_numpy()), expected, rtol=1e-10, atol=1e-10)

    def test_oracle_is_not_linear_surrogate(self):
        # The whole point: E[Y|X] must NOT equal the linear X @ W on nonlinear
        # mechanisms, otherwise the audit would be vacuous.
        for mech in ("smooth_additive", "periodic", "highdim_smooth"):
            with self.subTest(mechanism=mech):
                df, w, oracle, meta = self._gen(mech)
                target = int(meta["target_node"][1:])
                linear = df.to_numpy() @ w[:, target]
                self.assertFalse(np.allclose(oracle(df.to_numpy()), linear, atol=1e-6))

    def test_linear_path_is_unchanged_and_matches_legacy(self):
        # historical both-seeds-unset path must be reproducible
        df1, w1 = su.load_data(n_samples=500, n_nodes=10, expected_edges=15,
                               graph_type="ER", seed=1)
        df2, w2 = su.load_data(n_samples=500, n_nodes=10, expected_edges=15,
                               graph_type="ER", seed=1)
        pd.testing.assert_frame_equal(df1, df2)
        np.testing.assert_array_equal(w1, w2)
        # linear mechanism routed via simulate_synthetic_problem agrees with load_data
        df3, w3 = su.load_data(n_samples=500, n_nodes=10, expected_edges=15,
                               graph_type="ER", graph_seed=42, noise_seed=101,
                               mechanism="linear")
        df4, w4, _, _ = su.simulate_synthetic_problem(
            n_samples=500, n_nodes=10, expected_edges=15, graph_type="ER",
            mechanism="linear", graph_seed=42, noise_seed=101)
        pd.testing.assert_frame_equal(df3, df4)
        np.testing.assert_array_equal(w3, w4)

    def test_unknown_mechanism_rejected(self):
        with self.assertRaises(ValueError):
            su.simulate_synthetic_problem(mechanism="not_a_mechanism")

    def test_public_oracle_matches_generator_closure(self):
        for mech in ("smooth_additive", "compositional", "periodic", "highdim_smooth"):
            with self.subTest(mechanism=mech):
                df, w, oracle, meta = self._gen(mech)
                target = int(meta["target_node"][1:])
                public = su.structural_conditional_mean(w, df.to_numpy(), mech, target)
                np.testing.assert_allclose(oracle(df.to_numpy()), public, rtol=1e-10, atol=1e-10)

    def test_temporal_oracle_uses_lag_not_static(self):
        df, w, oracle, meta = self._gen("temporal_smooth")
        target = int(meta["target_node"][1:])
        temporal = oracle(df.to_numpy())
        static = su.structural_conditional_mean(w, df.to_numpy(), "temporal_smooth", target)
        self.assertTrue(np.isfinite(temporal).all())
        # The lag genuinely changes E[Y|X]; a static oracle would be wrong here.
        self.assertFalse(np.allclose(temporal, static, atol=1e-6))

    def test_save_artifacts(self):
        df, w, oracle, meta = self._gen("compositional")
        with tempfile.TemporaryDirectory() as d:
            paths = su.save_synthetic_artifacts(d, df, w, oracle, meta)
            for key in ("W_true", "generator_metadata", "oracle_prediction"):
                self.assertTrue(Path(paths[key]).exists(), key)
            oracle_df = pd.read_csv(paths["oracle_prediction"])
            self.assertEqual(len(oracle_df), len(df))
            self.assertIn("oracle_mean", oracle_df.columns)


class TestMechanismScopes(unittest.TestCase):
    def test_scopes_are_distinct_from_plain_ER(self):
        scopes = set(su.MECHANISM_SCOPE.values())
        self.assertEqual(len(scopes), len(su.MECHANISMS))
        self.assertNotEqual(su.MECHANISM_SCOPE["smooth_additive"], "synthetic/ER")


if __name__ == "__main__":
    unittest.main(verbosity=2)
