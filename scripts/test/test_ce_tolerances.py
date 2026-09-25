#!/usr/bin/env python3
"""Regression tests for continuous CE statistic/tolerance combinations.

The covariance arm in A3 uses the same uncertainty controls as the reference
CE arm.  These tests ensure that covariance + window/HAC tolerance remains a
valid configuration and that the tolerance path uses the configured statistic
instead of silently falling back to partial correlation.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hc_predictor_ce import (  # noqa: E402
    constraint_window_statistics,
    independent_expectation_tolerances,
)


def _cfg(statistic_kind: str, se_method: str) -> SimpleNamespace:
    return SimpleNamespace(
        ci_penalty_kind="conditional_expectation",
        ce_tolerance_mode="standard_error",
        ce_statistic_kind=statistic_kind,
        ce_se_method=se_method,
        ce_window_n_windows=5,
        ce_window_min_size=20,
        ce_hac_max_lag=6,
        ce_residualize_method="linear",
        ce_residualize_iterations=12,
        ce_residualize_eps=1e-3,
        ce_statistic_eps=1e-8,
        ce_statistic_shrinkage=0.05,
        ce_tolerance_sd_multiplier=1.96,
        ce_window_sign_epsilon=0.01,
    )


class TestContinuousCETolerances(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(7)
        self.X = torch.randn(100, 3, generator=generator)
        self.y = 0.3 * self.X[:, 0] + 0.2 * torch.randn(100, generator=generator)
        self.spec = {"x_index": 0, "y_index": 3, "z_indices": [1]}

    def test_covariance_window_uses_covariance_statistics(self):
        cfg = _cfg("covariance", "window")
        summary = constraint_window_statistics(self.X, self.y, self.spec, cfg=cfg)
        tolerance = independent_expectation_tolerances(
            self.X, self.y, [self.spec], cfg=cfg
        )
        self.assertEqual(summary["window_count"], 5)
        self.assertGreater(summary["window_std"], 0.0)
        self.assertTrue(torch.isfinite(tolerance).all())
        self.assertAlmostEqual(
            float(tolerance[0]),
            1.96 * float(summary["window_std"]),
            places=5,
        )

    def test_covariance_hac_is_supported(self):
        cfg = _cfg("covariance", "hac")
        tolerance = independent_expectation_tolerances(
            self.X, self.y, [self.spec], cfg=cfg
        )
        self.assertTrue(torch.isfinite(tolerance).all())
        self.assertGreater(float(tolerance[0]), 0.0)

    def test_partial_correlation_path_remains_supported(self):
        cfg = _cfg("partial_correlation", "window")
        tolerance = independent_expectation_tolerances(
            self.X, self.y, [self.spec], cfg=cfg
        )
        self.assertTrue(torch.isfinite(tolerance).all())
        self.assertGreater(float(tolerance[0]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
