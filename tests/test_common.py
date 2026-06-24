"""common pre-algorithm primitives: Hájek weights, no-true-propensities, MSM box (eq 3), support, geometry."""
import inspect

import numpy as np

import common


def test_hajek_per_arm_sum_n(k2):
    X, T, K = k2["X"], k2["T"], k2["K"]
    w, _ = common.ipw_weights_from_data(X, T, K)
    n = len(T)
    for k in range(K):                                   # per-arm Σ w = n is the project convention
        assert np.isclose(w[T == k].sum(), n, rtol=1e-6)


def test_no_true_propensities_signature():
    sig = inspect.signature(common.ipw_weights_from_data)
    names = list(sig.parameters)
    assert names[:2] == ["X", "T"]                        # takes X,T only — structurally cannot get e_true
    assert not any(p in sig.parameters for p in ("e_true", "S", "propensity_true", "true_prop"))


def test_msm_box_matches_eq3():
    w = np.array([0.5, 1.0, 2.0]); G = 2.0
    a, b = common.marginal_sensitivity_box(w, G)
    lo = 1 + (1.0 / G) * (w - 1); hi = 1 + G * (w - 1)   # Kallus & Zhou eq (3)
    assert np.allclose(a, np.minimum(lo, hi))
    assert np.allclose(b, np.maximum(lo, hi))


def test_matched_gamma():
    assert np.isclose(common.matched_gamma(0.0), 1.0)
    assert np.isclose(common.matched_gamma(2.0), np.exp(1.0))


def test_support_snap_and_ties():
    X = np.array([[0.10], [0.12], [0.90]])
    snapped = common.snap_to_grid(X, 6, -1.0, 1.0)
    groups = common.tie_groups(snapped)
    assert 1 <= len(groups) <= 3
    assert sum(len(g) for g in groups) == 3


def test_distance_matrix_zscore_changes_cost():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 10.0]])  # feature 2 on a much larger scale
    D0, _, _ = common.distance_matrix(X, zscore=False)
    Dz, _, _ = common.distance_matrix(X, zscore=True)
    assert np.allclose(D0, D0.T) and np.allclose(np.diag(D0), 0.0)
    assert not np.allclose(D0, Dz)                        # standardising changes the ground cost
