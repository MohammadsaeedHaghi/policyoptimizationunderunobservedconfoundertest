"""Marginal Sensitivity Model — code 1.1 / common.

The unobserved confounder S makes the **estimated** inverse weights wrong by an unknown amount.
The Marginal Sensitivity Model (Tan / Rosenbaum) bounds that error by a single factor Γ: the
true (S-aware) weight lies in a per-unit interval around the nominal (estimated, Hájek-normalised)
weight. The robust methods (R-OW, R-O, Regret-O/OW) optimise the **worst case over this box**.

Public API
----------
    matched_gamma(gamma)                     -> Γ = e^{γ/2}  (the matched Rosenbaum sensitivity)
    marginal_sensitivity_box(w_hat, Gamma)   -> (a, b)  per-unit interval [a_i, b_i], a <= b
    augmented_gamma_grid(gammas, gamma_true) -> sorted Γ-sweep with the matched Γ inserted
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

__all__ = ["matched_gamma", "marginal_sensitivity_box", "augmented_gamma_grid"]


def matched_gamma(gamma: float) -> float:
    """The matched Rosenbaum sensitivity Γ = e^{γ/2} for DGP confounding level ``gamma``.

    This is the Γ at which the MSM box is "just big enough" to contain the true confounding
    induced by S shifting the factual logit by ±γ/2 (verified empirically by the Rosenbaum
    checks: realised Λ̂ ≈ e^{γ/2} for single-signed confounding).
    """
    return float(np.exp(gamma / 2.0))


def marginal_sensitivity_box(w_hat: np.ndarray, Gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    r"""Per-unit MSM interval on the confounded inverse weight.

    The Tan/Rosenbaum interval is :math:`\{1 + t\,(\hat w_i - 1) : t \in [\Gamma^{-1}, \Gamma]\}`,
    with endpoints :math:`1 + \Gamma^{-1}(\hat w_i - 1)` and :math:`1 + \Gamma(\hat w_i - 1)`.

    For :math:`\hat w_i \ge 1` the first endpoint is the lower one, but the per-arm Hájek
    normalisation can push some weights **below 1**, in which case the endpoints swap. Taking the
    element-wise ``min``/``max`` returns ``[a, b]`` with ``a <= b`` in either regime — without
    this a single ``w_hat`` just under 1 inverts its box and makes the dual LP unbounded (seen at
    strong confounding, γ_true = 5).

    Requires ``Gamma >= 1``. ``Gamma == 1`` collapses the box to a point (a = b = w_hat) — the
    no-robustness case where the robust methods coincide with Direct IPW.
    """
    if Gamma < 1.0:
        raise ValueError("Gamma must be >= 1.0.")
    w_hat = np.asarray(w_hat, dtype=float)
    lo = 1.0 + (1.0 / Gamma) * (w_hat - 1.0)
    hi = 1.0 + Gamma * (w_hat - 1.0)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def augmented_gamma_grid(gammas: Iterable[float], gamma_true: float, *, digits: int = 4) -> list:
    """Sorted Γ-sweep that always includes the matched Γ = e^{γ_true/2}.

    Mirrors the runners' ``sorted(set(gammas) | {round(exp(gt/2), digits)})`` so the matched Γ is
    a sampled, flagged point on every sweep.
    """
    grid = {round(float(g), digits) for g in gammas}
    grid.add(round(matched_gamma(gamma_true), digits))
    return sorted(grid)
