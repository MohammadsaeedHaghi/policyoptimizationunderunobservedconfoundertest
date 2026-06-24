"""Result figures for the Policy Lab app — matplotlib only, NO Streamlit imports.

Aggregates the per-(method, seed, Γ, extension) outcome rows into comparison plots: deployed test value
vs Γ per method, with the Full-info / Best-(true-means) ceilings, plus a DGP preview panel.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless backend (Streamlit renders the Figure object)
import matplotlib.pyplot as plt

# colour = estimator FAMILY (matches index.html FAMILY_COLOR); shape distinguishes the uncertainty set
_COLORS = {
    "IPW-O-W": "#1f77b4", "IPW-O-X": "#1f77b4", "IPW-X-X": "#1f77b4",
    "Hajek-O-W": "#9467bd", "Hajek-O-X": "#9467bd",
    "DoublyRobust-X-X": "#2ca02c", "DoublyRobust-O-W": "#2ca02c", "DoublyRobust-O-X": "#2ca02c",
    "Direct-X-X": "#ff7f0e", "Kallus": "#d62728", "Oracle": "#444444",
}
_CEIL_COLORS = {"Full info": "#111111", "Best (true means)": "#2ca02c"}


def _method_color(label: str) -> str:
    return _COLORS.get(label.split("/")[0], "#444444")


def _agg_by_gamma(rows, field="realized_test"):
    """mean over (seed, extension) of ``field`` for each Γ → (gammas_sorted, means)."""
    by_g = defaultdict(list)
    for r in rows:
        v = r.get(field)
        if isinstance(v, (int, float)) and np.isfinite(v):
            by_g[round(float(r["Gamma"]), 4)].append(float(v))
    gammas = sorted(by_g)
    return np.array(gammas), np.array([np.mean(by_g[g]) for g in gammas])


def comparison_figure(results: Dict, *, field: str = "realized_test"):
    """Deployed value (default: realised test outcome) vs Γ, one line per method, ceilings as dashed lines."""
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=130)
    methods = results["methods"]
    for label in sorted(methods):
        g, y = _agg_by_gamma(methods[label], field)
        if len(g) == 0:
            continue
        if len(g) == 1:                                            # Γ-independent (IPW/PO) → flat line
            ax.axhline(y[0], color=_method_color(label), lw=2, alpha=.9, label=label, ls=":")
        else:
            ax.plot(g, y, "-o", color=_method_color(label), lw=2, ms=4, label=label)
    for name, val in (results.get("ceilings") or {}).items():
        if val is not None:
            ax.axhline(val, color=_CEIL_COLORS.get(name, "#999"), ls="--", lw=1.6, alpha=.85, label=name)
    ax.set_xlabel("Γ  (assumed sensitivity)")
    ax.set_ylabel({"realized_test": "deployed test outcome  E[Y]",
                   "exp_test": "deployed test true-mean  E[μ]"}.get(field, field))
    ax.set_title("Method comparison — higher is better")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, ncol=2, framealpha=.9)
    fig.tight_layout()
    return fig


def usage_figure(results: Dict):
    """Per-arm realised usage of each method at the matched Γ (the largest in the sweep), as grouped bars."""
    methods = results["methods"]
    n_arms = int(results.get("config", {}).get("n_arms", 2))
    labels, usages = [], []
    for label in sorted(methods):
        rows = methods[label]
        if not rows:
            continue
        gmax = max(round(float(r["Gamma"]), 4) for r in rows)
        use = [np.mean([float(r[f"train_use_{k}"]) for r in rows
                        if round(float(r["Gamma"]), 4) == gmax and f"train_use_{k}" in r])
               for k in range(n_arms)]
        labels.append(label); usages.append(use)
    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=130)
    if labels:
        usages = np.array(usages)
        w = 0.8 / n_arms
        for k in range(n_arms):
            ax.bar(np.arange(len(labels)) + k * w, usages[:, k], w, label=f"arm {k}")
        ax.set_xticks(np.arange(len(labels)) + 0.4 - w / 2)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("realised usage  (1/n)Σπ_k")
    ax.set_title("Per-arm policy usage (at matched Γ)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=.25)
    fig.tight_layout()
    return fig


def dgp_preview_figure(preview: Dict):
    """Two panels for the DGP page: treatment counts per arm + per-arm true mean μ_k distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=130)
    K = preview["n_arms"]
    ax1.bar(range(K), preview["arm_counts"], color="#4c78a8")
    ax1.set_xlabel("arm"); ax1.set_ylabel("count"); ax1.set_title("Treatment assignment")
    ax1.set_xticks(range(K))
    mu = np.asarray(preview["mu"])
    parts = ax2.violinplot([mu[:, k] for k in range(K)], showmeans=True)
    for b in parts["bodies"]:
        b.set_facecolor("#54a24b"); b.set_alpha(.6)
    ax2.set_xlabel("arm"); ax2.set_ylabel("μ_k(X)  true success prob")
    ax2.set_title("Per-arm true means (S-marginal)")
    ax2.set_xticks(range(1, K + 1)); ax2.set_xticklabels(range(K))
    fig.tight_layout()
    return fig
