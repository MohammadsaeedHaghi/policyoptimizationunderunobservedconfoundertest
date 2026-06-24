"""Built-in DGP presets for the DGP page — the discrete-X study cases where R-OW (Wasserstein) wins.

Each preset is a **code-mode** ``DGPSpec`` whose ``generate`` is ported VERBATIM from the original study
(`code/rw_implementation/experiments/discrete/dgps.py`), so a run reproduces the published comparison
(R-OW beats R-O and Direct IPW). They are 4/3-arm, X on a 21-point grid, with an unobserved Bernoulli
confounder S that both inflates outcomes (cₖ) and shifts treatment (Rosenbaum/Tan S-channel, strength γ).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .dgp import DGPSpec, default_spec

_GEN_DIR = Path(__file__).resolve().parent / "preset_generators"


def _load_gen(stem: str) -> str:
    """Read a verbatim ported generator (preset_generators/<stem>_gen.py) as a code-mode source string."""
    return (_GEN_DIR / f"{stem}_gen.py").read_text()


def _discrete_code(K, a, b, c, alpha, beta, d) -> str:
    """Emit the exact discrete-X K-arm generator (matches discrete/dgps.py) with these coefficients inlined."""
    return f'''\
def generate(n, gamma, rng):
    """Discrete-X (21-pt grid) {K}-arm DGP with an UNOBSERVED Bernoulli confounder S (Rosenbaum/Tan
    S-channel). Ported verbatim from experiments/discrete/dgps.py — coefficients tuned so R-OW beats
    both R-O (no Wasserstein) and Direct IPW on the realised test outcome."""
    import numpy as np
    GRID = np.round(np.arange(-1.0, 1.05, 0.1), 6)               # 21 points in [-1, 1]
    a, b, c   = {a}, {b}, {c}            # outcome:  clip( sigmoid(a_k + b_k*X + c_k*S), 1e-3, 1-1e-3 )
    alpha, beta, d_s = {alpha}, {beta}, {d}   # treatment: softmax( alpha_k + beta_k*X + gamma*(S-0.5)*d_k )
    K = {K}
    def sig(z):
        z = np.asarray(z, float); return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    def parm(X, S, k):
        return np.clip(sig(a[k] + b[k] * X + c[k] * S), 1e-3, 1 - 1e-3)
    X = rng.choice(GRID, size=n, replace=True)
    S = (rng.uniform(size=n) < 0.5).astype(int)                 # unobserved confounder
    Ypot = np.empty((n, K))
    for k in range(K):
        Ypot[:, k] = (rng.uniform(size=n) < parm(X, S, k)).astype(float)
    U = np.stack([alpha[k] + beta[k] * X + gamma * (S - 0.5) * d_s[k] for k in range(K)], axis=1)
    U = U - U.max(1, keepdims=True); P = np.exp(U); P = P / P.sum(1, keepdims=True)
    T = np.array([rng.choice(K, p=P[i]) for i in range(n)])
    Y = Ypot[np.arange(n), T]
    mu = np.column_stack([0.5 * parm(X, 0, k) + 0.5 * parm(X, 1, k) for k in range(K)])  # S-marginal true means
    return X.reshape(-1, 1), T, Y, Ypot, mu
'''


def _code_preset(K: int, code: str) -> DGPSpec:
    base = default_spec(K, 1)                                    # placeholder structured fields (unused in code mode)
    return DGPSpec(n_arms=K, x=base.x, s=base.s, outcome=base.outcome,
                   treatment=base.treatment, mode="code", code=code)


CUSTOM = "— custom —"

# label -> DGPSpec (code mode). Coefficients copied from discrete/dgps.py DEFAULTS.
PRESETS: Dict[str, Optional[DGPSpec]] = {
    CUSTOM: None,
    "exp_a · discrete 3-arm (R-OW wins)": _code_preset(3, _discrete_code(
        3, [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.9, 1.8],
        [0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [-1.0, 0.0, 1.0])),
    "exp_b · discrete 4-arm (R-OW wins)": _code_preset(4, _discrete_code(
        4, [0.0, 0.0, 0.0, 0.0], [-1.2, -0.4, 0.4, 1.2], [0.0, 0.7, 1.4, 2.0],
        [0.0, 0.0, 0.0, 0.0], [-0.4, -0.15, 0.15, 0.4], [-1.0, -0.3, 0.3, 1.0])),
    "exp_c · discrete 4-arm (R-OW wins)": _code_preset(4, _discrete_code(
        4, [0.0, 0.0, 0.0, 0.0], [-1.2, -0.4, 0.4, 1.2], [0.0, 0.6, 1.2, 1.8],
        [0.0, 0.0, 0.0, 0.0], [-0.5, -0.2, 0.2, 0.5], [-1.0, -0.3, 0.3, 1.0])),
    # --- continuous-X studies (verbatim generators in preset_generators/) ---
    "conti2d · continuous 2-D 4-arm (R-OW wins)": _code_preset(4, _load_gen("conti2d")),
    "conti_binary · continuous 2-D binary (R-OW wins)": _code_preset(2, _load_gen("conti_binary")),
    "conti · continuous-X exp_c 4-arm (R-OW wins)": _code_preset(4, _load_gen("conti")),
}

# Operating settings the published comparison used (so a run matches "before").
PRESET_NOTES: Dict[str, str] = {
    "exp_a · discrete 3-arm (R-OW wins)":
        "3-arm. Recommended config: γ_true = 3, **discretize OFF** (X is already a discrete 21-pt grid), "
        "z-score OFF, compare R-OW vs R-O vs IPW. R-OW should top R-O and Direct IPW.",
    "exp_b · discrete 4-arm (R-OW wins)":
        "4-arm, stronger S-inflation gradient. Recommended config: γ_true = 3, **discretize OFF**, z-score OFF, "
        "compare R-OW vs R-O vs IPW.",
    "exp_c · discrete 4-arm (R-OW wins)":
        "4-arm. The headline discrete case. Recommended config: γ_true = 3, **discretize OFF** (X is on a "
        "21-pt grid), z-score OFF, compare R-OW vs R-O vs IPW (Uncapped, or a data-share cap).",
    "conti2d · continuous 2-D 4-arm (R-OW wins)":
        "Radial 4-arm continuous DGP, X∈[-1,1]² (clean control + 3 sector specialists, outcomes inflated by an "
        "unobserved S, mis-targeted historical assignment). R-OW (box+Wasserstein) wins — sharp radial "
        "heterogeneity makes covariate balance matter. Recommended config: γ_true = 3, **discretize ON** "
        "(snap the 2-D X, mesh ≈ 6), **z-score ON**, compare R-OW vs R-O vs IPW (+ Kallus / Oracle).",
    "conti_binary · continuous 2-D binary (R-OW wins)":
        "2-arm (binary treatment) continuous DGP, X∈[-1,1]² (risky treatment; unobserved S inflates both the "
        "treated outcome and the treatment propensity; mis-targeted assignment). R-OW wins. Recommended config: "
        "γ_true = 3, **discretize ON** (mesh ≈ 6), **z-score ON**, loose cap, compare R-OW vs R-O vs IPW (+ Kallus).",
    "conti · continuous-X exp_c 4-arm (R-OW wins)":
        "Continuous-X (d=1) 4-arm — the exp_c coefficients with X drawn continuously (so the learned policy is "
        "extended off-support via Shapley/KNN). R-OW wins (4-arm Wasserstein edge). Recommended config: "
        "γ_true = 3, **discretize ON** (snap X, mesh 6), z-score OFF (1-D), compare R-OW vs R-O vs IPW.",
}
