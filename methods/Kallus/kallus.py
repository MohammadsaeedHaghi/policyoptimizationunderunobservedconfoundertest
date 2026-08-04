"""Kallus — PARAMETRIC Kallus & Zhou regret-minimiser (softmax policy class). Flat (no capacity).

This is the parametric counterpart of the free-π Regret-O / Hajek-OW LPs. The policy is a softmax

    π_θ(k | x) = softmax_k( θ_0·φ(x), …, θ_{K-1}·φ(x) ),   θ ∈ R^{K×p},   φ = feature map,

and we MINIMISE the worst-case **regret** of π_θ relative to the all-control baseline π₀ (treat-nobody),
over the marginal-sensitivity box (Rosenbaum Γ) — optionally intersected with per-arm Wasserstein balls:

    min_θ  sup_{w ∈ U(Γ[,ε])}  (1/n) Σ_i ( 1[T_i=0] − π_θ(T_i|X_i) ) Y_i w_i .

DERIVATION NOTE. The repository only had a parametric *value* maximiser (``srpo.parametric_multiarm``,
= parametric R-OW/R-O). This file is a NEW derivation: the parametric machinery (softmax policy, outer
subgradient loop, inner Gurobi LP, score-function envelope gradient) is kept, but the inner problem is
the worst-case REGRET (a MAX over the same uncertainty set) and the outer loop DESCENDS the regret.
There is therefore no old reference to bit-check against — it is verified by construction + sanity only.

WHY FLAT (no Capped/Uncapped). Capacity is a population-assignment constraint the free-π LP can impose;
a smooth parametric policy cannot enforce a hard per-arm capacity, so Kallus has a single (uncapped)
form. It is also parametric, so there is NO grid discretisation (no ``discretize``/``mesh``): the softmax
is a function of the raw covariates directly. Like every method it does NO statistical preprocessing —
``ips_weights`` are estimated upstream (NEVER the true propensities).

CAVEAT. For the free-π LP (Regret-O/OW) the all-control π₀ is feasible, so the worst-case regret is ≤ 0
("do-no-harm"). The softmax class cannot represent π₀ exactly (it would need θ→∞), so the parametric
worst-case regret can be slightly positive — the price of the restricted, smooth policy class.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Make the code-1.1 root importable.  parents: [0]=Kallus [1]=methods [2]=code 1.1
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.sensitivity import marginal_sensitivity_box       # the Γ box [a_i, b_i]
from common.geometry import pairwise_distance_matrix          # ground cost (Wasserstein variant only)
from common.gurobi_env import configure_gurobi_license        # solver licence
from common.hajek_regret import selfnorm_box_dinkelbach, selfnorm_wasserstein_dinkelbach  # self-normalised inner


# --------------------------------------------------------------------------- policy class
def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax (numerically stabilised)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def design_matrix(X: np.ndarray, basis: Sequence = ("affine",)) -> np.ndarray:
    """Feature map φ(X). ``("affine",)`` → [x, 1] (linear logits, default). ``("poly", D)`` and
    ``("rbf", centers, bw)`` (1-D X) give richer classes. Mirrors srpo.parametric_multiarm."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    kind = basis[0]
    if kind == "affine":
        return np.hstack([X, np.ones((X.shape[0], 1))])       # [x_1..x_d, 1]
    x = X[:, 0]
    if kind == "poly":
        deg = int(basis[1])
        return np.column_stack([x ** p for p in range(deg + 1)])
    if kind == "rbf":
        centers = np.asarray(basis[1], dtype=float); bw = float(basis[2])
        return np.column_stack([np.ones(len(x))] + [np.exp(-((x - c) ** 2) / (2.0 * bw ** 2)) for c in centers])
    raise ValueError(f"unknown basis {basis!r}")


def predict_kallus(theta: np.ndarray, X: np.ndarray, basis: Sequence = ("affine",)) -> np.ndarray:
    """π_θ(·|X) as an (n, K) simplex for any covariates X (this is how the policy is deployed —
    a smooth softmax, so NO Shapley/KNN extension is needed)."""
    theta = np.asarray(theta, dtype=float)                    # (K, p)
    return _softmax(design_matrix(X, basis) @ theta.T)        # (n, K)


# --------------------------------------------------------------------------- inner worst-case regret
def _inner_worst_case_regret_w(
    X, T, Y, pi_t, ips_weights, *, n_arms, Gamma, wasserstein, epsilon=None, D=None, debug=False,
) -> Tuple[np.ndarray, float]:
    """For a FIXED policy (its observed-arm probabilities ``pi_t``), return the per-treatment
    SELF-NORMALISED worst-case weighting ``w`` and the total worst-case REGRET:

        R̄ = Σ_t sup_{W∈U_t}  [ Σ_{i:T_i=t} r_i W_i ] / [ Σ_{i:T_i=t} W_i ],   r_i = (1[T_i=0] − pi_t[i]) Y_i,

    where U_t is the marginal-sensitivity box (built on RAW inverse weights ≥1) ∩ — when
    ``wasserstein`` — the per-arm Wasserstein ball. The returned ``w`` is the worst-case weighting
    NORMALISED per arm (Σ_{i∈I_t} w_i = 1), so it is exactly the per-unit factor in the policy
    gradient of the self-normalised value (the n-cancel is automatic). Box-only uses the fast
    Dinkelbach root-find; the Wasserstein case adds an LP transport subproblem per level.
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T).astype(int).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    pi_t = np.asarray(pi_t, dtype=float).ravel()
    w_hat = np.asarray(ips_weights, dtype=float).ravel()
    n = X.shape[0]
    a, b = marginal_sensitivity_box(w_hat, Gamma)             # per-unit Γ interval on the RAW true weight
    if np.any(a <= 0.0):
        raise ValueError("Kallus self-normalised: a_i ≤ 0 — pass RAW inverse weights (≥1), not Hájek-rescaled.")
    i_by_t = {k: np.where(T == k)[0] for k in range(n_arms)}
    coef = ((T == 0).astype(float) - pi_t) * Y                # regret coefficient r_i = (1[T=0] − π_t) Y
    if wasserstein:
        if epsilon is None or len(epsilon) != n_arms:
            raise ValueError("epsilon must have length n_arms when wasserstein=True.")
        if D is None:
            D = pairwise_distance_matrix(X)

    w_norm = np.zeros(n)
    regret = 0.0
    for k in range(n_arms):
        idx = i_by_t[k]
        if idx.size == 0:
            continue
        if wasserstein:
            val, W = selfnorm_wasserstein_dinkelbach(coef[idx], a[idx], b[idx], D[:, idx], float(epsilon[k]))
        else:
            val, W = selfnorm_box_dinkelbach(coef[idx], a[idx], b[idx])
        s = W.sum()
        w_norm[idx] = W / s if s > 1e-12 else W
        regret += val
    return w_norm, float(regret)


# --------------------------------------------------------------------------- result + fit
@dataclass
class KallusResult:
    """Fitted parametric Kallus policy + the worst-case regret it achieves."""
    theta: np.ndarray              # (K, p) softmax parameters — the learned policy
    objective_value: float         # worst-case REGRET at θ (minimised; ≤0 only if π₀ is representable)
    n_arms: int
    Gamma: float
    maximize: bool                 # True = reward convention (studies); False = the paper's loss convention
    wasserstein: bool
    basis: tuple
    epsilon: Optional[Tuple[float, ...]]
    n_iters: int
    n_restarts: int


def fit_kallus(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    ips_weights: np.ndarray,
    *,
    n_arms: int,
    Gamma: float,
    maximize: bool = True,
    wasserstein: bool = False,
    epsilon: Optional[Sequence[float]] = None,
    basis: Sequence = ("affine",),
    n_iters: int = 200,
    n_restarts: int = 6,
    eta0: float = 2.0,
    kappa: float = 0.5,
    init_scale: float = 0.25,
    normalize_grad: bool = True,
    select: str = "best",
    D: Optional[np.ndarray] = None,
    seed: int = 0,
) -> KallusResult:
    """Fit the parametric Kallus regret-minimiser by outer subgradient DESCENT on the worst-case regret.

    ``wasserstein=False`` (default) uses the box + per-arm Hájek calibration (the classic Kallus "odds"
    set, = Regret-O's set); ``wasserstein=True`` uses box ∩ per-arm Wasserstein balls (= Hajek-OW's set,
    needs ``epsilon`` length K). ``basis`` picks the feature map φ. ``ips_weights`` are the upstream
    (estimated, NEVER true) inverse weights.

    Convention (``maximize``). ``True`` (default) treats Y as a REWARD (the studies' convention): regret =
    V(π₀)−V(π_θ), minimised worst-case, so the policy improves on the all-control baseline. ``False`` treats Y as a
    LOSS — the paper's exact convention, regret = V(π_θ)−V(π₀) — implemented by negating Y (reward = −loss).

    Algorithm (faithful to Kallus & Zhou, Algorithm 1). min_θ R(θ)=max_w R(θ,w). For fixed θ the inner max_w is the
    LP above (Gurobi). The outer is subgradient descent with the EXACT policy gradient (envelope theorem):
        ∂R/∂θ = −Σ_i Yᵢ w*ᵢ ∇_θ π_θ(Tᵢ|Xᵢ),   ∇_θ π_θ(Tᵢ|Xᵢ) = π_θ(Tᵢ|Xᵢ)·(onehotᵢ − π_θ(·|Xᵢ))·φ(Xᵢ),
    i.e. grad = [ (Y⊙w*/n) ⊙ π_θ(T|X) · (onehot − π) ]ᵀ φ(X),   θ ← θ + (eta0/(k+1)^κ)·grad (ascend value ⇒
    descend regret). Note the per-unit π_θ(Tᵢ|Xᵢ) factor — this is ∇π (the paper), NOT the score-function ∇log π.
    Per restart we form the Polyak ITERATE-AVERAGE θ̄=(1/n_iters)Σ_k θ_k, and (``select="best"``, the default)
    also consider every iterate visited; across restarts we keep whichever θ attains the smallest worst-case
    regret. ``select="polyak"`` restores the average-only rule.

    OPTIMISER FIX (2026-08-04) — the defaults changed, so results move.
    ``n_iters`` 12→200, ``n_restarts`` 2→6, ``eta0`` 1.0→2.0, plus ``normalize_grad`` and ``select="best"``.
    The old settings returned a near-uniform softmax: measured on the semi-synthetic campaign at **Γ = 1**,
    where the MSM box is a single point and this reduces to ordinary IPW policy learning, a 1845-point brute
    force over the two effective parameters found regret 3–500× lower and recovered 93–98% of the available
    headroom, while the shipped fit recovered ≈ 0. Three compounding causes, all addressed above: a raw
    ∇π step that vanishes as the policy sharpens, a fixed small init that cannot reach the ‖θ‖ a decisive
    boundary needs, and Polyak-averaging iterates that never left the random init. At Γ ≫ 1 the fit was
    already near-optimal — the regret floor genuinely pins the answer to the baseline there, which is the
    method behaving as designed, not a defect.
    """
    if select not in ("best", "polyak"):
        raise ValueError("select must be 'best' or 'polyak'.")
    configure_gurobi_license()
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    T = np.asarray(T).astype(int).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    if X.ndim != 2:
        X = X.reshape(len(T), -1)
    n = X.shape[0]
    Phi = design_matrix(X, basis)                             # (n, p) feature map
    p_feat = Phi.shape[1]
    onehot = np.zeros((n, n_arms)); onehot[np.arange(n), T] = 1.0
    if wasserstein and D is None:
        D = pairwise_distance_matrix(X)
    eps = None if not wasserstein else tuple(float(e) for e in epsilon)
    # convention (3): reward (maximize=True, the studies) ⇒ regret V(π₀)−V(π); loss (maximize=False, the paper's
    # exact convention) ⇒ regret V(π)−V(π₀). Since reward = −loss, negating Y switches between the two.
    Yc = Y if maximize else -Y

    def inner(pi_t):                                          # worst-case (w*, regret) for fixed policy
        return _inner_worst_case_regret_w(X, T, Yc, pi_t, ips_weights, n_arms=n_arms, Gamma=Gamma,
                                          wasserstein=wasserstein, epsilon=eps, D=D)

    best_theta, best_regret = None, np.inf                    # MINIMISE regret ⇒ track the smallest
    # Restart scales spread over two orders of magnitude. A DECISIVE policy needs a large ‖θ‖ (for
    # the 2-arm affine case the boundary is π₁ = sigmoid(s·x + b), and a sharp threshold means
    # s ≈ 40); a small fixed init plus a decaying raw-gradient step provably cannot get there,
    # since Σ_k η₀/√k · ‖grad‖ ≈ 2η₀√n_iters·O(0.1) stays O(1). That is the whole reason this
    # optimiser used to return a near-uniform softmax and score ≈ 0 even at Γ = 1.
    scales = (np.geomspace(init_scale, init_scale * 100.0, n_restarts) if n_restarts > 1
              else np.array([init_scale], dtype=float))
    for rs in range(n_restarts):
        theta = rng.standard_normal((n_arms, p_feat)) * float(scales[rs])   # random restart
        theta_acc = np.zeros_like(theta); n_acc = 0          # accumulator for the Polyak iterate-average
        for k in range(n_iters):
            pi = _softmax(Phi @ theta.T)                     # (n, K) current policy
            pi_t = pi[np.arange(n), T]                        # observed-arm probabilities π_θ(T_i|X_i)
            try:
                w_star, regret_k = inner(pi_t)                # regret-worst-case weighting
            except RuntimeError:
                break
            # The inner solve ALREADY returns the worst-case regret at this θ, so tracking the best
            # iterate is free. Subgradient descent does not decrease the objective monotonically, and
            # the Polyak average of iterates that never left the neighbourhood of a random init is a
            # near-uniform softmax — much worse than the best point actually visited.
            if select == "best" and regret_k < best_regret:
                best_regret, best_theta = float(regret_k), theta.copy()
            # EXACT policy gradient ∇_θ π_θ(T_i|X_i) = π_θ(T_i|X_i)·(onehot − π)·φ — the per-unit π_t factor is
            # essential: this is ∇π (Kallus & Zhou Algorithm 1), NOT the score-function ∇log π.
            # w_star is the per-arm SELF-NORMALISED worst-case weight (Σ_{I_t}=1), so no 1/n factor here.
            g = (Yc * w_star * pi_t)[:, None] * (onehot - pi)   # (n, K)
            grad = g.T @ Phi                                  # (K, p): ∂V(π_θ,w*)/∂θ
            gn = float(np.linalg.norm(grad))
            if gn < 1e-12:                                    # saturated softmax ⇒ ∇π vanishes
                break
            # Normalise the step. ∇π (unlike ∇log π) carries a π_t·(1−π) factor that collapses as the
            # policy sharpens, so a raw-gradient step stalls exactly where it needs to keep moving.
            theta = theta + (eta0 / (k + 1) ** kappa) * (grad / gn if normalize_grad else grad)
            theta_acc += theta; n_acc += 1                    # accumulate the iterate for Polyak averaging
        if n_acc == 0:
            continue
        theta_bar = theta_acc / n_acc                        # Polyak ITERATE-AVERAGE (Algorithm 1, line 7)
        pi = _softmax(Phi @ theta_bar.T); pi_t = pi[np.arange(n), T]
        try:
            _, regret = inner(pi_t)                           # worst-case regret AT the averaged θ
        except RuntimeError:
            continue
        if regret < best_regret:                             # keep the lowest-regret averaged θ across restarts
            best_regret, best_theta = float(regret), theta_bar.copy()
    if best_theta is None:
        raise RuntimeError("Kallus parametric: no feasible restart.")
    return KallusResult(
        theta=best_theta, objective_value=best_regret, n_arms=int(n_arms), Gamma=float(Gamma),
        maximize=bool(maximize), wasserstein=bool(wasserstein), basis=tuple(basis), epsilon=eps,
        n_iters=int(n_iters), n_restarts=int(n_restarts),
    )
