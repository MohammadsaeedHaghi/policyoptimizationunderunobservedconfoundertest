#!/usr/bin/env python3
"""Hess, Frauen, Melnychuk & Feuerriegel (ICLR 2026), arXiv 2502.13022 -- implemented VERBATIM.

"Efficient and Sharp Off-Policy Learning under Unobserved Confounding". This is their
semi-parametrically efficient one-step estimator of the SHARP bound on the policy value under
the MSM (their Theorem 4.3 / Eq. 15), not a plug-in approximation of it.

WHAT WE HAD BEFORE, AND WHY IT WAS NOT THIS
    assets/grand/run_sharp_all.py computes a per-cell two-point Dorn-Guo bound from empirical
    bin means. That is the PLUG-IN estimand only -- exactly the "simple plug-in approach" the
    paper says it outperforms. It has no efficient influence function correction, no conditional
    quantile, and no cross-fitting. Everything below follows the paper.

THE PAPER, TERM BY TERM (their notation; their convention is LOWER Y IS BETTER, so they bound
the value from ABOVE and MINIMISE it):

  alpha^+ = Gamma / (1 + Gamma)                                        (Definition 4.2)
  b^+     = 1 - Gamma,          b^-     = 1 - 1/Gamma
  c^+(a,x)= b^+ e(a,x) + Gamma, c^-(a,x)= b^- e(a,x) + 1/Gamma
  F^-1_{x,a}(alpha^+)                          conditional quantile of Y | X=x, A=a
  Delta^+(y,a,x)     = 1{y <= F^-1_{x,a}(alpha^+)}
  Deltabar^+(y,a,x)  = 1{y >= F^-1_{x,a}(alpha^+)}
  mu^+(a,x)          = E[Y Delta^+    | X=x, A=a]        (lower-tail truncated mean)
  mubar^+(a,x)       = E[Y Deltabar^+ | X=x, A=a]        (upper-tail truncated mean)
  Q^{+,*}(a,x)       = c^-(a,x) mu^+(a,x) + c^+(a,x) mubar^+(a,x)      (Eq. 9)

  Vhat^{+,*}(pi) = P_n { sum_a pi_{a,X} [ Qhat^{+,*}_{a,X} - ehat_{a,X}( b^- muhat^+_{a,X}
                                                          + b^+ mubarhat^+_{a,X} ) ]
                         + pi_{A,X} ( b^- muhat^+_{A,X} + b^+ mubarhat^+_{A,X} )
                         + (pi_{A,X} / ehat_{A,X}) [ (chat^-_{A,X} - chat^+_{A,X})
                               ( Fhat^-1_{X,A}(alpha^+) (Deltahat^+_{Y,A,X} - alpha^+) )
                             + chat^-_{A,X} ( Y Deltahat^+_{Y,A,X}    - muhat^+_{A,X} )
                             + chat^+_{A,X} ( Y Deltabarhat^+_{Y,A,X} - mubarhat^+_{A,X} ) ] }
                                                                        (Eq. 15)

Because Eq. 15 is LINEAR in pi, it is a per-unit score: Vhat = (1/n) sum_i sum_a pi(a|X_i) g_i(a),
so `scores` below returns g and any policy class can be optimised against it. Algorithm 1's
sample split / cross-fitting for the nuisances is implemented in `fit_scores`.

SIGN CONVENTION. The paper minimises (lower Y better). Our pipeline maximises (higher Y better).
The reduction is exact: min over p of E[Y-value] = -(sup over p of E[(-Y)-value]), so we pass
-Y to the paper's estimator and then negate. `maximize=True` (our default) does this; the
formulas themselves are untouched.

VALIDATION (test_gamma1_is_aipw): at Gamma = 1 we have b^+ = b^- = 0, c^+ = c^- = 1 and
alpha^+ = 1/2, so Q^{+,*} collapses to E[Y|x,a] and Eq. 15 reduces EXACTLY to the AIPW/doubly
robust score Q(a,x) + 1{A=a}/e (Y - Q(A,X)). That identity is checked numerically.
"""
import numpy as np

__all__ = ["fit_scores", "sharp_hess_policy", "nuisances_knn"]


def _knn_idx(x_tr, x_ev, k):
    """Indices of the k nearest training points for each evaluation point (Euclidean)."""
    x_tr = np.atleast_2d(np.asarray(x_tr, float))
    x_ev = np.atleast_2d(np.asarray(x_ev, float))
    if x_tr.shape[0] == 1 and x_tr.shape[1] != x_ev.shape[1]:
        x_tr = x_tr.T
    d = np.sqrt(((x_ev[:, None, :] - x_tr[None, :, :]) ** 2).sum(-1))
    k = int(min(k, x_tr.shape[0]))
    return np.argsort(d, axis=1)[:, :k]


def nuisances_knn(X_tr, T_tr, Y_tr, X_ev, alpha_plus, k=50, clip=0.02):
    """Nuisances eta = {e(a,x), F^-1_{x,a}(alpha+), mu^+(a,x), mubar^+(a,x)} by k-NN.

    The paper uses neural instantiations; any flexible learner is admissible -- what matters for
    Theorem 4.3 is that the nuisances are fit on a SEPARATE fold (Algorithm 1). k-NN is used here
    because these benchmarks are low-dimensional and it needs no tuning, which keeps the
    comparison about the estimator rather than about network hyperparameters.
    """
    X_tr = np.atleast_2d(np.asarray(X_tr, float))
    if X_tr.shape[0] == 1: X_tr = X_tr.T
    X_ev = np.atleast_2d(np.asarray(X_ev, float))
    if X_ev.shape[0] == 1: X_ev = X_ev.T
    T_tr = np.asarray(T_tr).astype(int).ravel(); Y_tr = np.asarray(Y_tr, float).ravel()
    n_ev = X_ev.shape[0]
    e = np.zeros((n_ev, 2)); q = np.zeros((n_ev, 2))
    mu_lo = np.zeros((n_ev, 2)); mu_hi = np.zeros((n_ev, 2))
    nn_all = _knn_idx(X_tr, X_ev, k)
    for i in range(n_ev):
        nb = nn_all[i]
        e[i, 1] = float(np.clip(T_tr[nb].mean(), clip, 1 - clip)); e[i, 0] = 1.0 - e[i, 1]
    for a in (0, 1):
        idx_a = np.where(T_tr == a)[0]
        if len(idx_a) == 0:
            continue
        nn_a = _knn_idx(X_tr[idx_a], X_ev, k)
        for i in range(n_ev):
            ys = Y_tr[idx_a[nn_a[i]]]
            qa = float(np.quantile(ys, alpha_plus))
            q[i, a] = qa
            lo = ys[ys <= qa]; hi = ys[ys >= qa]
            mu_lo[i, a] = float(lo.sum() / len(ys)) if len(ys) else 0.0   # E[Y 1{Y<=q}]
            mu_hi[i, a] = float(hi.sum() / len(ys)) if len(ys) else 0.0   # E[Y 1{Y>=q}]
    return {"e": e, "q": q, "mu_lo": mu_lo, "mu_hi": mu_hi}


def scores(Y, T, eta, Gamma):
    """Per-unit, per-arm scores g_i(a) with Vhat^{+,*}(pi) = (1/n) sum_i sum_a pi(a|X_i) g_i(a).

    This is Eq. 15 written out; nothing is dropped or approximated.
    """
    Y = np.asarray(Y, float).ravel(); T = np.asarray(T).astype(int).ravel()
    e, q, mu_lo, mu_hi = eta["e"], eta["q"], eta["mu_lo"], eta["mu_hi"]
    n = len(Y)
    G = float(Gamma)
    a_plus = G / (1.0 + G)                      # alpha^+
    b_plus = 1.0 - G                            # b^+
    b_minus = 1.0 - 1.0 / G                     # b^-
    c_plus = b_plus * e + G                     # c^+(a,x), shape (n, 2)
    c_minus = b_minus * e + 1.0 / G             # c^-(a,x)
    Q = c_minus * mu_lo + c_plus * mu_hi        # Q^{+,*}(a,x)          (Eq. 9)

    g = Q - e * (b_minus * mu_lo + b_plus * mu_hi)        # first bracket, all arms
    rows = np.arange(n)
    # factual-arm terms
    g[rows, T] += (b_minus * mu_lo[rows, T] + b_plus * mu_hi[rows, T])
    qf = q[rows, T]
    d_lo = (Y <= qf).astype(float)                        # Delta^+
    d_hi = (Y >= qf).astype(float)                        # Deltabar^+
    corr = ((c_minus[rows, T] - c_plus[rows, T]) * (qf * (d_lo - a_plus))
            + c_minus[rows, T] * (Y * d_lo - mu_lo[rows, T])
            + c_plus[rows, T] * (Y * d_hi - mu_hi[rows, T]))
    g[rows, T] += corr / e[rows, T]
    return g


def fit_scores(X, T, Y, Gamma, k=50, n_folds=2, maximize=True, seed=0, standardize=True):
    """Cross-fitted Eq.-15 scores (Algorithm 1, with cross-fitting rather than a single split).

    maximize=True (our pipeline's convention, higher Y better): the paper's estimator is applied
    to -Y and the resulting scores negated, so that the returned g is a score to MAXIMISE and
    equals the sharp LOWER bound of the value on our scale.
    """
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] == 1: X = X.T
    T = np.asarray(T).astype(int).ravel()
    Ypaper = -np.asarray(Y, float).ravel() if maximize else np.asarray(Y, float).ravel()
    # THE AUTHORS STANDARDISE THE OUTCOME (their src/data_gen.py: Y = (Y - Y.mean())/Y.std()).
    # The sharp bound is equivariant under an affine rescaling of Y, because the weights satisfy
    # c^- alpha^+ + c^+ (1 - alpha^+) = 1, so Q^{+,*} -> (Q^{+,*} - m)/s and the OPTIMAL POLICY is
    # unchanged. What changes is the estimator's finite-sample variance: the one-step correction
    # carries Y/e, so an outcome on a large scale (gstar reaches |Y| = 14.5 against a CATE of ~1)
    # makes that term swamp the signal. Standardising is therefore part of their method, not a
    # cosmetic step, and omitting it was the single biggest discrepancy with their code.
    ysd = float(np.std(Ypaper)) if standardize else 1.0
    ymu = float(np.mean(Ypaper)) if standardize else 0.0
    if ysd <= 0: ysd = 1.0
    Ypaper = (Ypaper - ymu) / ysd
    n = X.shape[0]
    a_plus = float(Gamma) / (1.0 + float(Gamma))
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    g = np.zeros((n, 2))
    for f in folds:                                   # nuisances on the OTHER folds only
        tr = np.setdiff1d(np.arange(n), f)
        eta = nuisances_knn(X[tr], T[tr], Ypaper[tr], X[f], a_plus, k=k)
        g[f] = scores(Ypaper[f], T[f], eta, Gamma)
    return -g if maximize else g


def sharp_hess_policy(X, T, Y, Gamma, k=50, n_folds=2, cap=None, maximize=True, seed=0, standardize=True):
    """The policy that optimises the paper's objective.

    Eq. 15 is linear in pi, so over ALL measurable policies the optimum is the pointwise rule
    (treat iff the treated-arm score beats the control-arm score); with a capacity constraint it
    is the greedy top-cap by score gap, which is exact for a linear objective under a mass budget.
    """
    g = fit_scores(X, T, Y, Gamma, k=k, n_folds=n_folds, maximize=maximize, seed=seed, standardize=standardize)
    gap = g[:, 1] - g[:, 0]                            # >0 => treating is better (maximise)
    if cap is None:
        return (gap > 0).astype(float), g
    pi = np.zeros(len(gap))
    kk = int(np.floor(float(cap) * len(gap)))
    for i in np.argsort(-gap)[:kk]:
        if gap[i] > 0: pi[i] = 1.0
    return pi, g


def test_gamma1_is_aipw(n=4000, seed=0, tol=1e-8):
    """At Gamma = 1 the estimator must reduce EXACTLY to the AIPW/DR score."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 1))
    e_true = 1 / (1 + np.exp(-(0.8 * X[:, 0])))
    T = (rng.uniform(size=n) < e_true).astype(int)
    Y = X[:, 0] + 0.7 * T + rng.normal(0, 0.5, n)
    a_plus = 0.5
    eta = nuisances_knn(X, T, Y, X, a_plus, k=80)
    g = scores(Y, T, eta, 1.0)
    # AIPW with the same nuisances: Q(a,x) = mu_lo + mu_hi at Gamma=1 (alpha+ = 1/2)
    Q = eta["mu_lo"] + eta["mu_hi"]
    rows = np.arange(n)
    aipw = Q.copy()
    aipw[rows, T] += (Y - Q[rows, T]) / eta["e"][rows, T]
    err = float(np.abs(g - aipw).max())
    print("Gamma=1 vs AIPW: max abs diff = %.3e  -> %s" % (err, "PASS" if err < tol else "FAIL"))
    return err < tol


if __name__ == "__main__":
    test_gamma1_is_aipw()


def learn_policy_parametric(X, g, n_iter=800, lr=2.0, restarts=6, seed=0, maximize=True, hidden=0,
                            decay=True):
    """Algorithm 1, steps 4-8: optimise a PARAMETRIC policy class by gradient descent.

    The paper learns pi_theta over a parametric class (they use neural networks) rather than
    taking the pointwise argmin of the score, and this matters: a pointwise rule on n training
    points has to be deployed by nearest-neighbour, which is far rougher than the smooth policy
    a parametric class gives. Objective (their Eq. 15, linear in pi):
        (1/n) sum_i [ pi(1|X_i) g_i(1) + (1 - pi(1|X_i)) g_i(0) ]
    minimised in the paper's convention; with maximize=True we ascend instead, having already
    negated the scores. Logistic class by default; hidden>0 gives a 1-hidden-layer tanh MLP,
    closer to their neural instantiation.

    NOT the pointwise argmax. Eq. 15 is linear in pi, so `sharp_hess_policy` returns the exact
    optimum over ALL policies -- and that optimum OVERFITS badly: measured here it attains an
    Eq.-15 objective of 7.41 against this class's -0.08 while its true normalised value is -0.054
    against +0.790. The paper's guarantee (Theorem 5.1) is stated over a class of bounded
    Rademacher complexity for exactly this reason, so the parametric route is the faithful one.
    """
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] == 1: X = X.T
    n, d = X.shape
    Z = np.hstack([X, np.ones((n, 1))])
    gap = np.asarray(g)[:, 1] - np.asarray(g)[:, 0]        # coefficient on pi(1|x)
    base = float(np.mean(np.asarray(g)[:, 0]))
    sgn = 1.0 if maximize else -1.0
    rng = np.random.default_rng(seed)
    best, best_val = None, -np.inf

    def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    # OPTIMISER FIX (2026-08-04), the same defect found in methods/Kallus/kallus.py. The step below
    # is a raw ∇π, which carries a p·(1−p) factor that collapses exactly as the policy sharpens, so
    # a fixed lr from a fixed small init cannot reach the ‖θ‖ a decisive boundary needs. Measured
    # against a brute-force grid over this very class (2 parameters when x is scalar), the old
    # settings lost up to 0.96 normalised value — e.g. german_credit γ=1 scored −0.076 where the
    # class optimum is +0.886. Three changes: normalise the step, spread the restart scales, and
    # keep the best ITERATE rather than each restart's last one (the objective is available every
    # iteration for free, and ascent on this objective is not monotone once the step is normalised).
    scales = (np.geomspace(1.0, 100.0, restarts) if restarts > 1 else np.array([1.0]))

    def track(params_, p_):
        nonlocal best, best_val
        v = sgn * float(np.mean(gap * p_)) + base
        if v > best_val:
            best_val, best = v, (tuple(np.copy(q) for q in params_) if isinstance(params_, tuple)
                                 else np.copy(params_), hidden)

    for r in range(restarts):
        sc0 = float(scales[r])
        if hidden > 0:
            W1 = rng.normal(0, sc0, size=(Z.shape[1], hidden)); W2 = rng.normal(0, sc0, size=hidden + 1)
            for k in range(n_iter):
                H = np.tanh(Z @ W1); Hb = np.hstack([H, np.ones((n, 1))])
                p = sig(Hb @ W2)
                track((W1, W2), p)
                dobj = sgn * gap * p * (1 - p) / n
                gW2 = Hb.T @ dobj
                gH = np.outer(dobj, W2[:hidden]) * (1 - H ** 2)
                gW1 = Z.T @ gH
                gn = float(np.sqrt((gW1 ** 2).sum() + (gW2 ** 2).sum()))
                if gn < 1e-12: break
                st = lr / (k + 1) ** 0.5 if decay else lr
                W2 = W2 + st * gW2 / gn; W1 = W1 + st * gW1 / gn
            H = np.tanh(Z @ W1); track((W1, W2), sig(np.hstack([H, np.ones((n, 1))]) @ W2))
        else:
            th = rng.normal(0, sc0, size=Z.shape[1])
            for k in range(n_iter):
                p = sig(Z @ th)
                track(th, p)
                grad = Z.T @ (sgn * gap * p * (1 - p)) / n
                gn = float(np.linalg.norm(grad))
                if gn < 1e-12: break
                th = th + (lr / (k + 1) ** 0.5 if decay else lr) * grad / gn
            track(th, sig(Z @ th))
    return best


def apply_policy(params_hidden, Xnew):
    """Evaluate a policy returned by learn_policy_parametric at new covariates."""
    params, hidden = params_hidden
    Xn = np.atleast_2d(np.asarray(Xnew, float))
    if Xn.shape[0] == 1: Xn = Xn.T
    Zn = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
    def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    if hidden > 0:
        W1, W2 = params
        H = np.tanh(Zn @ W1)
        return sig(np.hstack([H, np.ones((H.shape[0], 1))]) @ W2)
    return sig(Zn @ params)
