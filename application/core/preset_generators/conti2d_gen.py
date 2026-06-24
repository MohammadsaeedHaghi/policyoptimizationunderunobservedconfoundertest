def generate(n, gamma, rng):
    import numpy as np
    K = 4
    # Specialist preferred angles (arm 0 = control has no angle); evenly spaced sectors.
    PHI = np.array([0.0, 0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    # --- PARAMS (inlined verbatim from conti2d/dgps.py) ---
    theta0     = 1.10            # control baseline logit (control mu approx 0.75): clean, solid, never-optimal fallback
    theta_base = -0.30           # specialist base logit at r=0
    A          = 5.0             # radial outcome slope (sharp X-heterogeneity)
    c          = [0.0, 2.0, 2.0, 2.0]   # unobserved-S effect on outcome (control clean; specialists confounded)
    alpha      = [0.55, 0.0, 0.0, 0.0]  # assignment intercepts (control gets a healthy share)
    B          = 1.0             # strength of the (mis-targeted) X-dependent historical assignment
    assign_rot = float(np.pi)    # MIS-TARGETED: assignment feature rotated by pi vs outcome best-arm angle
    d          = [0.0, 0.34, 0.67, 1.0] # S-channel direction (max|d|=1 => matched Rosenbaum Lambda ~ e^(gamma/2))

    def _sigmoid(z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos]); out[~pos] = ez / (1.0 + ez)
        return out

    def _softmax(U):
        U = U - U.max(axis=1, keepdims=True)
        E = np.exp(U)
        return E / E.sum(axis=1, keepdims=True)

    def _radial_feat(Xin, rot=0.0):
        # (n, K) feature feat_k = r*cos(phi - phi_k - rot) for specialists, 0 for control (k=0).
        Xin = np.atleast_2d(np.asarray(Xin, dtype=float))
        r = np.sqrt((Xin ** 2).sum(axis=1))
        phi = np.arctan2(Xin[:, 1], Xin[:, 0])
        feat = np.zeros((Xin.shape[0], K))
        for k in range(1, K):
            feat[:, k] = r * np.cos(phi - PHI[k] - rot)
        return feat

    def outcome_prob(Xin, Sin, k):
        # P(Y^k=1 | X, S) for arm k (outcome uses rot=0, the true best-arm geometry).
        feat = _radial_feat(Xin)[:, k]
        Sin = np.asarray(Sin, dtype=float)
        if k == 0:
            logit = theta0 + c[0] * Sin + 0.0 * feat   # 0*feat broadcasts to (n,)
        else:
            logit = theta_base + A * feat + c[k] * Sin
        return np.clip(_sigmoid(logit), 1e-3, 1.0 - 1e-3)

    def assignment_probs(Xin, Sin, g):
        # (n, K) P(T=k | X, S) softmax; assignment uses the rotated (mis-targeted) feature.
        feat = _radial_feat(Xin, assign_rot)
        Sin = np.asarray(Sin, dtype=float)
        U = np.stack([alpha[k] + B * feat[:, k] + g * (Sin - 0.5) * d[k]
                      for k in range(K)], axis=1)
        return _softmax(U)

    # --- rng call order matches the source exactly ---
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    S = (rng.uniform(size=n) < 0.5).astype(int)
    Ypot = np.empty((n, K))
    for k in range(K):
        Ypot[:, k] = (rng.uniform(size=n) < outcome_prob(X, S, k)).astype(float)
    P = assignment_probs(X, S, gamma)              # gamma is the Rosenbaum S-channel strength
    T = np.array([rng.choice(K, p=P[i]) for i in range(n)])
    Y = Ypot[np.arange(n), T]
    # S-marginal true means mu_k(X) = 1/2 P(Y^k|X,S=0) + 1/2 P(Y^k|X,S=1) (deployable truth; S unobserved)
    mu = np.stack([0.5 * outcome_prob(X, 0.0, k) + 0.5 * outcome_prob(X, 1.0, k)
                   for k in range(K)], axis=1)
    return X, T, Y, Ypot, mu
