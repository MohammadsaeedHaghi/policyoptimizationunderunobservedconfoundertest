def generate(n, gamma, rng):
    import numpy as np
    # Inlined exp_c (4-arm) coefficients from experiments.discrete.dgps.DEFAULTS["exp_c"]
    K = 4
    a     = [0.0, 0.0, 0.0, 0.0]
    b     = [-1.2, -0.4, 0.4, 1.2]                 # true-best arm shifts left->right across X
    c     = [0.0, 0.6, 1.2, 1.8]                   # S inflates the higher arms more (asymmetric)
    alpha = [0.0, 0.0, 0.0, 0.0]
    beta  = [-0.5, -0.2, 0.2, 0.5]                 # mild treatment-by-X
    d     = [-1.0, -0.3, 0.3, 1.0]                 # high-S => pushed toward the high-c arms

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

    def _p_arm(X, S, k):
        return np.clip(_sigmoid(a[k] + b[k] * X + c[k] * S), 1e-3, 1 - 1e-3)

    # rng call order matches source exactly: X, S, then Ypot per-arm, then T per-row
    X = rng.uniform(-1.0, 1.0, size=n)                 # CONTINUOUS X on [-1, 1]
    S = (rng.uniform(size=n) < 0.5).astype(int)        # UNOBSERVED confounder
    Ypot = np.empty((n, K))
    for k in range(K):
        Ypot[:, k] = (rng.uniform(size=n) < _p_arm(X, S, k)).astype(float)
    U = np.stack([alpha[k] + beta[k] * X + gamma * (S - 0.5) * d[k] for k in range(K)], axis=1)
    P = _softmax(U)
    T = np.array([rng.choice(K, p=P[i]) for i in range(n)])
    Y = Ypot[np.arange(n), T]
    # S-marginal true means E_S[P(Y^k=1|X,S)] with S~Bern(0.5) (deployable truth)
    mu = np.column_stack([0.5 * _p_arm(X, np.zeros(n), k) + 0.5 * _p_arm(X, np.ones(n), k)
                          for k in range(K)])
    return X.reshape(-1, 1), T, Y, Ypot, mu
