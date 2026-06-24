def generate(n, gamma, rng):
    import numpy as np
    # --- conti_binary DGP (K=2, 2 continuous features), ported verbatim from
    #     experiments/conti_binary/dgps.py (PARAMS inlined) ---
    K = 2  # N_ARMS
    theta0 = 0.50      # control logit (flat, clean, no S/X)
    theta1 = -1.20     # treatment base logit (NEGATIVE: risky)
    A = 3.0            # treatment-benefit slope along proj(X)
    c = 4.5            # unobserved-S effect on the TREATED outcome
    W2 = 0.6           # weight of X2 in proj = X1 + W2*X2
    alpha = [0.0, 0.0] # balanced assignment intercepts
    B = 0.5            # mis-targeted X-dependent historical assignment strength
    d = [0.0, 1.0]     # S-channel direction: control 0, treatment 1 (max|d|=1)

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

    def _proj(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X[:, 0] + W2 * X[:, 1]

    def outcome_prob(X, S, k):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        S = np.asarray(S, dtype=float)
        if k == 0:
            logit = theta0 + 0.0 * X[:, 0] + 0.0 * S      # flat & clean (broadcasts to (n,))
        else:
            logit = theta1 + A * _proj(X) + c * S
        return np.clip(_sigmoid(logit), 1e-3, 1.0 - 1e-3)

    def assignment_probs(X, S, gamma):
        proj = _proj(X)                                    # MIS-TARGETED: assignment uses -proj
        S = np.asarray(S, dtype=float)
        U0 = alpha[0] + 0.0 * proj + gamma * (S - 0.5) * d[0]
        U1 = alpha[1] - B * proj + gamma * (S - 0.5) * d[1]
        U = np.stack([U0 * np.ones_like(proj), U1 * np.ones_like(proj)], axis=1)
        return _softmax(U)

    # --- draw order matches the source generate() exactly ---
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    S = (rng.uniform(size=n) < 0.5).astype(int)
    Ypot = np.empty((n, K))
    for k in range(K):
        Ypot[:, k] = (rng.uniform(size=n) < outcome_prob(X, S, k)).astype(float)
    P = assignment_probs(X, S, gamma)
    T = np.array([rng.choice(K, p=P[i]) for i in range(n)])
    Y = Ypot[np.arange(n), T]

    # S-marginal true means mu_k(X) = 1/2 P(Y^k|X,S=0) + 1/2 P(Y^k|X,S=1)
    mu = np.stack([0.5 * outcome_prob(X, 0.0, k) + 0.5 * outcome_prob(X, 1.0, k)
                   for k in range(K)], axis=1)
    return X, T, Y, Ypot, mu
