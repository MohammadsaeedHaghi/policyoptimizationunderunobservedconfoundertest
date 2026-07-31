#!/usr/bin/env python3
"""Option 1: SUPERVISED INDEX REDUCTION -- project X onto 1-2 estimated directions.

Why. In d=8 at n=200 our methods collapsed to near never-treat (IPW-O-W 0.056 vs sharp 0.414):
Euclidean distances concentrate in high dimension, so the Lipschitz constraint and the Wasserstein
ground cost stop carrying information and the policy becomes ~200 unregularised free parameters.
Sharp's k-means cells survive because they ADAPT to the data. Reducing to a low-dimensional
supervised index restores a metric in which distances mean something, and it needs no solver
change -- it is preprocessing.

Two directions, both estimated with CROSS-FIT nuisances so the projection is not chosen using the
same residuals it will later be evaluated on:
  * beta_cate -- OLS of the cross-fit CATE-hat (muhat_1 - muhat_0) on X. Where the effect varies.
  * beta_prop -- logistic P(T=1 | X). Where the selection (hence the confounding bias) varies.
Gram-Schmidt orthogonalises the second against the first; both are unit-normalised so the reduced
covariate keeps a comparable scale to the [-1,1] originals.

FAIRNESS: the reduction is a representation choice, available to every method. Callers must give
the SAME reduced covariate to the baselines (sharp included), or the comparison measures the
reduction rather than the method.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def index_directions(X, T, Y, mu=None, n_arms=2):
    """Return (beta_cate, beta_prop_orth), unit-norm, orthogonal to each other."""
    X = np.asarray(X, float)
    if mu is None:
        import common
        mu = common.outcome_means(X, T, Y, n_arms=n_arms, cross_fit=True)
    cate_hat = mu[:, 1] - mu[:, 0]
    b1 = LinearRegression().fit(X, cate_hat).coef_
    n1 = np.linalg.norm(b1)
    b1 = b1 / n1 if n1 > 1e-12 else np.ones(X.shape[1]) / np.sqrt(X.shape[1])
    try:
        b2 = LogisticRegression(max_iter=2000).fit(X, np.asarray(T).astype(int).ravel()).coef_.ravel()
    except Exception:
        b2 = np.zeros(X.shape[1])
    b2 = b2 - (b2 @ b1) * b1                              # orthogonalise against the CATE direction
    n2 = np.linalg.norm(b2)
    b2 = b2 / n2 if n2 > 1e-12 else np.zeros(X.shape[1])
    return b1, b2


def reduce_X(X, Xte, dirs, scale=True):
    """Project train and test onto the given directions; optionally rescale to ~[-1, 1]."""
    B = np.column_stack([d for d in dirs if np.linalg.norm(d) > 1e-12])
    Z, Zte = np.asarray(X, float) @ B, np.asarray(Xte, float) @ B
    if scale:
        s = np.abs(Z).max(axis=0)
        s = np.where(s > 1e-12, s, 1.0)
        Z, Zte = Z / s, Zte / s
    return Z, Zte
