"""Shared pytest fixtures/helpers for the code-1.1 test suite.

Adds the code-1.1 root to sys.path, loads the hyphenated method solvers by path (importlib +
sys.modules registration, the same recipe the runner uses), and builds tiny K-arm datasets with
ESTIMATED weights + outcome means (never e_true).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]               # code 1.1/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import common  # noqa: E402  (after sys.path insert)

# method name -> (file stem, fn stem); mirrors run_experiment.REGISTRY (Oracle added; Kallus is parametric)
_FILES = {
    "IPW-O-W": ("ipw_o_w", "solve_ipw_o_w"), "IPW-O-X": ("ipw_o_x", "solve_ipw_o_x"),
    "Hajek-O-W": ("hajek_o_w", "solve_hajek_o_w"), "Hajek-O-X": ("hajek_o_x", "solve_hajek_o_x"),
    "IPW-X-X": ("ipw_x_x", "solve_ipw_x_x"), "Direct-X-X": ("direct_x_x", "solve_direct_x_x"),
    "DoublyRobust-X-X": ("doublyrobust_x_x", "solve_doublyrobust_x_x"), "Oracle": ("oracle", "solve_oracle"),
}


def load_module(name, relpath):
    """Load a module by path and register it before exec (so @dataclass + future-annotations resolve)."""
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_solver(method, variant):
    """Return the solver function for an LP method's variant (e.g. ('R-OW','Capped'))."""
    stem, fnstem = _FILES[method]
    csuf = variant.lower()
    f = ROOT / "methods" / method / variant / f"{stem}_{csuf}.py"
    return getattr(load_module(f"{stem}_{csuf}_t", f), f"{fnstem}_{csuf}")


def _make_dataset(n, K, seed):
    """A tiny confounded K-arm DGP (2-D X, unobserved S) + estimated weights/means. All arms covered."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    S = (rng.uniform(size=n) < 0.5).astype(float)
    Wmat = rng.normal(size=(K, 2))
    scores = X @ Wmat.T                                   # (n, K)
    probs = 1.0 / (1.0 + np.exp(-scores))
    Ypot = (rng.uniform(size=(n, K)) < probs).astype(float)
    mu = probs                                            # true arm means μ_k(X)
    logits = 0.5 * scores
    logits[:, 1:] += 1.5 * (S[:, None] - 0.5)             # S confounds the non-control arms
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    T = np.array([rng.choice(K, p=p[i]) for i in range(n)])
    for k in range(K):                                    # guarantee >=2 obs per arm (solvers require it)
        if (T == k).sum() < 2:
            T[rng.choice(n, size=2, replace=False)] = k
    Y = Ypot[np.arange(n), T]
    w, _ = common.ipw_weights_from_data(X, T, K)          # ESTIMATED weights (never e_true)
    muhat = common.outcome_means(X, T, Y, K)
    cap = tuple([1.0] + [max(0.45, 1.0 / K + 0.1)] * (K - 1))
    return dict(X=X, T=T, Y=Y, Ypot=Ypot, mu=mu, w=w, muhat=muhat, K=K, cap=cap,
                Gamma=common.matched_gamma(3.0))


@pytest.fixture(scope="session")
def k2():
    return _make_dataset(80, 2, 7)


@pytest.fixture(scope="session")
def k3():
    return _make_dataset(90, 3, 11)


@pytest.fixture
def loader():
    return load_solver


@pytest.fixture
def mod_loader():
    return load_module
