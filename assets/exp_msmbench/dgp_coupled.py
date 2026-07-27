"""msmbench-coupled: the KMZ'19 MSM benchmark with ONE disclosed change -- an X-U coupling knob.

    U ~ Bern(sigma(BETA * X))          (BETA = 0 recovers the original benchmark EXACTLY:
                                        U ~ Bern(1/2) independent of X; marginal P(U=1) = 1/2
                                        for every BETA by the symmetry of X ~ Unif[-2,2])

Everything else is byte-identical to dgp.py: same outcome function, same MSM-extremal
propensity at Gamma* = 4.95, same CATE (U cancels), same oracle. The knob moves the DGP along
the diagnostic map's coupling axis: corr(x, S) ~ 0 / 0.55 / 0.75 / 0.9 at BETA = 0 / 2.5 / 5 / 10.
Declared prediction: the O-W margin over box-only O-X grows with BETA -- the alpha-sweep
argument demonstrated ON the literature's own DGP family, with the original point kept as the
honest beta=0 anchor.

BETA is overridden by the wrapper files dgp_beta*.py (importlib pattern, as exp_owgap_v2's
cap wrappers).
"""
import numpy as np

GSTAR = 4.95
BETA = 10.0
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid contract only)
K = 2
CAP = (1.0, 0.3)
NOISE = 1.0
import sys as _sys, importlib.util as _ilu
from pathlib import Path as _P
_sp = _ilu.spec_from_file_location("kmz_base", str(_P(__file__).resolve().parent / "dgp.py"))
_base = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_base)

e_nom = _base.e_nom
mu0, mu1, cate, oracle_policy = _base.mu0, _base.mu1, _base.cate, _base.oracle_policy


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

def p_s1(x):
    # coupling knob: X = 2x, U ~ Bern(sigma(BETA * x)) in pipeline units
    return _sig(BETA * np.asarray(x, float))

def propensity(x, S): return _base.propensity(x, S)     # unchanged MSM extremal


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    S = np.where(rng.uniform(size=n) < p_s1(x), 1.0, -1.0)
    e = propensity(x, S)
    T = (rng.uniform(size=n) < e).astype(int)
    Y0 = mu0(x, S) + rng.normal(0, NOISE, size=n)
    Y1 = mu1(x, S) + rng.normal(0, NOISE, size=n)
    Y = np.where(T == 1, Y1, Y0)
    observed = {"X": x.reshape(-1, 1), "T": T, "Y": Y}
    full = {**observed, "S": S, "Y1": Y1, "Y0": Y0, "e_true": e}
    return observed, full


def grid_truth():
    X = LEVELS
    return dict(X=X, cate=cate(X), oracle=oracle_policy(X))
