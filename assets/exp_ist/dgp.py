"""IST recipe-D DGP adapter (experiment R1 pilot) -- REAL outcomes, injected confounding.

Contract-compatible with assets/run_owgap_lip_gamma_2d.py:
  - generate(n, seed) with seed < 1000  -> bootstrap from the BIASED (confounded) train pool
    (real X composite, real aspirin T, real 6-month Y; selection on hidden consciousness S
    injected at Lambda = 4.95 by prepare_ist.py).
  - generate(n, seed) with seed >= 1000 -> bootstrap from the UNTOUCHED RCT test pool, with
    Horvitz-Thompson pseudo potential outcomes Y1 = 2*T*Y, Y0 = 2*(1-T)*Y. Because treatment
    was randomized at exactly P(T)=1/2 and independently of X, for any policy pi:
        E[pi*Y1 + (1-pi)*Y0] = E[pi(X) Y(1) + (1-pi(X)) Y(0)]
    i.e. the runner's standard evaluation is an unbiased estimate of the true policy value
    with FULLY REAL outcomes (no synthesis anywhere).
  - oracle_policy(X): sign of the binned CATE estimated on the full unconfounded train RCT.

Run prepare_ist.py first (writes ist_prepared.npz).
"""
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
_D = np.load(HERE / "ist_prepared.npz")
_x, _S, _T, _Y = _D["x"], _D["S"], _D["T"], _D["Y"].astype(float)
_keep, _test = _D["keep"], _D["test"]
_BINS, _CATE = _D["bins"], _D["cate_sm"]
_KEEP_IDX = np.where(_keep)[0]
_TEST_IDX = np.where(_test)[0]

LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid contract only)
K = 2
CAP = (1.0, 0.3)
LAMBDA = float(_D["lam"])


def oracle_policy(X):
    X = np.asarray(X, float).ravel()
    bi = np.clip(np.digitize(X, _BINS) - 1, 0, len(_CATE) - 1)
    return (_CATE[bi] > 0).astype(float)


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    pool = _TEST_IDX if seed >= 1000 else _KEEP_IDX
    idx = rng.choice(pool, size=n, replace=True)
    X = _x[idx].reshape(-1, 1)
    T = _T[idx].astype(int)
    Y = _Y[idx]
    observed = {"X": X, "T": T, "Y": Y}
    full = {**observed, "S": _S[idx],
            "Y1": 2.0 * T * Y, "Y0": 2.0 * (1 - T) * Y}   # HT pseudo-outcomes (see docstring)
    return observed, full


def grid_truth():
    # RCT-estimated CATE interpolated onto the vestigial level grid (diagnostics only)
    cate = _CATE[np.clip(np.digitize(LEVELS, _BINS) - 1, 0, len(_CATE) - 1)]
    return dict(X=LEVELS, cate=cate, oracle=(cate > 0).astype(float))


def exact_value(pi_grid):
    raise NotImplementedError("IST has real outcomes; evaluate on test draws (HT), not a grid.")
