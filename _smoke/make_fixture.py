"""Build a tiny shared fixture for the maximize=False end-to-end smoke check.

Exercises common end-to-end (DGP -> estimated IPW weights -> outcome means) and freezes everything
into one NPZ so each per-method smoke agent tests on IDENTICAL data with no re-derivation. NOT a real
experiment — n is tiny, one seed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]      # code 1.1/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common
from run_experiment import _synthetic_dgp        # the built-in 2-arm demo DGP (guarded by __main__)

K = 2
CAP = (1.0, 0.40)
GAMMA_TRUE = 3.0
rng = np.random.default_rng(9000)

tr = _synthetic_dgp(220, GAMMA_TRUE, rng)
te = _synthetic_dgp(400, GAMMA_TRUE, rng)

w, Pmat = common.ipw_weights_from_data(tr.X, tr.T, K)     # ESTIMATED weights (never e_true)
muhat = common.outcome_means(tr.X, tr.T, tr.Y, K)         # outcome nuisance for DoublyRobust
matched = common.matched_gamma(GAMMA_TRUE)                # the Γ the box is built at

out = _ROOT / "_smoke" / "fixture.npz"
np.savez(
    out,
    Xtr=tr.X, Ttr=tr.T, Ytr=tr.Y, Ypot_tr=tr.Ypot, mu_tr=tr.mu,
    Xte=te.X, Tte=te.T, Yte=te.Y, Ypot_te=te.Ypot, mu_te=te.mu,
    w=w, muhat=muhat, cap=np.array(CAP), n_arms=K, Gamma=float(matched),
)
print(f"fixture -> {out}")
print(f"  n_train={len(tr.T)} n_test={len(te.T)} K={K} cap={CAP} matched_Gamma={matched:.4f}")
print(f"  w: min={w.min():.3f} max={w.max():.3f} mean={w.mean():.3f}")
print(f"  muhat shape={muhat.shape}  T arm counts train={np.bincount(tr.T)}")
