#!/usr/bin/env python3
"""A confounded DGP whose decision maker is RATIONAL.

WHY THIS EXISTS. In exp_gstar the CATE rises in x (mu1 - mu0 = S + 3x - 1) while the propensity
sigma(-1.5x + (1/2)ln(5) S) FALLS in x, so the units who benefit most were historically treated
least. That reads as an incompetent decision maker rather than a confounded one, and it is a fair
thing for a referee to object to.

THE FIX. The decision maker sees a private prognostic signal S that we do not, and acts rationally
on BOTH x and S:

    e(x,S) = Pr(T=1 | x,S) = sigma( a x + (1/2) ln(Gamma*) S ),      a > 0

Treatment probability RISES in x, and rises in S. Both are rational, because the benefit rises in
x and in S as well:

    Y0 = beta0 S + bx x + eps0
    Y1 = Y0 + kappa (x - x0) + delta S + eps1        =>  CATE = kappa (x - x0) + delta S

so the DM treats more where treating helps more. The analyst never sees S, which is exactly the
MSM setting -- and now the unobserved confounder is a *reason the DM was right*, not a reason it
was wrong.

Gamma* IS EXACT, BY ALGEBRA AND WITH NO CLIPPING. At any x the S-odds ratio is

    [e(x,+1)/(1-e(x,+1))] / [e(x,-1)/(1-e(x,-1))] = exp(ln Gamma*) = Gamma*,

so the matched sensitivity parameter is exactly Gamma*, as in gstar. `check()` verifies it.

COUPLING. S is drawn as Pr(S=+1 | x) = sigma(alpha x), so alpha sets corr(x, S); alpha = 0 makes S
independent of x. The x-measurable oracle treats where E[CATE | x] > 0, i.e. where
kappa (x - x0) + delta * tanh(alpha x / 2) > 0 -- monotone in x for kappa, delta, alpha >= 0, so
the optimal policy is a clean threshold and the benchmark stays easy to present.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


@dataclass
class Cfg:
    a: float = 2.0          # assignment slope in x  (>0 == rational DM)
    Gstar: float = 5.0      # exact S-odds ratio, matched Gamma
    alpha: float = 1.0      # coupling: Pr(S=+1|x) = sigmoid(alpha x)
    kappa: float = 2.0      # CATE slope in x
    x0: float = 0.0         # CATE zero crossing
    delta: float = 1.0      # CATE dependence on the hidden signal
    beta0: float = 2.0      # baseline (prognostic) dependence on S
    bx: float = 0.0         # baseline dependence on x
    bsx: float = 0.0        # x-VARYING part of the prognostic effect: Y0 gets (beta0 + bsx x) S
    sigma: float = 0.6      # outcome noise sd

    def as_dict(self):
        return asdict(self)


def draw(n, seed, cfg: Cfg):
    """One sample. Returns observed quantities and, separately, the oracle ones."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    S = np.where(rng.uniform(size=n) < _sig(cfg.alpha * x), 1.0, -1.0)
    e = _sig(cfg.a * x + 0.5 * np.log(cfg.Gstar) * S)          # NO clipping -> Gamma* exact
    T = (rng.uniform(size=n) < e).astype(int)
    # (beta0 + bsx x) S makes the SELECTION BIAS x-varying. With bsx = 0 the hidden signal shifts
    # the level only, the apparent CATE ordering in x survives, and a naive analyst still finds the
    # right threshold -- which is why bsx = 0 configs are too easy. A non-zero bsx tilts the
    # apparent CATE slope and can move the naive decision boundary off the true one.
    Y0 = (cfg.beta0 + cfg.bsx * x) * S + cfg.bx * x + rng.normal(0, cfg.sigma, n)
    cate = cfg.kappa * (x - cfg.x0) + cfg.delta * S
    Y1 = Y0 + cate + rng.normal(0, cfg.sigma, n)
    Y = np.where(T == 1, Y1, Y0)
    # what an x-measurable policy can hope to know: E[CATE | x]
    cate_cond = cfg.kappa * (x - cfg.x0) + cfg.delta * np.tanh(cfg.alpha * x / 2.0)
    return {"x": x, "S": S, "e": e, "T": T, "Y": Y, "Y0": Y0, "Y1": Y1,
            "cate": cate, "cate_cond": cate_cond}


def check(cfg: Cfg, n=200000, seed=0, tol=1e-10):
    """Verify the two properties the design rests on: Gamma* exact, and a RATIONAL assignment."""
    d = draw(n, seed, cfg)
    x = d["x"]
    ep, em = _sig(cfg.a * x + 0.5 * np.log(cfg.Gstar)), _sig(cfg.a * x - 0.5 * np.log(cfg.Gstar))
    orat = (ep / (1 - ep)) / (em / (1 - em))
    err = float(np.abs(orat - cfg.Gstar).max())
    # rationality: propensity and conditional benefit must move together in x
    q = np.argsort(x)
    ecorr = float(np.corrcoef(x, d["e"])[0, 1])
    ccorr = float(np.corrcoef(x, d["cate_cond"])[0, 1])
    bc = max(d["Y0"].mean(), d["Y1"].mean())
    orc = (d["cate_cond"] > 0).astype(float)
    orv = float(np.mean(orc * d["Y1"] + (1 - orc) * d["Y0"]))
    return {"gamma_odds_max_err": err, "gamma_exact": err < tol,
            "corr_x_e": ecorr, "corr_x_cate": ccorr,
            "rational": ecorr > 0 and ccorr > 0,
            "e_min": float(d["e"].min()), "e_max": float(d["e"].max()),
            "P_T1": float(d["T"].mean()), "frac_treat_oracle": float(orc.mean()),
            "oracle": orv, "never": float(d["Y0"].mean()), "all": float(d["Y1"].mean()),
            "headroom": orv - bc, "corr_x_S": float(np.corrcoef(x, d["S"])[0, 1])}


if __name__ == "__main__":
    c = Cfg()
    r = check(c)
    for k, v in r.items():
        print("  %-22s %s" % (k, ("%.6g" % v) if isinstance(v, float) else v))
