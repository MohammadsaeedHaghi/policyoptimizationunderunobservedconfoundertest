#!/usr/bin/env python3
"""SMOKE TEST: risk-based net-benefit CATE for the semi-synthetic campaign.

Replaces our invented tau(x) = a(sigmoid(b'x - c) - 1/2) with

    tau(x) = 2 * RRR * risk(x) - h,        risk(x) = P(Y0 = -1 | x)  (fitted, frozen)

i.e. treatment converts a constant RELATIVE fraction RRR of bad outcomes to good ones
(outcomes live in {-1,+1}, so averting one bad outcome is worth 2), at a constant burden
h. The oracle treats where risk(x) > h / (2 RRR) -- a risk THRESHOLD, the standard
clinical decision rule. Everything else in the construction (X, the index x, U = Y0 = the
real label, the gamma-confounded assignment e) is reused from the prepared population
UNCHANGED -- e does not depend on tau, so the confounding mechanism is bit-identical to
the main campaign's.

Stage A (no solver): sweep RRR x h-rule, report headroom / treated fraction / rationality.
Stage B (LPs, 3 seeds): the method suite at the matched Gamma on the RECOMMENDED cell
(RRR = 0.3, h at the median risk), n = 300 -- same verdict logic as smoke_rational.
"""
from __future__ import annotations

import sys, json, argparse, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_semisynth as R

RRRS = [0.2, 0.3, 0.5]
HQS = [0.4, 0.5, 0.6]          # h set so the risk threshold sits at this risk-quantile
RECS = [(0.3, 0.5), (0.5, 0.5)]   # run the LP stage at both: measured headroom is
                                  # 0.016-0.032 at RRR=0.3, so 0.5 is the fallback
LGRID = [None, 3.0, 1.0]
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
SIGMA_EPS = 0.1                # the spec's own outcome noise


def fit_risk(Xf, u, l2=1e-2):
    """Frozen ground-truth risk model: ridge-logistic on the FULL covariates for
    P(u = -1 | X). Returns (per-unit risk, linear score). The linear score becomes the
    new policy index -- under a risk-based CATE the natural index IS the risk score
    (the paper's own central object), not a random direction."""
    Z = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)
    y = (np.asarray(u) < 0).astype(float)
    p_dim = Z.shape[1]
    w = np.zeros(p_dim + 1)
    Za = np.column_stack([np.ones(len(y)), Z])
    for _ in range(100):
        z = Za @ w
        p = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        g = Za.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]]
        Hd = np.clip(p * (1 - p), 1e-6, None)
        H = (Za * Hd[:, None]).T @ Za / len(y) + l2 * np.diag(np.r_[0.0, np.ones(p_dim)])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    z = Za @ w
    return 1 / (1 + np.exp(-np.clip(z, -40, 40))), z


def risk_index(z):
    """The prepare_semisynth convention: standardise, scale by the 99th abs-percentile,
    clip to [-1, 1]."""
    idx = (z - z.mean()) / (z.std() + 1e-9)
    return np.clip(idx / (np.percentile(np.abs(idx), 99) + 1e-9), -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--dgp-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    npz = str(HERE / "prepared" / ("%s_g%g_d%d_pop.npz" % (a.data, a.gamma, a.dgp_seed)))
    R._init(npz)
    P = R._W["pop"]
    e, Y0, u, Xfull = P["e"], P["Y0"], P["u"], P["Xfull"]
    G = float(np.exp(2 * a.gamma))
    rng = np.random.default_rng(20260811)
    assert np.max(np.abs(Y0 - u)) == 0.0            # the construction's identity
    # e (the gamma-confounded assignment) is PER-UNIT and does not depend on tau or on
    # the index choice, so it is reused from the campaign population bit-identically.
    risk, zscore = fit_risk(Xfull, u)
    x = risk_index(zscore)
    P["x"] = x                                       # the solvers now see the risk index
    riskspread = float(np.quantile(risk, 0.95) - np.quantile(risk, 0.05))

    # ---------- Stage A: (RRR, h) pre-screen, no solver ----------
    eps = rng.normal(0, SIGMA_EPS, size=len(x))
    screen = []
    for rrr in RRRS:
        for hq in HQS:
            thr = float(np.quantile(risk, hq))
            h = 2 * rrr * thr
            tau = 2 * rrr * risk - h
            Y1n = Y0 + tau + eps
            orc = (tau > 0).astype(float)
            vo = float(np.mean(orc * Y1n + (1 - orc) * Y0))
            va, vn = float(np.mean(Y1n)), float(np.mean(Y0))
            head = vo - max(va, vn)
            rc = float(np.corrcoef(e, tau)[0, 1])
            screen.append({"RRR": rrr, "hq": hq, "h": round(h, 4),
                           "headroom": round(head, 4),
                           "oracle_treats": round(float(orc.mean()), 3),
                           "corr_e_tau": round(rc, 3),
                           "tau_range": [round(float(tau.min()), 3),
                                         round(float(tau.max()), 3)]})

    # ---------- Stage B: method suite at the candidate cells ----------
    common = R._W["common"]; S = R._W["S"]
    stageB = {}
    for rrr, hq in RECS:
      thr = float(np.quantile(risk, hq)); h = 2 * rrr * thr
      tau = 2 * rrr * risk - h
      P["Y1"] = Y0 + tau + rng.normal(0, SIGMA_EPS, size=len(x))
      P["tau"] = tau

      cells = []
      for sd in range(a.seeds):
          d = R.draw(sd)
          X, T, Yraw, Xte = d["Xtr"], d["Ttr"], d["Ytr"], d["Xte"]
          Y0t, Y1t = d["Y0te"], d["Y1te"]
          Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)
          orc = (d["cate_te"] > 0).astype(float)
          refs = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)),
                  "never": float(np.mean(Y0t)), "all": float(np.mean(Y1t))}
          bc = max(refs["never"], refs["all"]); sc = refs["oracle"] - bc

          w, _ = common.ipw_weights_from_data(X, T, 2)
          wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)
          mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)
          Dm = common.pairwise_distance_matrix(X)
          epsw = tuple(common.tight_epsilon(Dm, T, w, 2, is_distance=True, c_eps=1.0))

          def val(vals):
              pe = R._shapley_fast(Xte.reshape(-1, 1), *R._W["extract"](X, vals))
              return float(np.mean(pe * Y1t + (1 - pe) * Y0t))

          cell = {"refs": refs, "bc": bc, "sc": sc, "grid": {}, "xx": {},
                  "naive": val((mu[:, 1] - mu[:, 0] > 0).astype(float))}
          for m in OX + OW:
              cell["grid"][m] = {}
              for L, lk in zip(LGRID, ["inf", "3", "1"]):
                  try:
                      kw = dict(n_arms=2, Gamma=G, discretize=False, lipschitz=L)
                      if m == "IPW-O-X":            r = S[m](X, T, Y, w, **kw)
                      elif m == "DoublyRobust-O-X": r = S[m](X, T, Y, w, mu, **kw)
                      elif m == "Hajek-O-X":        r = S[m](X, T, Y, wraw, maximize=True, **kw)
                      else:
                          kw.update(zscore=False, epsilon=epsw)
                          if m == "IPW-O-W": r = S[m](X, T, Y, w, **kw)
                          elif m == "DoublyRobust-O-W": r = S[m](X, T, Y, w, mu, **kw)
                          else: r = S[m](X, T, Y, w, maximize=True, **kw)
                      cell["grid"][m][lk] = val(r.pi[1])
                  except Exception as ex:
                      print("FAIL %s L=%s: %s" % (m, lk, str(ex)[:80]), flush=True)
          for m in XX:
              cell["xx"][m] = {}
              for L, lk in zip(LGRID, ["inf", "3", "1"]):
                  try:
                      if m == "IPW-X-X":
                          r = S[m](X, T, Y, w, n_arms=2, discretize=False, lipschitz=L)
                      elif m == "DoublyRobust-X-X":
                          r = S[m](X, T, Y, w, mu, n_arms=2, discretize=False, lipschitz=L)
                      else:
                          r = S[m](X, T, Y, n_arms=2, discretize=False, lipschitz=L)
                      cell["xx"][m][lk] = val(r.pi[1])
                  except Exception as ex:
                      print("FAIL %s L=%s: %s" % (m, lk, str(ex)[:80]), flush=True)
          cells.append(cell)
          print("  seed %d done" % sd, flush=True)

      nz = lambda v, c: (v - c["bc"]) / c["sc"] if c["sc"] else float("nan")
      rows = {}
      for m in OX + OW:
          best = None
          for lk in ("inf", "3", "1"):
              vs = [nz(c["grid"][m][lk], c) for c in cells if lk in c["grid"][m]]
              if vs and (best is None or np.mean(vs) > best[0]):
                  best = (float(np.mean(vs)), lk, float(np.std(vs)))
          if best: rows[m] = {"norm": round(best[0], 3), "L": best[1], "sd": round(best[2], 3)}
      for m in XX:
          best = None
          for lk in ("inf", "3", "1"):
              vs = [nz(c["xx"][m][lk], c) for c in cells if lk in c["xx"][m]]
              if vs and (best is None or np.mean(vs) > best[0]):
                  best = (float(np.mean(vs)), lk, float(np.std(vs)))
          if best: rows[m] = {"norm": round(best[0], 3), "L": best[1], "sd": round(best[2], 3)}
      rows["naive"] = {"norm": round(float(np.mean([nz(c["naive"], c) for c in cells])), 3),
                       "L": "-", "sd": round(float(np.std([nz(c["naive"], c) for c in cells])), 3)}
      marg = {}
      for aa, bb in (("IPW-O-W", "IPW-O-X"), ("DoublyRobust-O-W", "DoublyRobust-O-X")):
          dd = [np.mean([nz(c["grid"][aa][lk], c) for c in cells])
                - np.mean([nz(c["grid"][bb][lk], c) for c in cells])
                for lk in ("inf", "3", "1") if all(lk in c["grid"][aa] for c in cells)]
          if dd: marg[aa] = {"mean": round(float(np.mean(dd)), 4),
                             "max": round(float(np.max(dd)), 4)}
      ow_best = max((rows[m]["norm"] for m in OW if m in rows), default=float("nan"))
      beat = {k: round(ow_best - rows[k]["norm"], 3)
              for k in OX + XX + ["naive"] if k in rows}
      hr = float(np.mean([c["sc"] for c in cells]))
      verdict = bool(hr > 0.02 and ow_best > 0 and all(v > 0 for v in beat.values()))
      stageB["RRR=%g,hq=%g" % (rrr, hq)] = {
          "h": round(h, 4), "rows": rows, "margin": marg, "ow_best": ow_best,
          "beat": beat, "headroom": round(hr, 4), "pass": verdict}

    json.dump({"dataset": a.data, "gamma": a.gamma, "matched_G": G,
               "risk_model": "ridge-logistic on FULL covariates, frozen; "
                             "policy index = the risk score",
               "risk_spread_q05_q95": round(riskspread, 4),
               "screen": screen, "stageB": stageB, "seeds": a.seeds},
              open(a.out, "w"), indent=1)
    for k, v in stageB.items():
        print("VERDICT %s [%s]: %s (headroom %.3f, O-W best %.3f, beat %s)"
              % (a.data, k, "PASS" if v["pass"] else "fail", v["headroom"],
                 v["ow_best"], v["beat"]), flush=True)


if __name__ == "__main__":
    main()
