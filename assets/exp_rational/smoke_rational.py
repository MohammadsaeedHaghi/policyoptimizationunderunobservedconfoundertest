#!/usr/bin/env python3
"""Smoke test: find a RATIONAL-decision-maker DGP where the O-W family wins.

Runs the uncapped method suite on one config of `dgp_rational.Cfg` at the matched Gamma* and
reports, per method, the normalised value (0 = best constant policy, 1 = the x-measurable oracle).
Success is judged on three things at once, because any one of them alone is easy to game:

  headroom        oracle - best constant must be materially > 0, or nothing can separate methods
  O-W > O-X       the transport margin at IDENTICAL L and c_eps -- the actual contribution
  O-W > X-X, naive  the robust set has to be worth having at all

Protocol copied from the working runners: outcome CENTRED before every solve (uncentred Y makes
the odds box inert), free-pi solutions deployed to the test set by the Shapley/1-NN extension, and
X-X swept over the SAME L grid so the comparison is at matched smoothness.
"""
from __future__ import annotations

import sys, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/home1/haghim/code 1.1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
sys.path.insert(0, str(HERE))

from dgp_rational import Cfg, draw, check

LGRID = [None, 3.0, 1.0]
CEPS = [1.0, 2.0]
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
_W = {}


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def _init():
    import common
    from shapley import extract_support
    _W["common"] = common; _W["extract"] = extract_support
    S = {}
    S["IPW-O-X"] = _load("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped")
    S["DoublyRobust-O-X"] = _load("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped")
    S["Hajek-O-X"] = _load("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py", "solve_hajek_o_x_uncapped")
    S["IPW-O-W"] = _load("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
    S["DoublyRobust-O-W"] = _load("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped")
    S["Hajek-O-W"] = _load("methods/Hajek-O-W/Uncapped/hajek_o_w_uncapped.py", "solve_hajek_o_w_uncapped")
    S["IPW-X-X"] = _load("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped")
    S["DoublyRobust-X-X"] = _load("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py", "solve_doublyrobust_x_x_uncapped")
    S["Direct-X-X"] = _load("methods/Direct-X-X/Uncapped/direct_x_x_uncapped.py", "solve_direct_x_x_uncapped")
    _W["S"] = S


def _shapley_fast(Xnew, sX, sp, block=200):
    Xnew = np.asarray(Xnew, float).reshape(-1, 1)
    sX = np.asarray(sX, float).reshape(-1, 1); sp = np.asarray(sp, float).ravel()
    out = np.empty(len(Xnew)); dg = np.arange(len(sX))
    for s0 in range(0, len(Xnew), block):
        xb = Xnew[s0:s0 + block]
        dd = np.sqrt(((xb[:, None, :] - sX[None, :, :]) ** 2).sum(-1))
        Sm = dd[:, :, None] + dd[:, None, :]; Sm = np.where(Sm > 0, Sm, 1.0)
        A = (dd[:, None, :] * sp[None, :, None] + dd[:, :, None] * sp[None, None, :]) / Sm
        A[:, dg, dg] = sp[None, :]
        v = A.max(axis=1).min(axis=1)
        ex = dd.min(axis=1) <= 0
        if ex.any(): v[ex] = sp[dd[ex].argmin(axis=1)]
        out[s0:s0 + block] = v
    return out


def run_cell(cfg: Cfg, seed: int, n_train: int, n_test: int):
    common = _W["common"]; S = _W["S"]
    tr = draw(n_train, 1000 + seed, cfg)
    te = draw(n_test, 90000 + seed, cfg)
    X = tr["x"].reshape(-1, 1); T = tr["T"]; Yraw = tr["Y"]
    Xte, Y0t, Y1t = te["x"], te["Y0"], te["Y1"]
    # CENTRE the training outcome -- non-centred Y makes the odds box inert
    Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)

    orc = (te["cate_cond"] > 0).astype(float)
    refs = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)),
            "never": float(np.mean(Y0t)), "all": float(np.mean(Y1t))}
    bc = max(refs["never"], refs["all"]); sc = refs["oracle"] - bc

    w, _ = common.ipw_weights_from_data(X, T, 2)
    wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)
    Dm = common.pairwise_distance_matrix(X)
    G = float(cfg.Gstar)

    def dep(vals):
        sX, sp = _W["extract"](X, vals)
        return _shapley_fast(Xte, sX, sp)

    def val(vals):
        pe = dep(vals); return float(np.mean(pe * Y1t + (1 - pe) * Y0t))

    # NAIVE: plug-in cross-fitted outcome model, no robustness at all
    naive = val((mu[:, 1] - mu[:, 0] > 0).astype(float))

    grid = {}
    for m in OX + OW:
        grid[m] = {}
        for ce in CEPS:
            if m in OX and ce != CEPS[0]:
                continue
            eps = tuple(common.tight_epsilon(Dm, T, w, 2, is_distance=True, c_eps=ce))
            ck = "%g" % ce; grid[m][ck] = {}
            for L, lk in zip(LGRID, ["inf" if L is None else "%g" % L for L in LGRID]):
                try:
                    kw = dict(n_arms=2, Gamma=G, discretize=False, lipschitz=L)
                    if m == "IPW-O-X":            r = S[m](X, T, Y, w, **kw)
                    elif m == "DoublyRobust-O-X": r = S[m](X, T, Y, w, mu, **kw)
                    elif m == "Hajek-O-X":        r = S[m](X, T, Y, wraw, maximize=True, **kw)
                    else:
                        kw.update(zscore=False, epsilon=eps)
                        if m == "IPW-O-W": r = S[m](X, T, Y, w, **kw)
                        elif m == "DoublyRobust-O-W": r = S[m](X, T, Y, w, mu, **kw)
                        else: r = S[m](X, T, Y, w, maximize=True, **kw)
                    grid[m][ck][lk] = val(r.pi[1])
                except Exception as ex:
                    print("FAIL %s ce=%s L=%s: %s" % (m, ck, lk, str(ex)[:80]), flush=True)

    xx = {}
    for m in XX:
        xx[m] = {}
        for L, lk in zip(LGRID, ["inf" if L is None else "%g" % L for L in LGRID]):
            try:
                if m == "IPW-X-X": r = S[m](X, T, Y, w, n_arms=2, discretize=False, lipschitz=L)
                elif m == "DoublyRobust-X-X": r = S[m](X, T, Y, w, mu, n_arms=2, discretize=False, lipschitz=L)
                else: r = S[m](X, T, Y, n_arms=2, discretize=False, lipschitz=L)
                xx[m][lk] = val(r.pi[1])
            except Exception as ex:
                print("FAIL %s L=%s: %s" % (m, lk, str(ex)[:80]), flush=True)

    return {"refs": refs, "bc": bc, "sc": sc, "naive": naive, "grid": grid, "xx": xx}


def main():
    ap = argparse.ArgumentParser()
    for k, v in Cfg().as_dict().items():
        ap.add_argument("--" + k, type=float, default=v)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--single-seed", type=int, default=-1,
                    help="run ONE seed and dump the raw cell; lets a SLURM array "
                         "parallelise across seeds instead of looping them")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--tag", default="cfg")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg = Cfg(**{k: getattr(a, k) for k in Cfg().as_dict()})

    ck = check(cfg)
    print("CFG %s" % json.dumps(cfg.as_dict()), flush=True)
    print("  Gamma exact %s (err %.1e) | rational %s (corr x~e %+.3f, x~CATE %+.3f) | "
          "e in [%.3f, %.3f] | headroom %+.4f | oracle treats %.2f"
          % (ck["gamma_exact"], ck["gamma_odds_max_err"], ck["rational"], ck["corr_x_e"],
             ck["corr_x_cate"], ck["e_min"], ck["e_max"], ck["headroom"],
             ck["frac_treat_oracle"]), flush=True)
    if not ck["gamma_exact"] or not ck["rational"]:
        print("  REJECT: design invariants fail"); return

    _init()
    t0 = time.time()
    if a.single_seed >= 0:
        cell = run_cell(cfg, a.single_seed, a.n_train, a.n_test)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"tag": a.tag, "cfg": cfg.as_dict(), "seed": a.single_seed,
                   "n_train": a.n_train, "cell": cell}, open(a.out, "w"))
        print("  seed %d done (%.0fs) -> %s" % (a.single_seed, time.time() - t0, a.out), flush=True)
        return
    cells = [run_cell(cfg, s, a.n_train, a.n_test) for s in range(a.seeds)]

    def nz(v, c):
        return (v - c["bc"]) / c["sc"]

    rows = {}
    for m in OX + OW:
        best = None
        for ce in cells[0]["grid"][m]:
            for lk in cells[0]["grid"][m][ce]:
                vs = [nz(c["grid"][m][ce][lk], c) for c in cells if lk in c["grid"][m].get(ce, {})]
                if not vs: continue
                if best is None or np.mean(vs) > best[0]: best = (float(np.mean(vs)), ce, lk, float(np.std(vs)))
        if best: rows[m] = {"norm": best[0], "c_eps": best[1], "L": best[2], "sd": best[3]}
    for m in XX:
        best = None
        for lk in cells[0]["xx"][m]:
            vs = [nz(c["xx"][m][lk], c) for c in cells if lk in c["xx"][m]]
            if not vs: continue
            if best is None or np.mean(vs) > best[0]: best = (float(np.mean(vs)), "-", lk, float(np.std(vs)))
        if best: rows[m] = {"norm": best[0], "c_eps": best[1], "L": best[2], "sd": best[3]}
    rows["naive"] = {"norm": float(np.mean([nz(c["naive"], c) for c in cells])), "c_eps": "-", "L": "-",
                     "sd": float(np.std([nz(c["naive"], c) for c in cells]))}

    # transport margin at IDENTICAL (c_eps, L)
    marg = {}
    for aa, bb in (("IPW-O-W", "IPW-O-X"), ("DoublyRobust-O-W", "DoublyRobust-O-X")):
        dd = []
        for ce in cells[0]["grid"][aa]:
            for lk in cells[0]["grid"][aa][ce]:
                va = [nz(c["grid"][aa][ce][lk], c) for c in cells]
                vb = [nz(c["grid"][bb]["1"][lk], c) for c in cells if "1" in c["grid"][bb]]
                if va and vb and len(va) == len(vb): dd.append(float(np.mean(va) - np.mean(vb)))
        if dd: marg[aa] = {"mean": float(np.mean(dd)), "max": float(np.max(dd))}

    hr = float(np.mean([c["sc"] for c in cells]))
    ow_best = max((rows[m]["norm"] for m in OW if m in rows), default=-9)
    beat = {k: ow_best - rows[k]["norm"] for k in list(OX) + list(XX) + ["naive"] if k in rows}
    ok = hr > 0.05 and ow_best > 0 and all(v > 0 for v in beat.values())

    print("  headroom %.4f | %d seeds x n=%d" % (hr, a.seeds, a.n_train), flush=True)
    print("  %-20s %8s %8s %6s %6s" % ("method", "norm", "sd", "c_eps", "L"), flush=True)
    for m in OW + OX + XX + ["naive"]:
        if m in rows:
            r = rows[m]
            print("  %-20s %8.3f %8.3f %6s %6s" % (m, r["norm"], r["sd"], r["c_eps"], r["L"]), flush=True)
    print("  transport margin (same L, c_eps): %s"
          % {k: round(v["mean"], 4) for k, v in marg.items()}, flush=True)
    print("  O-W best %.3f beats: %s" % (ow_best, {k: round(v, 3) for k, v in beat.items()}), flush=True)
    print("  VERDICT: %s   (%.0fs)" % ("PASS" if ok else "fail", time.time() - t0), flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"tag": a.tag, "cfg": cfg.as_dict(), "check": ck, "rows": rows,
                   "margin": marg, "headroom": hr, "ow_best": ow_best, "beat": beat,
                   "pass": bool(ok), "seeds": a.seeds, "n_train": a.n_train},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
