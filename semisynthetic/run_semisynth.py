#!/usr/bin/env python3
"""One (gamma, seed) cell of the risk-score-paper semi-synthetic campaign.

Solves ONLY at the matched Gamma = e^{2 gamma} (exact for this DGP, since pi0 is never clipped),
sweeping c_eps x L x rationality x methods. One job per cell.

Adapted from `RCT datasets/run_rct.py`: same _shapley_fast deployment, same outcome-centring
protocol, same solver call signatures. Differences: a single Gamma, and the C4 `rationality`
switch on the four solvers that support it.
"""
import sys, os, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/home1/haghim/code 1.1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
sys.path.insert(0, str(ROOT / "methods" / "SharpHess"))

LGRID = [None, 3.0, 1.0]
CEPS = [1.0, 1.025, 2.0]                 # 1.025 is the paper's own epsilon multiplier
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
RAT_OK = ["IPW-O-W", "DoublyRobust-O-W"]   # C4 reported for the O-W family only
N_TRAIN, N_TEST = 300, 4000
PGRID = np.linspace(-1.0, 1.0, 41)
_W = {}


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def _init(npz_path):
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
    _W["kallus"] = _load("methods/Kallus/kallus.py", "fit_kallus")
    _W["kpred"] = _load("methods/Kallus/kallus.py", "predict_kallus")
    import sharp_hess
    _W["hess"] = sharp_hess
    z = np.load(npz_path)
    _W["pop"] = {k: z[k] for k in z.files}


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


def draw(seed):
    """Redraw the ASSIGNMENT and the train/test split. Y0/Y1 are the DGP's own, fixed."""
    P = _W["pop"]; rng = np.random.default_rng(10_000 + seed)
    x, e, Y0, Y1 = P["x"], P["e"], P["Y0"], P["Y1"]
    n = len(x)
    T = (rng.uniform(size=n) < e).astype(int)
    Yo = np.where(T == 1, Y1, Y0)
    perm = rng.permutation(n)
    tr, te = perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + N_TEST]
    return dict(Xtr=x[tr].reshape(-1, 1), Ttr=T[tr], Ytr=Yo[tr],
                Xte=x[te], Y0te=Y0[te], Y1te=Y1[te], cate_te=(Y1 - Y0)[te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)      # split/assignment seed
    ap.add_argument("--dgp-seed", type=int, default=0)      # which (a,b,c,lam) draw
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = float(np.exp(2 * a.gamma))                     # the MATCHED Gamma; no sweep
    npz = str(HERE / "prepared" / ("%s_g%g_d%d_pop.npz" % (a.data, a.gamma, a.dgp_seed)))
    meta = [m for m in json.loads((HERE / "prepared" / "_index.json").read_text())
            if m["dataset"] == a.data and abs(m["gamma"] - a.gamma) < 1e-12
            and m.get("dgp_seed", 0) == a.dgp_seed][0]
    t0 = time.time()
    _init(npz)

    d = draw(a.seed)
    X, T, Yraw = d["Xtr"], d["Ttr"], d["Ytr"]
    Xte, Y1t, Y0t = d["Xte"], d["Y1te"], d["Y0te"]
    common = _W["common"]; S = _W["S"]; K = 2

    # CENTRE-AND-SCALE the training outcome. Y0 in {-1,+1} looks centred but Y1 = Y0 + tau is
    # not; without this the LP degenerates to pi = 1{Y>0} and becomes Gamma-inert. Evaluation
    # always uses the RAW test potential outcomes, so reported values stay on the data scale.
    _ym, _ys = float(np.mean(Yraw)), float(np.std(Yraw)) or 1.0
    Y = (Yraw - _ym) / _ys

    def dep(Xnew, vals):
        sX, sp = _W["extract"](X, vals)
        return _shapley_fast(Xnew, sX, sp)

    def val(vals):
        pe = dep(Xte, vals); return float(np.mean(pe * Y1t + (1 - pe) * Y0t))

    w, _ = common.ipw_weights_from_data(X, T, K)
    wraw, _ = common.ipw_weights_from_data(X, T, K, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(X)
    Lk = ["inf" if L is None else "%g" % L for L in LGRID]

    def call(m, g, L, eps, rat):
        kw = dict(n_arms=K, Gamma=g, discretize=False, lipschitz=L)
        rk = dict(rationality=rat) if m in RAT_OK else {}
        if m == "IPW-O-X": return S[m](X, T, Y, w, **kw, **rk)
        if m == "DoublyRobust-O-X": return S[m](X, T, Y, w, mu, **kw, **rk)
        if m == "Hajek-O-X": return S[m](X, T, Y, wraw, maximize=True, **kw)
        kw.update(zscore=False, epsilon=eps)
        if m == "IPW-O-W": return S[m](X, T, Y, w, **kw, **rk)
        if m == "DoublyRobust-O-W": return S[m](X, T, Y, w, mu, **kw, **rk)
        return S[m](X, T, Y, w, maximize=True, **kw)

    grid = {}          # grid[rat][method][ceps][L] = test value
    obj = {}           # the solver's own worst-case objective, for the monotonicity check
    for rat in (False, True):
        rk = "rat1" if rat else "rat0"
        meths = (OX + OW) if not rat else RAT_OK
        grid[rk] = {m: {} for m in meths}; obj[rk] = {m: {} for m in meths}
        for ce in CEPS:
            eps = tuple(common.tight_epsilon(Dm, T, w, K, is_distance=True, c_eps=ce))
            ck = "%g" % ce
            for m in meths:
                if m in OX and ce != CEPS[0]:
                    continue                       # O-X is c_eps-free: solve once
                grid[rk][m][ck] = {}; obj[rk][m][ck] = {}
                for L, lk in zip(LGRID, Lk):
                    try:
                        r = call(m, G, L, eps, rat)
                        grid[rk][m][ck][lk] = val(r.pi[1])
                        obj[rk][m][ck][lk] = float(r.objective_value)
                    except Exception as ex:
                        print("FAIL %s rat=%d ce=%s L=%s: %s" % (m, rat, ck, lk, str(ex)[:90]),
                              flush=True)

    # ---- Gamma-free arms ----
    xx = {}
    for m in XX:
        xx[m] = {}
        for L, lk in zip(LGRID, Lk):
            try:
                if m == "IPW-X-X": r = S[m](X, T, Y, w, n_arms=K, discretize=False, lipschitz=L)
                elif m == "DoublyRobust-X-X": r = S[m](X, T, Y, w, mu, n_arms=K, discretize=False, lipschitz=L)
                else: r = S[m](X, T, Y, n_arms=K, discretize=False, lipschitz=L)
                xx[m][lk] = {"value": val(r.pi[1]),
                             "curve": [round(float(v), 4) for v in dep(PGRID, r.pi[1])]}
            except Exception as ex:
                print("FAIL %s L=%s: %s" % (m, lk, str(ex)[:90]), flush=True)
    hess = {}
    try:
        sc = _W["hess"].fit_scores(X, T, Y, Gamma=G, k=15, n_folds=2, maximize=True, seed=a.seed)
        th = _W["hess"].learn_policy_parametric(X, sc, n_iter=300, lr=0.05, restarts=3,
                                                seed=a.seed, maximize=True)
        pe = _W["hess"].apply_policy(th, Xte.reshape(-1, 1))
        hess["%g" % G] = {"value": float(np.mean(pe * Y1t + (1 - pe) * Y0t))}
    except Exception as ex:
        print("FAIL Hess: %s" % str(ex)[:90], flush=True)
    kal = {}
    try:
        r = _W["kallus"](X, T, Y, wraw, n_arms=K, Gamma=G, maximize=True, seed=a.seed)
        pe = _W["kpred"](r.theta, Xte.reshape(-1, 1))[:, 1]
        kal["%g" % G] = {"value": float(np.mean(pe * Y1t + (1 - pe) * Y0t))}
    except Exception as ex:
        print("FAIL Kallus: %s" % str(ex)[:90], flush=True)

    orc = (d["cate_te"] > 0).astype(float)
    nv = (mu[:, 1] - mu[:, 0] > 0).astype(float)
    refs = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)),
            "never_treat": float(np.mean(Y0t)), "all_treat": float(np.mean(Y1t)),
            "naive_dr": val(nv), "P_T1": float(T.mean())}

    payload = {"dataset": a.data, "gamma": a.gamma, "Gamma": G, "seed": a.seed, "meta": meta,
               "ceps": CEPS, "lipschitz": Lk, "n_train": N_TRAIN, "n_test": N_TEST,
               "dgp_seed": a.dgp_seed, "grid": grid, "objective": obj, "xx": xx, "hess": hess, "kallus": kal,
               "refs": refs, "Y_standardisation": {"mean": _ym, "sd": _ys},
               "elapsed_s": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(a.out, "w"))
    print("saved %s  (%.0fs)" % (a.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
