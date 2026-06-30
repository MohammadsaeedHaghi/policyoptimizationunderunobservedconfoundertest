"""For ONE locked DGP, sweep (c_eps, Γ) over the full Γ grid, both regimes, to find the ε that gives the
O-W peak at small Γ (2-3) and the collapse onset at Γ≈4. Parallel over (c_eps, seed). Prints O-W value-vs-Γ
per c_eps + the best competitor, both regimes."""
import sys, json, time, importlib.util, argparse
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
LV = np.round(np.linspace(-1, 1, 7), 6); K = 2
COMP = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W"]
_W = {}
P = dict(alpha=10.0, cs=0.8, cx=2.0, d0=8.0, d1=9.5, b0=0.0, b1=2.0, noise=0.6)
if "--p" in sys.argv:
    _pv = [float(x) for x in sys.argv[sys.argv.index("--p") + 1].split(",")]
    P = dict(alpha=_pv[0], cs=_pv[1], cx=_pv[2], d0=_pv[3], d1=_pv[4], b0=_pv[5], b1=_pv[6], noise=_pv[7])

def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
def _init():
    import common; _W["common"] = common; S = {}
    for reg, suf, Cap in [("uncap", "uncapped", "Uncapped"), ("cap", "capped", "Capped")]:
        for nm, stem in [("IPW-X-X","ipw_x_x"),("DoublyRobust-X-X","doublyrobust_x_x"),("Direct-X-X","direct_x_x"),
                         ("IPW-O-X","ipw_o_x"),("DoublyRobust-O-X","doublyrobust_o_x"),("Hajek-O-X","hajek_o_x"),
                         ("IPW-O-W","ipw_o_w"),("DoublyRobust-O-W","doublyrobust_o_w")]:
            S[(reg, nm)] = _load("methods/%s/%s/%s_%s.py" % (nm, Cap, stem, suf), "solve_%s_%s" % (stem, suf))
    _W["S"] = S; _W["CAP"] = (1.0, 0.5)
def _tg(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])
def gen(n, sd):
    rng = np.random.default_rng(sd); X = rng.choice(LV, size=n)
    S = np.where(rng.uniform(size=n) < _sig(P["alpha"] * X), 1.0, -1.0)
    e = np.clip(_sig(P["cs"] * S - P["cx"] * X), 0.02, 0.98); T = (rng.uniform(size=n) < e).astype(int)
    Y = np.where(T == 1, P["d1"] * S + P["b1"] * X, P["d0"] * S + P["b0"] * X) + rng.normal(0, P["noise"], size=n)
    return {"X": X.reshape(-1, 1), "T": T, "Y": Y}
ES = 2 * _sig(P["alpha"] * LV) - 1.0; EY0 = P["d0"] * ES + P["b0"] * LV; EY1 = P["d1"] * ES + P["b1"] * LV
def val(pg): pg = np.asarray(pg, float); return float(np.mean(pg * EY1 + (1 - pg) * EY0))

def job(args):
    sd, N, GAM, CEPS = args
    common = _W["common"]; S = _W["S"]; CAP = _W["CAP"]
    o = gen(N, sd); w, _ = common.ipw_weights_from_data(o["X"], o["T"], K)
    wraw, _ = common.ipw_weights_from_data(o["X"], o["T"], K, normalize=False)
    mu = common.outcome_means(o["X"], o["T"], o["Y"], n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(o["X"])
    out = {"uncap": {}, "cap": {}}
    for reg in ("uncap", "cap"):
        kw0 = {} if reg == "uncap" else {"cap": CAP}
        out[reg]["IPW-X-X"] = val(_tg(S[(reg,"IPW-X-X")](o["X"], o["T"], o["Y"], w, n_arms=K, discretize=False, **kw0)))
        out[reg]["DoublyRobust-X-X"] = val(_tg(S[(reg,"DoublyRobust-X-X")](o["X"], o["T"], o["Y"], w, mu, n_arms=K, discretize=False, **kw0)))
        out[reg]["Direct-X-X"] = val(_tg(S[(reg,"Direct-X-X")](o["X"], o["T"], o["Y"], n_arms=K, discretize=False, **kw0)))
        for m in ["IPW-O-X","DoublyRobust-O-X","Hajek-O-X"]: out[reg][m] = []
        for ce in CEPS:
            for m in OW: out[reg]["%s@%g" % (m, ce)] = []
        for g in GAM:
            out[reg]["IPW-O-X"].append(val(_tg(S[(reg,"IPW-O-X")](o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False, **kw0))))
            out[reg]["DoublyRobust-O-X"].append(val(_tg(S[(reg,"DoublyRobust-O-X")](o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, **kw0))))
            out[reg]["Hajek-O-X"].append(val(_tg(S[(reg,"Hajek-O-X")](o["X"], o["T"], o["Y"], wraw, n_arms=K, Gamma=g, maximize=True, discretize=False, **kw0))))
            for ce in CEPS:
                eps = tuple(common.tight_epsilon(Dm, o["T"], w, K, is_distance=True, c_eps=ce))
                try:
                    vi = val(_tg(S[(reg,"IPW-O-W")](o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0)))
                except Exception:
                    vi = None
                try:
                    vd = val(_tg(S[(reg,"DoublyRobust-O-W")](o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0)))
                except Exception:
                    vd = None
                out[reg]["IPW-O-W@%g" % ce].append(vi)
                out[reg]["DoublyRobust-O-W@%g" % ce].append(vd)
    return sd, out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--gammas", default="1,1.5,2,2.5,3,4,5,6,8")
    ap.add_argument("--ceps", default="0.5,1.0,2.0"); ap.add_argument("--out", default=""); ap.add_argument("--p", default="")
    a = ap.parse_args()
    GAM = [float(x) for x in a.gammas.split(",")]; CEPS = [float(x) for x in a.ceps.split(",")]; SEEDS = list(range(a.seeds))
    Path("gurobi.env").write_text("Threads 1\n")
    orc = float(np.mean(np.where(EY1 > EY0, EY1, EY0)))
    jobs = [(sd, a.n, GAM, CEPS) for sd in SEEDS]
    print("eps-sweep: P=%s\n N=%d seeds=%d Γ=%s c_eps=%s oracle=%.3f" % (P, a.n, a.seeds, GAM, CEPS, orc), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        res = pool.map(job, jobs)
    NG = len(GAM)
    def _na(v):  # normalise every series to length NG (flat scalars broadcast), None -> nan
        arr = [np.nan if x is None else float(x) for x in np.atleast_1d(v)]
        if len(arr) == 1: arr = arr * NG
        return np.array(arr, float)
    agg = {"uncap": {}, "cap": {}}
    for sd, out in res:
        for reg in ("uncap", "cap"):
            for m, v in out[reg].items(): agg[reg].setdefault(m, []).append(_na(v))
    mean = {reg: {m: np.nanmean(np.array(agg[reg][m]), axis=0) for m in agg[reg]} for reg in ("uncap", "cap")}
    # SAVE FIRST (never lose compute), nan -> null
    if a.out:
        def js(x): return None if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), 4)
        Path(a.out).write_text(json.dumps({"P": P, "gammas": GAM, "ceps": CEPS, "n": a.n, "seeds": SEEDS, "oracle": orc,
            "mean": {reg: {m: [js(x) for x in mean[reg][m]] for m in mean[reg]} for reg in ("uncap", "cap")}}, indent=2))
        print("saved %s" % a.out, flush=True)
    for reg in ("uncap", "cap"):
        M = mean[reg]
        compflat = [max(M[c][i] for c in COMP) for i in range(NG)]
        print("\n[%s] Γ:              %s   oracle=%.3f" % (reg, "  ".join("%5.1f" % g for g in GAM), orc))
        print("  best-competitor   " + "  ".join("%5.2f" % x for x in compflat))
        for ce in CEPS:
            for m in OW:
                print("  %-13s@%-4g" % (m.replace("DoublyRobust", "DR"), ce) + "  ".join("%5.2f" % x for x in M["%s@%g" % (m, ce)]))
    print("\n(%.1f min)" % ((time.time() - t0) / 60), flush=True)

if __name__ == "__main__":
    main()
