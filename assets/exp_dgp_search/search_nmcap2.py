"""Search a NON-MONOTONE discrete-X DGP where BOTH O-W methods beat naive DoublyRobust-X-X UNDER THE CAP
(defeating the cap-rescue that protects naive in monotone DGPs).

Mechanism (cf assets/exp_nonmono): the treatment helps a MIDDLE band of X and HARMS the edges, with
EDGE-concentrated confounding -> under cap<=50% the naive method ranks the (confounded, good-looking) edges
first and treats the WRONG band, while the O / O-W robust methods downweight the edge-imbalanced treated mass
and treat the right middle band.

DGP family:
  X ~ Uniform 7 levels in [-1,1];  S in {-1,+1} UNOBSERVED, P(S=+1)=1/2 (independent of X)
  e(X,S)=clip(sigma(cs*S + ce*X^2), .02,.98)   (cs: S-confounding; ce: EDGE-heavy selection -> treated edge-heavy
                                                -> X-imbalance the Wasserstein term can flag)
  mu0=d0*S,  mu1=d1*S + b*(c - X^2)            (S shifts both arms = bias; treatment helps |X|<sqrt(c), harms edges)
  Y(t)=mu_t + N(0,noise^2)
  CATE(X)=b*(c - X^2)  (since E[S|X]=0) -> oracle treats the MIDDLE band X^2<c.  mean_X[eY0]=0 so V=mean[pi*CATE].

Score: capped  min(IPW-O-W, DR-O-W) - DoublyRobust-X-X  at Gamma in {2,3}  (want > 0 by a good margin).
"""
import sys, json, time, importlib.util, argparse
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
LV = np.round(np.linspace(-1, 1, 7), 6); K = 2
ALLM = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
_W = {}

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
def gen(p, n, sd):
    rng = np.random.default_rng(sd); X = rng.choice(LV, size=n)
    S = np.where(rng.uniform(size=n) < _sig(p["alpha"] * X), 1.0, -1.0)
    e = np.clip(_sig(p["cs"] * S + p["ce"] * X**2), 0.02, 0.98); T = (rng.uniform(size=n) < e).astype(int)
    m0 = p["d0"] * S; m1 = p["d1"] * S + p["b"] * (p["c"] - X**2)
    Y = np.where(T == 1, m1, m0) + rng.normal(0, p["noise"], size=n)
    return {"X": X.reshape(-1, 1), "T": T, "Y": Y}
def truth(p):
    ES = 2*_sig(p["alpha"]*LV)-1.0; eY0 = p["d0"]*ES; eY1 = p["d1"]*ES + p["b"]*(p["c"] - LV**2)
    return eY0, eY1
def job(args):
    ci, p, sd, N, GAM, ceps = args
    common = _W["common"]; S = _W["S"]; CAP = _W["CAP"]
    eY0, eY1 = truth(p)
    def val(pg): pg = np.asarray(pg, float); return float(np.mean(pg * eY1 + (1 - pg) * eY0))
    o = gen(p, N, sd); w, _ = common.ipw_weights_from_data(o["X"], o["T"], K)
    wraw, _ = common.ipw_weights_from_data(o["X"], o["T"], K, normalize=False)
    mu = common.outcome_means(o["X"], o["T"], o["Y"], n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(o["X"]); eps = tuple(common.tight_epsilon(Dm, o["T"], w, K, is_distance=True, c_eps=ceps))
    res = {"uncap": {}, "cap": {}}
    for reg in ("uncap", "cap"):
        kw0 = {} if reg == "uncap" else {"cap": CAP}
        res[reg]["IPW-X-X"] = [val(_tg(S[(reg, "IPW-X-X")](o["X"], o["T"], o["Y"], w, n_arms=K, discretize=False, **kw0)))] * len(GAM)
        res[reg]["DoublyRobust-X-X"] = [val(_tg(S[(reg, "DoublyRobust-X-X")](o["X"], o["T"], o["Y"], w, mu, n_arms=K, discretize=False, **kw0)))] * len(GAM)
        res[reg]["Direct-X-X"] = [val(_tg(S[(reg, "Direct-X-X")](o["X"], o["T"], o["Y"], n_arms=K, discretize=False, **kw0)))] * len(GAM)
        for m in ["IPW-O-X","DoublyRobust-O-X","Hajek-O-X","IPW-O-W","DoublyRobust-O-W"]: res[reg][m] = []
        for g in GAM:
            res[reg]["IPW-O-X"].append(val(_tg(S[(reg, "IPW-O-X")](o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False, **kw0))))
            res[reg]["DoublyRobust-O-X"].append(val(_tg(S[(reg, "DoublyRobust-O-X")](o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, **kw0))))
            res[reg]["Hajek-O-X"].append(val(_tg(S[(reg, "Hajek-O-X")](o["X"], o["T"], o["Y"], wraw, n_arms=K, Gamma=g, maximize=True, discretize=False, **kw0))))
            try: vi = val(_tg(S[(reg, "IPW-O-W")](o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0)))
            except Exception: vi = None
            try: vd = val(_tg(S[(reg, "DoublyRobust-O-W")](o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0)))
            except Exception: vd = None
            res[reg]["IPW-O-W"].append(vi); res[reg]["DoublyRobust-O-W"].append(vd)
    return ci, sd, res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ceps", type=float, default=1.0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gammas", default="1.5,2,3,4"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    GAM = [float(x) for x in a.gammas.split(",")]; SEEDS = list(range(a.seeds))
    Path("gurobi.env").write_text("Threads 1\n")
    CONFIGS = {}
    for alpha in (4.0, 8.0):
        for cs in (1.0, 1.5):
            for ce in (2.0, 3.0):
                for b in (2.5, 3.5):
                    CONFIGS["a%g_cs%g_ce%g_b%g" % (alpha, cs, ce, b)] = dict(
                        alpha=alpha, cs=cs, ce=ce, d0=3.0, d1=4.0, b=b, c=0.3, noise=0.6)
    names = list(CONFIGS)
    jobs = [(ci, CONFIGS[ci], sd, a.n, GAM, a.ceps) for ci in names for sd in SEEDS]
    print("nmcap search: %d configs x %d seeds = %d jobs, N=%d Γ=%s c_eps=%g" % (len(names), a.seeds, len(jobs), a.n, GAM, a.ceps), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        out = pool.map(job, jobs)
    agg = {ci: {"uncap": {}, "cap": {}} for ci in names}
    for ci, sd, res in out:
        for reg in ("uncap", "cap"):
            for m, v in res[reg].items(): agg[ci][reg].setdefault(m, []).append([np.nan if x is None else x for x in v])
    GI = {g: i for i, g in enumerate(GAM)}; small = [g for g in GAM if g in (2.0, 3.0)]
    rows = []
    for ci in names:
        mean = {reg: {m: np.nanmean(np.array(agg[ci][reg][m]), axis=0) for m in agg[ci][reg]} for reg in ("uncap", "cap")}
        rec = {"config": ci, "p": CONFIGS[ci]}
        for reg in ("uncap", "cap"):
            M = mean[reg]
            ow = min(max(M["IPW-O-W"][GI[g]] for g in small), max(M["DoublyRobust-O-W"][GI[g]] for g in small))
            naive = M["DoublyRobust-X-X"][0]
            allcomp = max(max(M[c]) for c in ["IPW-X-X","DoublyRobust-X-X","Direct-X-X","IPW-O-X","DoublyRobust-O-X","Hajek-O-X"])
            rec[reg + "_ow"] = round(ow, 4); rec[reg + "_naive"] = round(naive, 4); rec[reg + "_allcomp"] = round(allcomp, 4)
            rec[reg + "_gap_vs_naive"] = round(ow - naive, 4); rec[reg + "_gap_vs_all"] = round(ow - allcomp, 4)
        rec["mean"] = {reg: {m: [round(float(x), 4) for x in mean[reg][m]] for m in mean[reg]} for reg in ("uncap", "cap")}
        rows.append(rec)
    rows.sort(key=lambda r: r["cap_gap_vs_all"], reverse=True)   # want capped O-W to beat ALL competitors
    print("\n== TOP by CAPPED O-W gap over ALL competitors (Γ=%s) ==" % GAM, flush=True)
    print("  %-30s cap[ow/all/naive]  uncap[ow/all]" % "config", flush=True)
    for r in rows[:12]:
        print("  %-30s %.2f/%.2f/%.2f  %.2f/%.2f" % (r["config"], r["cap_ow"], r["cap_allcomp"], r["cap_naive"], r["uncap_ow"], r["uncap_allcomp"]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"gammas": GAM, "n": a.n, "seeds": SEEDS, "ceps": a.ceps, "rows": rows}, indent=2))
        print("\nsaved %s (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)

if __name__ == "__main__":
    main()
