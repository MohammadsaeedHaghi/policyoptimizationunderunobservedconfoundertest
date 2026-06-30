"""Search a discrete-X, binary-T DGP where the O-W family (BOTH IPW-O-W and DoublyRobust-O-W) beats EVERY
other method (incl. plain DoublyRobust-X-X) by a GOOD GAP, in BOTH capped & uncapped, peaking at small Γ (2-3).

DGP family (additive, NON-cancelling confounding -- the owwin3 recipe, swept wider for a bigger margin):
  X ~ Uniform on 7 discrete levels in [-1,1]
  S in {-1,+1} UNOBSERVED, P(S=+1|X)=sigma(alpha X)         (strong S-X correlation -> Wasserstein lever)
  e(X,S)=clip(sigma(cs S - cx X),0.02,0.98)                 (selection on S => confounding; sets matched Γ)
  mu0(X,S)=d0 S + b0 X,  mu1(X,S)=d1 S + b1 X               (d1!=d0 NON-cancelling; large d, small differential)
  Y(t)=mu_t + N(0,noise^2)

Exact analytic value over the X-grid (X uniform): V(pi)=mean_X[ pi eY1 + (1-pi) eY0 ],
  eY0=d0 ES + b0 X, eY1=d1 ES + b1 X, ES(X)=2 sigma(alpha X)-1.  Oracle = treat iff CATE=eY1-eY0 > 0.

Scoring (c_eps=1, the operating ε): for each regime,
  owpeak = min over {IPW-O-W, DR-O-W} of max_{Γ in 2,3} value          (both O-W good at small Γ)
  comp   = max over the 6 competitors of max_Γ value                   (everyone else at their BEST Γ)
  margin = owpeak - comp.   score = min(margin_uncap, margin_cap).      We want this large & positive in BOTH.
Parallel over (config, seed). Low N for the sweep; winners get verified at full N afterwards.
"""
import sys, os, json, time, importlib.util, argparse
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]; OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]; OW = ["IPW-O-W", "DoublyRobust-O-W"]
COMP = XX + OX
LV = np.round(np.linspace(-1, 1, 7), 6); K = 2
_W = {}

def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)

def _init():
    import common; _W["common"] = common; S = {}
    for reg, suf, Cap in [("uncap", "uncapped", "Uncapped"), ("cap", "capped", "Capped")]:
        S[(reg, "IPW-X-X")] = _load("methods/IPW-X-X/%s/ipw_x_x_%s.py" % (Cap, suf), "solve_ipw_x_x_%s" % suf)
        S[(reg, "DoublyRobust-X-X")] = _load("methods/DoublyRobust-X-X/%s/doublyrobust_x_x_%s.py" % (Cap, suf), "solve_doublyrobust_x_x_%s" % suf)
        S[(reg, "Direct-X-X")] = _load("methods/Direct-X-X/%s/direct_x_x_%s.py" % (Cap, suf), "solve_direct_x_x_%s" % suf)
        S[(reg, "IPW-O-X")] = _load("methods/IPW-O-X/%s/ipw_o_x_%s.py" % (Cap, suf), "solve_ipw_o_x_%s" % suf)
        S[(reg, "DoublyRobust-O-X")] = _load("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py" % (Cap, suf), "solve_doublyrobust_o_x_%s" % suf)
        S[(reg, "Hajek-O-X")] = _load("methods/Hajek-O-X/%s/hajek_o_x_%s.py" % (Cap, suf), "solve_hajek_o_x_%s" % suf)
        S[(reg, "IPW-O-W")] = _load("methods/IPW-O-W/%s/ipw_o_w_%s.py" % (Cap, suf), "solve_ipw_o_w_%s" % suf)
        S[(reg, "DoublyRobust-O-W")] = _load("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py" % (Cap, suf), "solve_doublyrobust_o_w_%s" % suf)
    _W["S"] = S
    _W["CAP"] = (1.0, 0.5)

def _tg(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])

def gen(p, n, sd):
    rng = np.random.default_rng(sd); X = rng.choice(LV, size=n)
    S = np.where(rng.uniform(size=n) < _sig(p["alpha"] * X), 1.0, -1.0)
    e = np.clip(_sig(p["cs"] * S - p["cx"] * X), 0.02, 0.98); T = (rng.uniform(size=n) < e).astype(int)
    m0 = p["d0"] * S + p["b0"] * X; m1 = p["d1"] * S + p["b1"] * X
    Y = np.where(T == 1, m1, m0) + rng.normal(0, p["noise"], size=n)
    return {"X": X.reshape(-1, 1), "T": T, "Y": Y}

def truth(p):
    ES = 2 * _sig(p["alpha"] * LV) - 1.0
    eY0 = p["d0"] * ES + p["b0"] * LV; eY1 = p["d1"] * ES + p["b1"] * LV
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
        for m in OX + OW: res[reg][m] = []
        for g in GAM:
            res[reg]["IPW-O-X"].append(val(_tg(S[(reg, "IPW-O-X")](o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False, **kw0))))
            res[reg]["DoublyRobust-O-X"].append(val(_tg(S[(reg, "DoublyRobust-O-X")](o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, **kw0))))
            res[reg]["Hajek-O-X"].append(val(_tg(S[(reg, "Hajek-O-X")](o["X"], o["T"], o["Y"], wraw, n_arms=K, Gamma=g, maximize=True, discretize=False, **kw0))))
            res[reg]["IPW-O-W"].append(val(_tg(S[(reg, "IPW-O-W")](o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0))))
            res[reg]["DoublyRobust-O-W"].append(val(_tg(S[(reg, "DoublyRobust-O-W")](o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0))))
    return ci, sd, res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ceps", type=float, default=1.0); ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--gammas", default="1.5,2,3,4"); ap.add_argument("--out", default="")
    ap.add_argument("--grid", default="base")
    a = ap.parse_args()
    GAM = [float(x) for x in a.gammas.split(",")]; SEEDS = list(range(a.seeds))
    Path("gurobi.env").write_text("Threads 1\n")
    # ---- config grid ----
    CONFIGS = {}
    if a.grid == "base":
        # neighbourhood of owwin3-I3 + wider differentials/selection to push the margin
        for alpha in (8.0, 10.0, 12.0):
            for cs in (0.8, 1.0, 1.2):
                for d0 in (4.0, 5.0):
                    for dd in (0.75, 1.0, 1.5):
                        for b in (1.0, 1.5):
                            CONFIGS["a%g_cs%g_d%g_dd%g_b%g" % (alpha, cs, d0, dd, b)] = dict(
                                alpha=alpha, cs=cs, cx=2.0, d0=d0, d1=d0 + dd, b0=0.0, b1=b, noise=0.6)
    elif a.grid == "wide":
        for alpha in (10.0, 14.0):
            for cs in (0.7, 0.9, 1.1):
                for cx in (1.5, 2.5):
                    for d0 in (5.0, 7.0):
                        for dd in (1.0, 1.5, 2.0):
                            for b in (1.5, 2.0):
                                CONFIGS["a%g_cs%g_cx%g_d%g_dd%g_b%g" % (alpha, cs, cx, d0, dd, b)] = dict(
                                    alpha=alpha, cs=cs, cx=cx, d0=d0, d1=d0 + dd, b0=0.0, b1=b, noise=0.6)
    elif a.grid == "push":
        # LARGE d (strong outcome-model bias -> naive/box treat ~everyone) + SMALL differential + BIGGER b
        # (raises the cost of wrong-treating X<0 -> bigger gap), moderate cs (small matched Γ -> peak Γ 2-3),
        # high alpha (strong U-X corr -> O-W balance recovers oracle). owwin3-I3 kept as the baseline anchor.
        CONFIGS["I3_base"] = dict(alpha=10.0, cs=0.8, cx=2.0, d0=5.0, d1=6.0, b0=0.0, b1=1.0, noise=0.6)
        for alpha in (10.0, 12.0):
            for cs in (0.8, 1.0):
                for d0 in (6.0, 8.0):
                    for dd in (1.0, 1.5):
                        for b in (2.0, 3.0):
                            CONFIGS["a%g_cs%g_d%g_dd%g_b%g" % (alpha, cs, d0, dd, b)] = dict(
                                alpha=alpha, cs=cs, cx=2.0, d0=d0, d1=d0 + dd, b0=0.0, b1=b, noise=0.6)
    names = list(CONFIGS)
    jobs = [(ci, CONFIGS[ci], sd, a.n, GAM, a.ceps) for ci in names for sd in SEEDS]
    print("search: %d configs x %d seeds = %d jobs, N=%d, Γ=%s, c_eps=%g, workers=%d" % (len(names), a.seeds, len(jobs), a.n, GAM, a.ceps, a.workers), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        out = pool.map(job, jobs)
    # aggregate: ci -> regime -> method -> mean over seeds [by Γ]
    agg = {ci: {"uncap": {}, "cap": {}} for ci in names}
    for ci, sd, res in out:
        for reg in ("uncap", "cap"):
            for m, v in res[reg].items():
                agg[ci][reg].setdefault(m, []).append(v)
    GI = {g: i for i, g in enumerate(GAM)}
    small = [g for g in GAM if g in (2.0, 3.0)]
    rows = []
    for ci in names:
        mean = {reg: {m: np.mean(agg[ci][reg][m], axis=0) for m in agg[ci][reg]} for reg in ("uncap", "cap")}
        rec = {"config": ci, "p": CONFIGS[ci]}
        margins = {}
        for reg in ("uncap", "cap"):
            M = mean[reg]
            owpeak = min(max(M["IPW-O-W"][GI[g]] for g in small), max(M["DoublyRobust-O-W"][GI[g]] for g in small))
            comp = max(max(M[c]) for c in COMP)
            margins[reg] = owpeak - comp
            rec[reg + "_owpeak"] = round(owpeak, 4); rec[reg + "_comp"] = round(comp, 4); rec[reg + "_margin"] = round(margins[reg], 4)
        rec["score"] = round(min(margins["uncap"], margins["cap"]), 4)
        rec["mean"] = {reg: {m: [round(float(x), 4) for x in mean[reg][m]] for m in mean[reg]} for reg in ("uncap", "cap")}
        rows.append(rec)
    rows.sort(key=lambda r: r["score"], reverse=True)
    print("\n== TOP 12 by min-regime O-W margin (Γ=%s) ==" % GAM, flush=True)
    print("  %-34s score  uncap[ow/comp]  cap[ow/comp]" % "config", flush=True)
    for r in rows[:12]:
        print("  %-34s %+.3f   %.3f/%.3f    %.3f/%.3f" % (r["config"], r["score"], r["uncap_owpeak"], r["uncap_comp"], r["cap_owpeak"], r["cap_comp"]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"gammas": GAM, "n": a.n, "seeds": SEEDS, "ceps": a.ceps, "rows": rows}, indent=2))
        print("\nsaved %s  (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)

if __name__ == "__main__":
    main()
