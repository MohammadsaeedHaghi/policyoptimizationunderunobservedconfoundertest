"""Search for a discrete DGP where the OW family (IPW-O-W AND DoublyRobust-O-W) beats EVERY other method
(incl. DoublyRobust-O-X, Direct-X-X, Hajek-O-X) at a SMALL Γ, uncapped. Levers: large ε (c_eps), strong S-X
correlation, outcome noise (to blunt DR's outcome-model edge). N=300, 3 seeds, exact value, small Γ grid."""
import sys, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(ROOT)); import common
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
U = "Uncapped"; sfx = "uncapped"
ipwxx = L("methods/IPW-X-X/%s/ipw_x_x_%s.py" % (U, sfx), "solve_ipw_x_x_%s" % sfx)
drxx  = L("methods/DoublyRobust-X-X/%s/doublyrobust_x_x_%s.py" % (U, sfx), "solve_doublyrobust_x_x_%s" % sfx)
dirxx = L("methods/Direct-X-X/%s/direct_x_x_%s.py" % (U, sfx), "solve_direct_x_x_%s" % sfx)
ipwox = L("methods/IPW-O-X/%s/ipw_o_x_%s.py" % (U, sfx), "solve_ipw_o_x_%s" % sfx)
drox  = L("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py" % (U, sfx), "solve_doublyrobust_o_x_%s" % sfx)
hjox  = L("methods/Hajek-O-X/%s/hajek_o_x_%s.py" % (U, sfx), "solve_hajek_o_x_%s" % sfx)
ipwow = L("methods/IPW-O-W/%s/ipw_o_w_%s.py" % (U, sfx), "solve_ipw_o_w_%s" % sfx)
drow  = L("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py" % (U, sfx), "solve_doublyrobust_o_w_%s" % sfx)
def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
LV = np.round(np.linspace(-1, 1, 7), 6); K = 2; N = 300; SEEDS = [0, 1, 2]; GAM = [1.5, 2.0, 3.0, 4.0]
def make(p):
    d0, d1 = p["d0"], p["d1"]                       # differential S effect: NON-cancelling confounding
    def gen(n, sd):
        rng = np.random.default_rng(sd); X = rng.choice(LV, size=n)
        S = np.where(rng.uniform(size=n) < sig(p["alpha"] * X), 1.0, -1.0)
        e = np.clip(sig(p["cs"] * S - p["cx"] * X), 0.02, 0.98); T = (rng.uniform(size=n) < e).astype(int)
        Y = np.where(T == 1, d1 * S + p["b"] * X, d0 * S) + rng.normal(0, p["noise"], size=n)
        return dict(X=X.reshape(-1, 1), T=T, Y=Y)
    ES = 2 * sig(p["alpha"] * LV) - 1.0
    return gen, d0 * ES, d1 * ES + p["b"] * LV       # eY0, eY1
def tg(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])
CONFIGS = {  # large d0,d1 (strong model bias) + SMALL differential (subtle CATE)
 "I1": dict(alpha=8.0, cs=1.0, cx=2.0, d0=4.0, d1=5.0, b=1.0, noise=0.6),
 "I2": dict(alpha=8.0, cs=1.0, cx=2.0, d0=5.0, d1=6.5, b=1.5, noise=0.6),
 "I3": dict(alpha=10.0, cs=0.8, cx=2.0, d0=5.0, d1=6.0, b=1.0, noise=0.6),
 "I4": dict(alpha=8.0, cs=1.2, cx=2.0, d0=4.0, d1=6.0, b=1.5, noise=0.8),
}
CEPS = [1.0, 2.0, 3.0, 4.0]
t0 = time.time()
for name, p in CONFIGS.items():
    gen, eY0, eY1 = make(p)
    def val(pg): pg = np.asarray(pg, float); return float(np.mean(pg * eY1 + (1 - pg) * eY0))
    orc = val((LV > 0).astype(float))
    data = []
    for sd in SEEDS:
        o = gen(N, sd); w, _ = common.ipw_weights_from_data(o["X"], o["T"], K)
        wraw, _ = common.ipw_weights_from_data(o["X"], o["T"], K, normalize=False)
        mu = common.outcome_means(o["X"], o["T"], o["Y"], n_arms=K, cross_fit=True)
        Dm = common.pairwise_distance_matrix(o["X"]); data.append((o, w, wraw, mu, Dm))
    # competitors (no ε): XX (Γ-free) + OX/Hajek over Γ
    comp = {}
    for nm in ["IPW-X-X", "DR-X-X", "Direct-X-X"]:
        vs = []
        for o, w, wraw, mu, Dm in data:
            if nm == "IPW-X-X": r = ipwxx(o["X"], o["T"], o["Y"], w, n_arms=K, discretize=False)
            elif nm == "DR-X-X": r = drxx(o["X"], o["T"], o["Y"], w, mu, n_arms=K, discretize=False)
            else: r = dirxx(o["X"], o["T"], o["Y"], n_arms=K, discretize=False)
            vs.append(val(tg(r)))
        comp[nm] = np.mean(vs)
    for nm, fn, dr in [("IPW-O-X", ipwox, False), ("DR-O-X", drox, True), ("Hajek-O-X", hjox, "h")]:
        best = -9
        for g in GAM:
            vs = []
            for o, w, wraw, mu, Dm in data:
                if dr == "h": r = fn(o["X"], o["T"], o["Y"], wraw, n_arms=K, Gamma=g, maximize=True, discretize=False)
                elif dr: r = fn(o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False)
                else: r = fn(o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False)
                vs.append(val(tg(r)))
            best = max(best, np.mean(vs))
        comp[nm] = best
    compmax = max(comp.values())
    # OW family per c_eps over small Γ
    print("[%s] orc=%.2f | competitors: %s (max=%.3f)" % (name, orc, {k: round(v, 3) for k, v in comp.items()}, compmax), flush=True)
    for ce in CEPS:
        owbyg = {}
        for g in GAM:
            vo, vd = [], []
            for o, w, wraw, mu, Dm in data:
                eps = tuple(common.tight_epsilon(Dm, o["T"], w, K, is_distance=True, c_eps=ce))
                vo.append(val(tg(ipwow(o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps))))
                vd.append(val(tg(drow(o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps))))
            owbyg[g] = (np.mean(vo), np.mean(vd))
        # find smallest Γ where BOTH OW > compmax
        winG = next((g for g in GAM if min(owbyg[g]) > compmax + 0.003), None)
        tag = "  <<< OW BEATS ALL @Γ=%s" % winG if winG else ""
        print("   c_eps=%.0f  IPW-OW=%s  DR-OW=%s%s" % (ce, [round(owbyg[g][0], 3) for g in GAM], [round(owbyg[g][1], 3) for g in GAM], tag), flush=True)
print("done %.1f min" % ((time.time() - t0) / 60), flush=True)
