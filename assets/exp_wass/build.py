"""Build the continuous-X 'Wasserstein wins' experiment (Fourth experiment): ground-truth + observed-data
explanation charts, and the FAST/box methods at N_train=2000 evaluated on a large test set. Builds the
interactive vars (WASS_*), splices into index.html, saves PNGs. The slow Wasserstein methods (IPW-O-W,
DoublyRobust-O-W) are added later by run_wass.py. Usage: python3 build.py"""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_wass"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("wdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["wdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwxx = L("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped")
directxx = L("methods/Direct-X-X/Uncapped/direct_x_x_uncapped.py", "solve_direct_x_x_uncapped")
ipwox = L("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped")
hajekox = L("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py", "solve_hajek_o_x_uncapped")
drxx = L("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py", "solve_doublyrobust_x_x_uncapped")
drox = L("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped")

K, N, MESH = 2, 1000, 11
G = d.GRID
GAMMAS = [1, 2, 4, 8]
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)

# ---- train + test ----
obs, full = d.generate(N, 0)
w, P = common.ipw_weights_from_data(obs["X"], obs["T"], K)          # LOGISTIC propensity (continuous X)
wraw = 1.0 / common.factual_propensity(P, obs["T"])
muhat = common.outcome_means(obs["X"], obs["T"], obs["Y"], K)
_, tf = d.generate(40000, 99); Xt = tf["X"].ravel(); St = tf["S"]

def mesh_policy(res):
    sup = res.support_X.ravel(); lv = np.array(sorted(set(np.round(sup, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(sup, 6) == round(float(c), 6))[0][0]]) for c in lv])
    return lv, pol
def to_grid(res, Xq):
    lv, pol = mesh_policy(res); nn = lv[np.argmin(np.abs(np.asarray(Xq)[:, None] - lv[None, :]), axis=1)]
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in nn])
def realised(res): pit = to_grid(res, Xt); return float(np.mean(pit * d.mu1(Xt, St) + (1 - pit) * d.mu0(Xt, St)))
def polgrid(res): return [round(float(v), 4) for v in to_grid(res, G)]

# ---- ground-truth curves ----
t = d.grid_truth()
def ser(idl, label, color, y, **kw):
    dd = {"id": idl, "label": label, "color": color, "y": [round(float(v), 4) for v in y]}; dd.update(kw); return dd
def chart(xlab, ylab, ymin, ymax, series, note, hlines=None, x=None):
    c = {"x": [float(v) for v in (x if x is not None else G)], "xmin": -1.0, "xmax": 1.0, "xlabel": xlab, "ylabel": ylab,
         "ymin": ymin, "ymax": ymax, "series": series, "note": note}
    if hlines: c["hlines"] = hlines
    return c
BLUE, RED, BLUE_DK, RED_DK = "#1f77b4", "#dc2626", "#16537e", "#9a1b1b"
W = {}
W["outcome"] = chart("X  (continuous covariate)", "E[Y(t) | X, S]", -2.2, 2.2, [
    ser("m1p", "E[Y(1)|X,S=+1]  treated", BLUE, t["m1_plus"], marker="square"),
    ser("m1m", "E[Y(1)|X,S=−1]  treated", BLUE, t["m1_minus"], marker="circle"),
    ser("m1a", "E[Y(1)|X]  treated · avg", BLUE_DK, t["eY1"], width=4.6),
    ser("m0p", "E[Y(0)|X,S=+1]  control", RED, t["m0_plus"], marker="square"),
    ser("m0m", "E[Y(0)|X,S=−1]  control", RED, t["m0_minus"], marker="circle"),
    ser("m0a", "E[Y(0)|X]  control · avg", RED_DK, t["eY0"], width=4.6),
], "Blue=treated, red=control. Square=S=+1, circle=S=−1, bold=average over S. S inflates the baseline; treatment adds X.")
W["cate"] = chart("X", "CATE(X) = E[Y(1)−Y(0) | X]", -1.1, 1.1, [
    ser("cate", "CATE(X) = X", "#4f46e5", t["cate"], width=3.4),
], "CATE(X)=X (same for both S), so the optimal policy is train iff X>0.",
   hlines=[{"y": 0.0, "color": "#94a3b8", "dash": "4,4", "label": "no effect"}])
W["sx"] = chart("X", "P(S=+1 | X)", 0.0, 1.0, [
    ser("ps1", "P(S=+1 | X) = σ(4X)", "#7c3aed", t["p_s1"], width=3.4),
], "The KEY: the unobserved S is strongly correlated with X, so balancing X also balances S.",
   hlines=[{"y": 0.5, "color": "#94a3b8", "dash": "4,4", "label": "50/50"}])
W["prop"] = chart("X", "P(enroll | X, S)", 0.0, 1.0, [
    ser("ep", "e(X,S=+1)", BLUE, t["e_plus"], marker="square"),
    ser("em", "e(X,S=−1)", RED, t["e_minus"], marker="circle"),
    ser("emarg", "marginal ẽ(X)", "#334155", t["e_marg"], width=3.6),
], "Mis-targeted (treats low X) + S-confounded. The marginal ẽ(X) VARIES with X (unlike the HR DGP) ⇒ real covariate imbalance.")

# ---- observed-data diagnostic: treated vs control covariate density ----
edges = np.linspace(-1, 1, 22); cx = 0.5 * (edges[:-1] + edges[1:])
xr = obs["X"].ravel()
htr, _ = np.histogram(xr[obs["T"] == 1], bins=edges, density=True)
hco, _ = np.histogram(xr[obs["T"] == 0], bins=edges, density=True)
W["obs"] = chart("X", "covariate density  (N=2000)", 0.0, float(max(htr.max(), hco.max()) * 1.1), [
    ser("tr", "treated (T=1)", "#0e7490", htr), ser("co", "control (T=0)", "#94a3b8", hco),
], "Observed covariate imbalance: treated skew to LOW X, control to HIGH X. This is the imbalance Wasserstein corrects.", x=cx)

# ---- methods (fast/box) ----
res_xx = ipwxx(obs["X"], obs["T"], obs["Y"], w, n_arms=K, discretize=True, mesh=MESH)
res_dir = directxx(obs["X"], obs["T"], obs["Y"], n_arms=K, discretize=True, mesh=MESH)
res_drx = drxx(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, discretize=True, mesh=MESH)
v_xx, v_dir, v_drx = realised(res_xx), realised(res_dir), realised(res_drx)
ox, hox, dro = {}, {}, {}
for g in GAMMAS:
    ox[g] = ipwox(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(g), discretize=True, mesh=MESH)
    hox[g] = hajekox(obs["X"], obs["T"], obs["Y"], wraw, n_arms=K, Gamma=float(g), maximize=True, discretize=True, mesh=MESH)
    dro[g] = drox(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, Gamma=float(g), discretize=True, mesh=MESH)
oracle = d.oracle_policy(G)
v_oracle = d.realized_value_means(d.oracle_policy(Xt), Xt, St)
v_alltreat = d.realized_value_means(np.ones_like(Xt), Xt, St)
print("Oracle=%.3f all-treat=%.3f | IPW-X-X=%.3f Direct-X-X=%.3f DR-X-X=%.3f" % (v_oracle, v_alltreat, v_xx, v_dir, v_drx))
print("Γ    IPW-O-X  Hajek-O-X  DR-O-X")
for g in GAMMAS:
    print(" %d   %.3f   %.3f    %.3f" % (g, realised(ox[g]), realised(hox[g]), realised(dro[g])))

# ---- value-vs-Γ chart (fast methods now; Wasserstein patched later) ----
COL = {"IPW-X-X": "#1f77b4", "IPW-O-X": "#1f77b4", "Hajek-O-X": "#9467bd", "DoublyRobust-X-X": "#2ca02c",
       "DoublyRobust-O-X": "#2ca02c", "Direct-X-X": "#ff7f0e", "IPW-O-W": "#1f77b4", "DoublyRobust-O-W": "#2ca02c", "Oracle": "#444444"}
flat = lambda v: [round(v, 4)] * len(GAMMAS)
def vs(resd): return [round(realised(resd[g]), 4) for g in GAMMAS]
W["value"] = {"x": [float(g) for g in GAMMAS], "xmin": 1.0, "xmax": 8.0, "xlabel": "Γ  (sensitivity-model strength)",
              "ylabel": "realised E[Y] on test", "ymin": 0.0, "ymax": 0.30,
              "series": [ser("IPW-X-X", "IPW-X-X", COL["IPW-X-X"], flat(v_xx)),
                         ser("Direct-X-X", "Direct-X-X", COL["Direct-X-X"], flat(v_dir)),
                         ser("DoublyRobust-X-X", "DoublyRobust-X-X", COL["DoublyRobust-X-X"], flat(v_drx)),
                         ser("IPW-O-X", "IPW-O-X", COL["IPW-O-X"], vs(ox)),
                         ser("Hajek-O-X", "Hajek-O-X", COL["Hajek-O-X"], vs(hox)),
                         ser("DoublyRobust-O-X", "DoublyRobust-O-X", COL["DoublyRobust-O-X"], vs(dro)),
                         ser("Oracle", "Oracle", COL["Oracle"], flat(v_oracle))],
              "hlines": [{"y": round(v_oracle, 4), "color": "#94a3b8", "dash": "4,4", "label": "oracle"}],
              "note": "Realised test E[Y] vs Γ, N_train=2000. IPW-O-W / DoublyRobust-O-W (Wasserstein) added when their solves finish."}
# ---- policy chart at default Γ=4 (Γ dropdown) ----
def psa(g):
    return [ser("IPW-X-X", "IPW-X-X", COL["IPW-X-X"], polgrid(res_xx)),
            ser("Direct-X-X", "Direct-X-X", COL["Direct-X-X"], polgrid(res_dir)),
            ser("DoublyRobust-X-X", "DoublyRobust-X-X", COL["DoublyRobust-X-X"], polgrid(res_drx)),
            ser("IPW-O-X", "IPW-O-X", COL["IPW-O-X"], polgrid(ox[g])),
            ser("Hajek-O-X", "Hajek-O-X", COL["Hajek-O-X"], polgrid(hox[g])),
            ser("DoublyRobust-O-X", "DoublyRobust-O-X", COL["DoublyRobust-O-X"], polgrid(dro[g])),
            ser("Oracle", "Oracle (train iff X>0)", COL["Oracle"], [round(float(v), 4) for v in oracle])]
W["policy"] = {"x": [float(v) for v in G], "xmin": -1.0, "xmax": 1.0, "xlabel": "X  (continuous covariate)",
               "ylabel": "π(treat | X)", "ymin": -0.05, "ymax": 1.05, "gammas": [float(g) for g in GAMMAS],
               "defaultGamma": 4.0, "matchedGamma": 4.0, "seriesByGamma": {gk(g): psa(g) for g in GAMMAS},
               "note": "Deployed π(treat|X), N_train=2000. Γ-free: IPW-X-X/Direct-X-X/DoublyRobust-X-X/Oracle; -O-X move with Γ. Wasserstein added later."}

(HERE / "wass_charts.json").write_text(json.dumps(W))
(HERE / "wass_fast.json").write_text(json.dumps({"oracle": v_oracle, "all_treat": v_alltreat, "IPW-X-X": v_xx,
    "Direct-X-X": v_dir, "DoublyRobust-X-X": v_drx, "gammas": GAMMAS,
    "IPW-O-X": vs(ox), "Hajek-O-X": vs(hox), "DoublyRobust-O-X": vs(dro)}, indent=2))

# PNGs
def png(name, c):
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); MK = {"square": "s", "circle": "o"}
    for s in c["series"]:
        ax.plot(c["x"], s["y"], marker=MK.get(s.get("marker"), ""), ms=4, lw=s.get("width", 2.0), color=s["color"], label=s["label"])
    for h in c.get("hlines", []): ax.axhline(h["y"], ls="--", lw=1, color=h["color"])
    ax.set_xlabel(c["xlabel"]); ax.set_ylabel(c["ylabel"]); ax.set_ylim(c["ymin"], c["ymax"]); ax.legend(fontsize=7); ax.grid(alpha=.25); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / (name + ".png"), dpi=120); plt.close(fig)
for nm, c in W.items():
    if "seriesByGamma" not in c: png(nm, c)

# splice WASS_CHARTS + renderChart calls
idx = ROOT / "index.html"; h = idx.read_text()
line = "var WASS_CHARTS = %s;" % json.dumps(W, separators=(",", ":"))
if re.search(r'^var WASS_CHARTS = .*;$', h, flags=re.M):
    h = re.sub(r'^var WASS_CHARTS = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var HR_CHARTS = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
calls = "\n".join("renderChart('ichart-wass-%s',WASS_CHARTS.%s);" % (k, k) for k in W)
h = re.sub(r"\nrenderChart\('ichart-wass-[^\n]*", "", h)
if "renderChart('ichart-wass-outcome'" not in h:
    h = h.replace("renderChart('ichart-hr-outcome',HR_CHARTS.outcome);",
                  "renderChart('ichart-hr-outcome',HR_CHARTS.outcome);\n" + calls, 1)
idx.write_text(h)
print("spliced WASS_CHARTS + %d calls" % len(W))
