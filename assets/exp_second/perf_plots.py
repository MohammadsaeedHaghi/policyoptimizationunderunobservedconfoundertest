"""Aggregate the 5-seed runs of the 'second experiment' (2-arm, 2-D) and draw per-regime performance plots (static):
  realized_vs_gamma_{uncap,cap}.png  · realized_bar_{uncap,cap}.png · objective_vs_gamma_{uncap,cap}.png · treat_fraction_{uncap,cap}.png
Also writes exp_second_results.json (aggregated means)."""
import json, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("dgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
seeds = sorted(HERE.glob("_multiseed/seed*.json")); D = [json.loads(p.read_text()) for p in seeds]
G = D[0]["gammas"]; MG = D[0]["matched_gamma"]; mi = G.index(MG)
META = [("R-OW", "#d62728"), ("R-OW-DR", "#a01f1f"), ("R-O", "#9467bd"), ("R-O-DR", "#6d4a7d"),
        ("IPW", "#2ca02c"), ("AIPW", "#ff7f0e"), ("Regret-O", "#1f77b4"), ("Hajek-OW", "#17becf"), ("Kallus", "#7f7f7f")]
NEVER, TREATALL, OPT = 0.622, 0.615, 0.732   # E[Y] references (μ-based, unconstrained)
def series(regime, field):
    out = {}
    for m, _ in META:
        vals = [[d["regimes"][regime]["methods"][m][str(gi)].get(field) for gi in range(len(G))] for d in D]
        A = np.array([[np.nan if v is None else v for v in r] for r in vals], float)
        out[m] = (np.nanmean(A, 0), np.nanstd(A, 0))
    return out
def ceil_mean(regime, key): return float(np.mean([d["regimes"][regime]["ceilings"][key] for d in D]))
agg = {"gammas": G, "matched_gamma": MG, "gamma_conf": D[0]["gamma_conf"], "regimes": {}}
for regime, rlabel in (("uncap", "UNCAPPED  (treat ≤ 100%)"), ("cap", "CAPPED  (treat ≤ 50%)")):
    cap = D[0]["regimes"][regime]["cap"]
    rt = series(regime, "rt_test"); tr = series(regime, "treat"); ob = series(regime, "obj")
    fi = ceil_mean(regime, "full_info_test"); bm = ceil_mean(regime, "best_means_test")
    agg["regimes"][regime] = {"cap": cap, "full_info": fi, "best_means": bm,
                              "rt_test_matched": {m: float(rt[m][0][mi]) for m, _ in META}}
    # realised vs Γ
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for m, c in META:
        y, sd = rt[m]; ax.plot(G, y, "-o", color=c, ms=4, lw=2.1, label=m); ax.fill_between(G, y - sd, y + sd, color=c, alpha=0.10)
    ax.axhline(bm, color="#15803d", lw=1.4, ls="--", label="best-means %.3f" % bm)
    ax.axhline(TREATALL, color="#9aa3b2", lw=1.2, ls=":", label="treat-all %.3f" % TREATALL)
    ax.axvline(MG, color="#c9ccd6", lw=1.1, ls="--")
    ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_ylabel("realised test E[Y]")
    ax.set_title("Second experiment (2-D) — realised value vs Γ · %s\nmean ± SD over %d seeds · full-info=%.3f (off-scale)" % (rlabel, len(D), fi), fontsize=10.6)
    ax.legend(fontsize=8.2, ncol=2, loc="lower right"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(HERE / ("realized_vs_gamma_%s.png" % regime), dpi=120); plt.close(fig)
    # realised bar at matched Γ
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    names = [m for m, _ in META]; ys = [rt[m][0][mi] for m, _ in META]; es = [rt[m][1][mi] for m, _ in META]; cs = [c for _, c in META]
    ax.bar(range(len(names)), ys, yerr=es, color=cs, alpha=.9, capsize=3)
    ax.axhline(bm, color="#15803d", lw=1.4, ls="--", label="best-means %.3f" % bm)
    ax.axhline(TREATALL, color="#9aa3b2", lw=1.2, ls=":", label="treat-all %.3f" % TREATALL)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(min(ys) - 0.03, max(max(ys), bm) + 0.02); ax.set_ylabel("realised test E[Y]")
    ax.set_title("Realised value at matched Γ=%.2f · %s · mean ± SD over %d seeds" % (MG, rlabel, len(D)), fontsize=10.4)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=.2, axis="y")
    fig.tight_layout(); fig.savefig(HERE / ("realized_bar_%s.png" % regime), dpi=120); plt.close(fig)
    # objective vs Γ
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for m, c in META:
        if m in ("IPW", "AIPW"): continue
        y, sd = ob[m]
        if np.all(np.isnan(y)): continue
        ax.plot(G, y, "-o", color=c, ms=4, lw=2.1, label=m); ax.fill_between(G, y - sd, y + sd, color=c, alpha=0.10)
    ax.axhline(0, color="#999", lw=1, ls=":"); ax.axvline(MG, color="#c9ccd6", lw=1.1, ls="--")
    ax.set_xlabel("Γ"); ax.set_ylabel("worst-case in-sample objective")
    ax.set_title("Worst-case objective vs Γ · %s · mean ± SD over %d seeds" % (rlabel, len(D)), fontsize=10.4)
    ax.legend(fontsize=8.4, ncol=2, loc="best"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(HERE / ("objective_vs_gamma_%s.png" % regime), dpi=120); plt.close(fig)
    # treat fraction vs Γ
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for m, c in META:
        y, sd = tr[m]; ax.plot(G, y, "-o", color=c, ms=4, lw=2.1, label=m)
    ax.axhline(cap[1], color="#dc2626", lw=1.3, ls="--", label="cap = %.2f" % cap[1])
    ax.axhline(0.47, color="#15803d", lw=1.1, ls=":", label="optimal frac (proj>0.04) ≈ 0.47")
    ax.axvline(MG, color="#c9ccd6", lw=1.1, ls="--")
    ax.set_xlabel("Γ"); ax.set_ylabel("fraction treated  (1/n)Σ π(treat|x)"); ax.set_ylim(-0.03, 1.03)
    ax.set_title("Fraction treated vs Γ · %s · mean over %d seeds" % (rlabel, len(D)), fontsize=10.4)
    ax.legend(fontsize=8.2, ncol=2, loc="best"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(HERE / ("treat_fraction_%s.png" % regime), dpi=120); plt.close(fig)
    print("regime %s: R-OW@mΓ=%.3f R-O=%.3f IPW=%.3f AIPW=%.3f best-means=%.3f" % (regime, rt["R-OW"][0][mi], rt["R-O"][0][mi], rt["IPW"][0][mi], rt["AIPW"][0][mi], bm))
(HERE / "exp_second_results.json").write_text(json.dumps(agg, indent=1))
print("perf plots done; wrote exp_second_results.json")
