"""Aggregate the 5-seed runs of the 'first experiment' and draw per-regime performance plots (static):
  realized_vs_gamma_{uncap,cap}.png  — realised test E[Y] vs Γ, all methods + ceilings, ±SD
  realized_bar_{uncap,cap}.png       — realised test E[Y] at matched Γ, bar chart, ±SD
  objective_vs_gamma_{uncap,cap}.png — worst-case in-sample objective vs Γ
  treat_fraction_{uncap,cap}.png     — fraction treated vs Γ (+ cap line)
Also writes exp_first_results.json (aggregated means)."""
import json, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("dgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
seeds = sorted(HERE.glob("_multiseed/seed*.json"))
D = [json.loads(p.read_text()) for p in seeds]
G = D[0]["gammas"]; MG = D[0]["matched_gamma"]; mi = G.index(MG) if MG in G else len(G) - 1
# (data-key, display label, family colour, linestyle, marker) — colour = estimator family, shape = uncertainty set
META = [("R-OW", "IPW-O-W", "#1f77b4", "-", "o"), ("R-OW-DR", "DoublyRobust-O-W", "#2ca02c", "-", "o"),
        ("R-O", "IPW-O-X", "#1f77b4", "-", "s"), ("R-O-DR", "DoublyRobust-O-X", "#2ca02c", "-", "s"),
        ("IPW", "IPW-X-X", "#1f77b4", "--", ""), ("AIPW", "DoublyRobust-X-X", "#2ca02c", "--", ""),
        ("Regret-O", "Hajek-O-X", "#9467bd", "-", "s"), ("Hajek-OW", "Hajek-O-W", "#9467bd", "-", "o"),
        ("Kallus", "Kallus", "#d62728", "-", "^")]
NEVER, TREATALL, OPT = 0.719, 0.762, 0.817   # E[Y] references (μ-based, uncapped)
def series(regime, field):
    """mean,sd over seeds of methods[m][gi][field] -> dict m -> (means[gi], sds[gi])."""
    out = {}
    for m, *_ in META:
        vals = []
        for d in D:
            row = d["regimes"][regime]["methods"][m]
            vals.append([(row[str(gi)][field] if str(gi) in row else row.get(gi, {}).get(field)) for gi in range(len(G))])
        A = np.array([[np.nan if v is None else v for v in r] for r in vals], float)
        out[m] = (np.nanmean(A, 0), np.nanstd(A, 0))
    return out
def ceil_mean(regime, key):
    return float(np.mean([d["regimes"][regime]["ceilings"][key] for d in D]))

agg = {"gammas": G, "matched_gamma": MG, "gamma_conf": D[0]["gamma_conf"], "regimes": {}}
for regime, rlabel in (("uncap", "UNCAPPED  (treat ≤ 100%)"), ("cap", "CAPPED  (treat ≤ 50%)")):
    cap = D[0]["regimes"][regime]["cap"]
    rt = series(regime, "rt_test"); tr = series(regime, "treat"); ob = series(regime, "obj")
    fi = ceil_mean(regime, "full_info_test"); bm = ceil_mean(regime, "best_means_test")
    agg["regimes"][regime] = {"cap": cap, "full_info": fi, "best_means": bm,
                              "rt_test_matched": {m: float(rt[m][0][mi]) for m, *_ in META}}
    # --- realised vs Γ ---
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for m, lbl, c, ls, mk in META:
        y, sd = rt[m]
        ax.plot(G, y, color=c, ls=ls, marker=mk, ms=4, lw=2.1, label=lbl)
        ax.fill_between(G, y - sd, y + sd, color=c, alpha=0.10)
    ax.axhline(bm, color="#15803d", lw=1.4, ls="--", label="best-means %.3f" % bm)
    ax.axhline(TREATALL, color="#9aa3b2", lw=1.2, ls=":", label="treat-all %.3f" % TREATALL)
    ax.axvline(MG, color="#c9ccd6", lw=1.1, ls="--"); ax.text(MG, ax.get_ylim()[0], " matched Γ", fontsize=8, color="#64748b", va="bottom")
    ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_ylabel("realised test E[Y]")
    ax.set_title("First experiment — realised value vs Γ · %s\nmean ± SD over %d seeds · full-info=%.3f (off-scale)" % (rlabel, len(D), fi), fontsize=10.6)
    ax.legend(fontsize=8.2, ncol=2, loc="lower right"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(HERE / ("realized_vs_gamma_%s.png" % regime), dpi=120); plt.close(fig)
    # --- realised bar at matched Γ ---
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    names = [lbl for _, lbl, *_ in META]; ys = [rt[m][0][mi] for m, *_ in META]; es = [rt[m][1][mi] for m, *_ in META]; cs = [c for _, _, c, _, _ in META]
    ax.bar(range(len(names)), ys, yerr=es, color=cs, alpha=.9, capsize=3)
    ax.axhline(bm, color="#15803d", lw=1.4, ls="--", label="best-means %.3f" % bm)
    ax.axhline(TREATALL, color="#9aa3b2", lw=1.2, ls=":", label="treat-all %.3f" % TREATALL)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(min(ys) - 0.03, max(max(ys), bm) + 0.02); ax.set_ylabel("realised test E[Y]")
    ax.set_title("Realised value at matched Γ=%.2f · %s · mean ± SD over %d seeds" % (MG, rlabel, len(D)), fontsize=10.4)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=.2, axis="y")
    fig.tight_layout(); fig.savefig(HERE / ("realized_bar_%s.png" % regime), dpi=120); plt.close(fig)
    # --- objective vs Γ ---
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for m, lbl, c, ls, mk in META:
        if m in ("IPW", "AIPW"): continue
        y, sd = ob[m]
        if np.all(np.isnan(y)): continue
        ax.plot(G, y, color=c, ls=ls, marker=mk, ms=4, lw=2.1, label=lbl); ax.fill_between(G, y - sd, y + sd, color=c, alpha=0.10)
    ax.axhline(0, color="#999", lw=1, ls=":"); ax.axvline(MG, color="#c9ccd6", lw=1.1, ls="--")
    ax.set_xlabel("Γ"); ax.set_ylabel("worst-case in-sample objective")
    ax.set_title("Worst-case objective vs Γ · %s · mean ± SD over %d seeds" % (rlabel, len(D)), fontsize=10.4)
    ax.legend(fontsize=8.4, ncol=2, loc="best"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(HERE / ("objective_vs_gamma_%s.png" % regime), dpi=120); plt.close(fig)
    # --- treat fraction vs Γ ---
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for m, lbl, c, ls, mk in META:
        y, sd = tr[m]
        ax.plot(G, y, color=c, ls=ls, marker=mk, ms=4, lw=2.1, label=lbl)
    ax.axhline(cap[1], color="#dc2626", lw=1.3, ls="--", label="cap = %.2f" % cap[1])
    ax.axhline(11 / 21, color="#15803d", lw=1.1, ls=":", label="optimal frac (X≥0) ≈ 0.52")
    ax.axvline(MG, color="#c9ccd6", lw=1.1, ls="--")
    ax.set_xlabel("Γ"); ax.set_ylabel("fraction treated  (1/n)Σ π(treat|x)"); ax.set_ylim(-0.03, 1.03)
    ax.set_title("Fraction treated vs Γ · %s · mean over %d seeds" % (rlabel, len(D)), fontsize=10.4)
    ax.legend(fontsize=8.2, ncol=2, loc="best"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(HERE / ("treat_fraction_%s.png" % regime), dpi=120); plt.close(fig)
    print("regime %s: R-OW@mΓ=%.3f IPW=%.3f AIPW=%.3f best-means=%.3f" % (regime, rt["R-OW"][0][mi], rt["IPW"][0][mi], rt["AIPW"][0][mi], bm))
(HERE / "exp_first_results.json").write_text(json.dumps(agg, indent=1))
print("perf plots done; wrote exp_first_results.json")
