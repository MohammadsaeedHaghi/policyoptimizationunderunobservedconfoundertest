"""Build value-vs-Γ + policy charts (and result tables) for the first-prime XX/OX run, color-coded by family/suffix,
and splice them into the first-prime tab (both regimes).  Reads prime_results.json; run AFTER run_prime.py."""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_first_prime"
sp = importlib.util.spec_from_file_location("pdgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sys.modules["pdgp"] = dgp; sp.loader.exec_module(dgp)
R = json.loads((HERE / "prime_results.json").read_text())
GAMMAS = R["gammas"]; G = np.array(R["grid"]); MG = R["matched_gamma"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]; OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
FAM = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Direct": "#ff7f0e", "Hajek": "#9467bd", "Oracle": "#444444"}
def fam(m): return m.split("-")[0]
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
def ser(idl, y, **kw):
    dd = {"id": idl, "label": idl, "color": FAM.get(fam(idl), "#888"), "y": [round(float(v), 4) for v in y]}; dd.update(kw); return dd

CH = {}
for rname in ("uncap", "cap"):
    Rg = R["regimes"][rname]; orc = Rg["oracle"]
    allv = [v for m in XX + OX for v in Rg["value"][m]] + [orc]
    ymin, ymax = min(allv) - 0.02, max(allv) + 0.02
    series = [ser(m, Rg["value"][m]) for m in XX + OX] + [ser("Oracle", [orc] * len(GAMMAS))]
    CH[rname + "_value"] = {"x": [float(g) for g in GAMMAS], "xmin": float(min(GAMMAS)), "xmax": float(max(GAMMAS)),
        "xlabel": "Γ  (sensitivity-model strength)", "ylabel": "realised test E[Y]", "ymin": round(ymin, 3), "ymax": round(ymax, 3),
        "series": series, "hlines": [{"y": orc, "color": "#94a3b8", "dash": "4,4", "label": "best-means oracle"},
                                     {"y": MG, "color": "#c9ccd6", "dash": "3,3", "label": ""}],
        "note": "%s (cap=%s): realised test E[Y] vs Γ, N_train=%d. Colour = estimator family, shape = uncertainty set (XX dashed, OX squares). Matched Γ≈%.1f." %
                ("UNCAPPED" if rname == "uncap" else "CAPPED (treat<=50%)", tuple(Rg["cap"]), R["N_train"], MG)}
    # policy chart: XX flat + OX per-Γ + oracle step
    oracle_pol = [round(float(v), 4) for v in dgp.oracle_policy(G)]
    def psa(g):
        k = gk(g)
        s = [ser(m, Rg["policy"][m]["_flat"]) for m in XX]
        s += [ser(m, Rg["policy"][m][k]) for m in OX]
        s += [ser("Oracle", oracle_pol)]
        return s
    CH[rname + "_policy"] = {"x": [float(v) for v in G], "xmin": -1.0, "xmax": 1.0, "xlabel": "X  (employability score)",
        "ylabel": "π(treat | X)", "ymin": -0.05, "ymax": 1.05, "gammas": [float(g) for g in GAMMAS],
        "defaultGamma": float(MG), "matchedGamma": float(MG), "seriesByGamma": {gk(g): psa(g) for g in GAMMAS},
        "note": "%s deployed π(treat|X), N_train=%d. XX methods Γ-free; OX move with Γ (dropdown). Oracle = treat iff X>=0." %
                ("UNCAPPED" if rname == "uncap" else "CAPPED", R["N_train"])}

(HERE / "prime_result_charts.json").write_text(json.dumps(CH))

# result tables
def fmt(v): return ("%.3f" % v)
def table(rname):
    Rg = R["regimes"][rname]; orc = Rg["oracle"]
    head = "<tr><th>method</th>" + "".join("<th>Γ=%s</th>" % gk(g) for g in GAMMAS) + "</tr>"
    rows = []
    # best per Γ among all methods (for bold)
    best = [max(Rg["value"][m][i] for m in XX + OX) for i in range(len(GAMMAS))]
    for m in XX + OX:
        vals = Rg["value"][m]
        cells = "".join("<td%s>%s</td>" % (" style='font-weight:700'" if abs(vals[i] - best[i]) < 1e-9 else "", fmt(vals[i])) for i in range(len(GAMMAS)))
        rows.append("<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>%s</td>%s</tr>" % (FAM[fam(m)], m, cells))
    orow = "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>Oracle (best-means)</td>%s</tr>" % (FAM["Oracle"], "".join("<td>%s</td>" % fmt(orc) for _ in GAMMAS))
    cap = "treat &le; 100%" if rname == "uncap" else "treat &le; 50%"
    return ("<table class='restab'>%s%s%s</table>\n<p class='muted' style='margin:6px 0 0'>%s: realised test E[Y] by Γ, N_train=%d, single seed. Bold = best method at that Γ. XX methods are Γ-free; OX sweep Γ.</p>"
            % (head, "".join(rows), orow, cap, R["N_train"]))
(HERE / "prime_table_uncap.html").write_text(table("uncap"))
(HERE / "prime_table_cap.html").write_text(table("cap"))

# PNGs (value charts)
def png(name, c):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for s in c["series"]:
        n = s["id"]; ls = "--" if n.endswith("-X-X") else ("-" if n != "Oracle" else ":")
        mk = "s" if n.endswith("-O-X") else ""
        ax.plot(c["x"], s["y"], color=s["color"], ls=ls, marker=mk, ms=5, lw=2.1, label=n)
    ax.axhline(c["hlines"][0]["y"], ls="--", lw=1, color="#94a3b8")
    ax.set_xlabel(c["xlabel"]); ax.set_ylabel(c["ylabel"]); ax.set_ylim(c["ymin"], c["ymax"]); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.25); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / (name + ".png"), dpi=120); plt.close(fig)
png("value_uncap", CH["uncap_value"]); png("value_cap", CH["cap_value"])

# splice var + render calls
idx = ROOT / "index.html"; h = idx.read_text()
line = "var PRIME_RES = %s;" % json.dumps(CH, separators=(",", ":"))
if re.search(r'^var PRIME_RES = .*;$', h, flags=re.M):
    h = re.sub(r'^var PRIME_RES = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var PRIME_CHARTS = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
calls = "\n".join("renderChart('ichart-prime-%s',PRIME_RES.%s);" % (k.replace("_", "-"), k) for k in CH)
h = re.sub(r"\nrenderChart\('ichart-prime-(uncap|cap)-[^\n]*", "", h)
if "renderChart('ichart-prime-uncap-value'" not in h:
    anchor = "renderChart('ichart-prime-obs',PRIME_CHARTS.obs);"
    assert anchor in h, "prime-obs render anchor missing"
    h = h.replace(anchor, anchor + "\n" + calls, 1)
idx.write_text(h)
print("spliced PRIME_RES (%d charts) + tables; keys=%s" % (len(CH), list(CH)))
