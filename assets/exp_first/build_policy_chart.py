"""Build the two interactive treat-probability policy charts (UNCAPPED + CAPPED) for the 'first experiment' and splice
them into index.html. Each is the standard renderChart format: y = π(treat | X) over the 21-X grid, one line per method,
with a Γ-selector + method-toggle. The plotted policy is the MEAN over all seeds (each method's deployed treat-prob at
each grid X, averaged across seeds) — not a single seed. Targets ichart-first-uncap-policy / ichart-first-cap-policy."""
import json, importlib.util, re
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; NEW = HERE.parents[1]
sp = importlib.util.spec_from_file_location("dgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
seeds = sorted((HERE / "_multiseed").glob("seed*.json"))
D = [json.loads(p.read_text()) for p in seeds]; NS = len(D)
d0 = D[0]; grid = d0["grid"]; G = d0["gammas"]; MG = d0["matched_gamma"]; mi = G.index(MG)
# (label, id, colour) — methods + ceilings; matches the rest of the site's palette
META = [("R-OW", "RW", "#d62728"), ("R-OW-DR", "RW_DR", "#a01f1f"), ("R-O", "RO", "#9467bd"), ("R-O-DR", "RO_DR", "#6d4a7d"),
        ("IPW", "IPW", "#2ca02c"), ("AIPW", "AIPW", "#ff7f0e"), ("Regret-O", "RegretO", "#1f77b4"), ("Hajek-OW", "RegretOW", "#17becf"),
        ("Kallus", "Kallus", "#7f7f7f"), ("Best-means", "BestMeans", "#15803d"), ("Full-info", "FullInfo", "#111111")]
def gkey(g): return str(int(g)) if g == int(g) else str(g)
SEEDKEYS = [str(i) for i in range(NS)] + ["avg"]
SEEDLABEL = {str(i): "Seed %d" % i for i in range(NS)}; SEEDLABEL["avg"] = "Average"
def grid_for(seedkey, regime, gi, lab):
    """treat-prob for method `lab` at Γ-index gi — for a given seed, or the mean over seeds ('avg'). None if absent."""
    if seedkey == "avg":
        arrs = [d["regimes"][regime]["grids"][str(gi)].get(lab) for d in D]
        arrs = [a for a in arrs if a is not None]
        if not arrs: return None
        return [round(float(v), 4) for v in np.mean(np.asarray(arrs, float), axis=0)]
    a = D[int(seedkey)]["regimes"][regime]["grids"][str(gi)].get(lab)
    return None if a is None else [round(float(v), 4) for v in a]
def build(regime):
    sbsg = {}                                         # seriesBySeedGamma: {seedkey: {gkey: [series]}}
    for sk in SEEDKEYS:
        m = {}
        for gi, g in enumerate(G):
            ser = []
            for lab, sid, col in META:
                y = grid_for(sk, regime, gi, lab)
                if y is None: continue
                ser.append({"id": sid, "label": lab, "color": col, "y": y})
            m[gkey(g)] = ser
        sbsg[sk] = m
    return {"x": grid, "xmin": -1.0, "xmax": 1.0, "xlabel": "X", "ylabel": "π(treat | x)", "ymin": 0.0, "ymax": 1.0,
            "gammas": G, "defaultGamma": MG, "matchedGamma": MG,
            "seeds": [{"key": sk, "label": SEEDLABEL[sk]} for sk in SEEDKEYS], "defaultSeed": "avg",
            "seriesBySeedGamma": sbsg,
            "note": "π(treat | X) over the 21-X grid · pick seed / Average, Γ and methods"}
UNCAP = build("uncap"); CAP = build("cap")
(HERE / "policy_uncap.json").write_text(json.dumps(UNCAP)); (HERE / "policy_cap.json").write_text(json.dumps(CAP))
# ---- splice into index.html (idempotent: replaces existing var + call if present) ----
idx = NEW / "index.html"; h = idx.read_text()
def splice(h, var, data, cid):
    line = "var %s = %s;" % (var, json.dumps(data, separators=(',', ':')))
    if re.search(r'^var %s = .*;$' % var, h, flags=re.M):
        h = re.sub(r'^var %s = .*;$' % var, lambda m: line, h, count=1, flags=re.M)
    else:
        anc = "renderChartPanels('ichart-3arm-policy',POLICY_PROB_3ARM);"
        h = h.replace(anc, line + "\n" + anc + "\nrenderChart('%s',%s);" % (cid, var), 1)
    if ("renderChart('%s',%s);" % (cid, var)) not in h:
        h = h.replace("renderChartPanels('ichart-3arm-policy',POLICY_PROB_3ARM);",
                      "renderChartPanels('ichart-3arm-policy',POLICY_PROB_3ARM);\nrenderChart('%s',%s);" % (cid, var), 1)
    return h
h = splice(h, "POLICY_FIRST_UNCAP", UNCAP, "ichart-first-uncap-policy")
h = splice(h, "POLICY_FIRST_CAP", CAP, "ichart-first-cap-policy")
idx.write_text(h)
# ---- regenerate the standalone viewer copy (index.html + auto-open sub-exp_first) ----
INJ = ('<script>window.addEventListener("load",function(){setTimeout(function(){'
       'document.querySelectorAll(".tab-panel").forEach(function(p){p.classList.remove("active")});'
       'var t=document.getElementById("tab-experiment");if(t)t.classList.add("active");'
       'document.querySelectorAll(".subpanel").forEach(function(p){p.classList.remove("active")});'
       'document.getElementById("sub-exp_first").classList.add("active");'
       'document.querySelectorAll(".tab-btn,.subtab-btn").forEach(function(b){b.classList.remove("active")});'
       'if(window.MathJax&&MathJax.typesetPromise)MathJax.typesetPromise();},500)});</script>')
head, _, tail = h.rpartition('</body>')
(NEW / "_exp_first_view.html").write_text(head + INJ + '</body>' + tail)
print("spliced POLICY_FIRST_UNCAP/CAP (%d seeds + Average, %d Γ each) into index.html; regenerated viewer" % (NS, len(G)))
