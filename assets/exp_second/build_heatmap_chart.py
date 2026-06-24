"""Build the two interactive 2-D policy-heatmap charts (UNCAPPED + CAPPED) for the 'second experiment' from the
per-seed grids in seed*.json, and splice them into index.html. Each shows π(treat | X1,X2) over the 11×11 grid with
method + Γ + seed (Seed 0..4 / Average) dropdowns. Rendered by renderHeatmap('ichart-second-{uncap,cap}-hm', ...)."""
import json, importlib.util, re
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; NEW = HERE.parents[1]
sp = importlib.util.spec_from_file_location("dgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
seeds = sorted((HERE / "_multiseed").glob("seed*.json")); D = [json.loads(p.read_text()) for p in seeds]; NS = len(D)
d0 = D[0]; AX = d0["axis"]; G = d0["gammas"]; MG = d0["matched_gamma"]
METHODS = ["R-OW", "R-OW-DR", "R-O", "R-O-DR", "IPW", "AIPW", "Regret-O", "Hajek-OW", "Kallus", "Best-means", "Full-info"]
SEEDKEYS = [str(i) for i in range(NS)] + ["avg"]
SEEDLABEL = {str(i): "Seed %d" % i for i in range(NS)}; SEEDLABEL["avg"] = "Average"
def gkey(g): return str(int(g)) if g == int(g) else str(g)
def grid_for(seedkey, regime, gi, lab):
    if seedkey == "avg":
        arrs = [d["regimes"][regime]["grids"][str(gi)].get(lab) for d in D]
        arrs = [a for a in arrs if a is not None]
        if not arrs: return None
        return [round(float(v), 2) for v in np.mean(np.asarray(arrs, float), axis=0)]
    a = D[int(seedkey)]["regimes"][regime]["grids"][str(gi)].get(lab)
    return None if a is None else [round(float(v), 2) for v in a]
def build(regime):
    gridBy = {}
    for sk in SEEDKEYS:
        gm = {}
        for gi, g in enumerate(G):
            mm = {}
            for lab in METHODS:
                y = grid_for(sk, regime, gi, lab)
                if y is not None: mm[lab] = y
            gm[gkey(g)] = mm
        gridBy[sk] = gm
    return {"ax1": AX, "ax2": AX, "methods": [{"id": m, "label": m} for m in METHODS],
            "gammas": G, "defaultGamma": MG, "matchedGamma": MG,
            "seeds": [{"key": sk, "label": SEEDLABEL[sk]} for sk in SEEDKEYS], "defaultSeed": "avg",
            "gridBy": gridBy,
            "note": "π(treat | X₁,X₂) over the 11×11 grid · pick method / Γ / seed (or Average)"}
UNCAP = build("uncap"); CAP = build("cap")
(HERE / "heatmap_uncap.json").write_text(json.dumps(UNCAP)); (HERE / "heatmap_cap.json").write_text(json.dumps(CAP))
idx = NEW / "index.html"; h = idx.read_text()
def splice(h, var, data, cid):
    line = "var %s = %s;" % (var, json.dumps(data, separators=(',', ':')))
    if re.search(r'^var %s = .*;$' % var, h, flags=re.M):
        h = re.sub(r'^var %s = .*;$' % var, lambda m: line, h, count=1, flags=re.M)
    else:
        anc = "renderChart('ichart-first-cap-policy',POLICY_FIRST_CAP);"
        h = h.replace(anc, line + "\n" + anc + "\nrenderHeatmap('%s',%s);" % (cid, var), 1)
    if ("renderHeatmap('%s',%s);" % (cid, var)) not in h:
        h = h.replace("renderChart('ichart-first-cap-policy',POLICY_FIRST_CAP);",
                      "renderChart('ichart-first-cap-policy',POLICY_FIRST_CAP);\nrenderHeatmap('%s',%s);" % (cid, var), 1)
    return h
h = splice(h, "HEATMAP_SECOND_UNCAP", UNCAP, "ichart-second-uncap-hm")
h = splice(h, "HEATMAP_SECOND_CAP", CAP, "ichart-second-cap-hm")
idx.write_text(h)
# ---- regenerate the standalone viewer copy (index.html + auto-open sub-exp_second) ----
INJ = ('<script>window.addEventListener("load",function(){setTimeout(function(){'
       'document.querySelectorAll(".tab-panel").forEach(function(p){p.classList.remove("active")});'
       'var t=document.getElementById("tab-experiment");if(t)t.classList.add("active");'
       'document.querySelectorAll(".subpanel").forEach(function(p){p.classList.remove("active")});'
       'document.getElementById("sub-exp_second").classList.add("active");'
       'document.querySelectorAll(".tab-btn,.subtab-btn").forEach(function(b){b.classList.remove("active")});'
       'if(window.MathJax&&MathJax.typesetPromise)MathJax.typesetPromise();},500)});</script>')
head, _, tail = h.rpartition('</body>')
(NEW / "_exp_second_view.html").write_text(head + INJ + '</body>' + tail)
print("spliced HEATMAP_SECOND_UNCAP/CAP (%d methods × %d Γ × %d seed-keys) into index.html; regenerated viewer" % (len(METHODS), len(G), len(SEEDKEYS)))
