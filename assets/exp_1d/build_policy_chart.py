"""Build the interactive treat-probability policy charts (UNCAPPED + CAPPED, for the bern and uni S-variants) of the
2-arm 1-D experiment and splice ALL inline chart vars into index.html, then regenerate the standalone _exp_1d_view.html.
Each policy chart is the standard renderChart format: y = π(treat | X) over the 21-X grid, one line per method + the
Full-info / Best-means ceilings, with a seed-selector (Seed 0..N + Average = mean policy across seeds), a Γ-selector
and a method-toggle. Also re-reads _chartdata.json (written by exp_1d_multiseed_agg.py) to splice the regime-nested
REALIZED_DATA / OBJECTIVE_DATA vars. Run AFTER exp_1d_multiseed.py (all seeds) + exp_1d_multiseed_agg.py.
Usage: python3 build_policy_chart.py"""
import json, re
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent; NEW=HERE.parents[1]
MODES={"bern":NEW/"assets"/"exp_1d","uni":NEW/"assets"/"exp_1d_uniformS"}
# (grid-key in _multiseed grids, display label, series id, colour) — methods + ceilings; matches the site palette
META=[("RW","R-OW","RW","#d62728"),("RW_DR","R-OW-DR","RW_DR","#a01f1f"),("RO","R-O","RO","#9467bd"),("RO_DR","R-O-DR","RO_DR","#6d4a7d"),
      ("IPW","IPW","IPW","#2ca02c"),("AIPW","AIPW","AIPW","#ff7f0e"),("RegretO","Regret-O","RegretO","#1f77b4"),("RegretOW","Hajek-OW","RegretOW","#17becf"),
      ("Kallus","Kallus","Kallus","#7f7f7f"),("Best-means","Best-means","BestMeans","#15803d"),("Full-info","Full-info","FullInfo","#111111")]
def gkey(g): g=float(g); return str(int(g)) if g==int(g) else str(g)
def build_policy(mode, regime):
    seeds=sorted((MODES[mode]/"_multiseed").glob(f"{mode}_seed*.json"))
    D=[json.loads(p.read_text()) for p in seeds]; NS=len(D)
    if NS==0 or "regimes" not in D[0]: return None
    grid=D[0]["grid"]; G=D[0]["gammas"]; MG=D[0]["matched_gamma"]
    SEEDKEYS=[str(i) for i in range(NS)]+["avg"]
    SEEDLABEL={str(i):f"Seed {i}" for i in range(NS)}; SEEDLABEL["avg"]="Average"
    def grid_for(sk,gi,key):
        if sk=="avg":
            arrs=[d["regimes"][regime]["grids"][str(gi)].get(key) for d in D]; arrs=[a for a in arrs if a is not None]
            if not arrs: return None
            return [round(float(v),4) for v in np.mean(np.asarray(arrs,float),axis=0)]
        a=D[int(sk)]["regimes"][regime]["grids"][str(gi)].get(key)
        return None if a is None else [round(float(v),4) for v in a]
    sbsg={}
    for sk in SEEDKEYS:
        m={}
        for gi,g in enumerate(G):
            ser=[]
            for key,lab,sid,col in META:
                y=grid_for(sk,gi,key)
                if y is None: continue
                ser.append({"id":sid,"label":lab,"color":col,"y":y})
            m[gkey(g)]=ser
        sbsg[sk]=m
    return {"x":grid,"xmin":-1.0,"xmax":1.0,"xlabel":"X","ylabel":"π(treat | x)","ymin":0.0,"ymax":1.0,
            "gammas":G,"defaultGamma":MG,"matchedGamma":MG,
            "seeds":[{"key":sk,"label":SEEDLABEL[sk]} for sk in SEEDKEYS],"defaultSeed":"avg","seriesBySeedGamma":sbsg,
            "note":"π(treat | X) over the 21-X grid · pick seed / Average, Γ and methods"}
CAP={}; UNCAP={}
for mode in MODES:
    if not (MODES[mode]/"_multiseed").exists(): continue
    c=build_policy(mode,"cap"); u=build_policy(mode,"uncap")
    if c is not None: CAP[mode]=c; (MODES[mode]/"policy_cap.json").write_text(json.dumps(c))
    if u is not None: UNCAP[mode]=u; (MODES[mode]/"policy_uncap.json").write_text(json.dumps(u))
# realized / objective (regime-nested) from _chartdata.json
cd=json.loads((NEW/"assets"/"exp_1d"/"_chartdata.json").read_text())
REAL=cd["realized"]; OBJ=cd["objective"]
# ---- splice into index.html ----
idx=NEW/"index.html"; h=idx.read_text()
def repl_or_insert(h,var,data,after=None):
    line="var %s = %s;"%(var,json.dumps(data,separators=(',',':')))
    if re.search(r'^var %s = .*;$'%re.escape(var),h,flags=re.M):
        return re.sub(r'^var %s = .*;$'%re.escape(var),lambda m:line,h,count=1,flags=re.M)
    if after is not None:
        return re.sub(r'^(var %s = .*;)$'%re.escape(after),lambda m:m.group(1)+"\n"+line,h,count=1,flags=re.M)
    return h
h=repl_or_insert(h,"REALIZED_DATA",REAL)
h=repl_or_insert(h,"OBJECTIVE_DATA",OBJ)
h=repl_or_insert(h,"POLICY_EXP1D_CAP",CAP,after="POLICY_DATA")
h=repl_or_insert(h,"POLICY_EXP1D_UNCAP",UNCAP,after="POLICY_EXP1D_CAP")
idx.write_text(h)
# ---- regenerate standalone viewer (auto-open sub-exp_1d + capped inner tab) ----
INJ=('<script>window.addEventListener("load",function(){setTimeout(function(){'
     'document.querySelectorAll(".tab-panel").forEach(function(p){p.classList.remove("active")});'
     'var t=document.getElementById("tab-experiment");if(t)t.classList.add("active");'
     'document.querySelectorAll(".subpanel").forEach(function(p){p.classList.remove("active")});'
     'var s=document.getElementById("sub-exp_1d");if(s)s.classList.add("active");'
     'document.querySelectorAll(".tab-btn,.subtab-btn").forEach(function(b){b.classList.remove("active")});'
     'if(window.MathJax&&MathJax.typesetPromise)MathJax.typesetPromise();},500)});</script>')
head,_,tail=h.rpartition('</body>')
(NEW/"_exp_1d_view.html").write_text(head+INJ+'</body>'+tail)
print("spliced REALIZED_DATA, OBJECTIVE_DATA, POLICY_EXP1D_CAP/UNCAP (modes=%s) into index.html; regenerated _exp_1d_view.html"%(",".join(sorted(CAP))))
