"""Build the interactive RANDOMIZED-POLICY chart (small-multiples: one panel per arm) from seed0.json and splice it
into index.html. Each method's policy is randomized, so we draw the actual probabilities π(arm k | X) — three side-by-side
panels (arm 0 / 1 / 2), sharing ONE method-toggle + ONE Γ-selector. Rendered by renderChartPanels('ichart-3arm-policy',...)."""
import json, importlib.util, re
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent; NEW=HERE.parents[1]
sp=importlib.util.spec_from_file_location("dgp",str(HERE/"dgp.py")); dgp=importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
d=json.loads((HERE/"_multiseed"/"seed0.json").read_text())
grid=d["grid"]; G=dgp.GAMMAS; mi=dgp.mi; cap=d["cap"]; K=dgp.K
# (label, series-id, colour) — order = legend order; the two ceilings + value methods shown by default, the rest hidden
META=[("Best-means","BestMeans","#15803d"),("R-OW","ROW","#d62728"),("R-O","RO","#9467bd"),
      ("AIPW","AIPW","#ff7f0e"),("IPW","IPW","#2ca02c"),
      ("R-OW-DR","ROWDR","#a01f1f"),("R-O-DR","RODR","#6d4a7d"),
      ("Hajek-OW","RegretOW","#17becf"),("Regret-O","RegretO","#1f77b4"),("Full-info","FullInfo","#111111")]
FIXED={"IPW","AIPW","Full-info","Best-means"}                    # Γ-independent policies (policy_grid_fixed)
DEFAULT_HIDDEN=["ROWDR","RODR","RegretOW","RegretO","FullInfo"]  # de-clutter: toggle on in the dropdown
def gkey(g): return str(int(g)) if g==int(g) else str(g)
def row(lab,gi,a):                                               # π(arm a | X) for method `lab` at Γ-index gi
    P=d["policy_grid_fixed"][lab] if lab in FIXED else d["policy_grid_byG"][str(gi)][lab]
    return [round(float(v),4) for v in np.asarray(P)[a]]
byGammaArm={}
for gi,g in enumerate(G):
    byGammaArm[gkey(g)]=[[{"id":sid,"label":lab,"color":col,"y":row(lab,gi,a)} for lab,sid,col in META] for a in range(K)]
c1,c2=round(100*cap[1]),round(100*cap[2])
DATA={"x":grid,"xmin":-1.0,"xmax":1.0,"xlabel":"X","arms":list(range(K)),
      "armTitles":["π(arm 0 | X) — control · uncapped",
                   "π(arm 1 | X) — capped ≈%d%%"%c1,
                   "π(arm 2 | X) — capped ≈%d%%"%c2],
      "armYlabel":["π(arm 0 | X)","π(arm 1 | X)","π(arm 2 | X)"],
      "gammas":G,"defaultGamma":3.0,"matchedGamma":G[mi],   # default Γ=3 (value-peak) shows the randomization best
      "legend":[{"id":sid,"label":lab,"color":col} for lab,sid,col in META],
      "defaultHidden":DEFAULT_HIDDEN,"byGammaArm":byGammaArm,
      "note":"randomized policy π(arm k | X) · seed 0 · pick Γ and toggle methods"}
(HERE/"policy_chart.json").write_text(json.dumps(DATA))
# ---- splice into index.html: swap the var + the render call (handles first run OR re-run) ----
idx=NEW/"index.html"; h=idx.read_text()
varline="var POLICY_PROB_3ARM = "+json.dumps(DATA,separators=(',',':'))+";"
h,n1=re.subn(r'^var POLICY_(?:DATA|PROB)_3ARM = .*;$',lambda m:varline,h,count=1,flags=re.M)
h,n2=re.subn(r"renderChart(?:Panels)?\('ichart-3arm-policy',POLICY_(?:DATA|PROB)_3ARM\);",
             "renderChartPanels('ichart-3arm-policy',POLICY_PROB_3ARM);",h,count=1)
h,n3=re.subn(r'(id="ichart-3arm-policy" style="max-width:)\d+px"','\\g<1>1180px"',h,count=1)
assert n1==1 and n2==1, "splice failed: var=%d call=%d div=%d"%(n1,n2,n3)
idx.write_text(h)
# ---- regenerate the standalone viewer copy (index.html + auto-open-3arm-subtab) so it never goes stale ----
INJ=('<script>window.addEventListener("load",function(){setTimeout(function(){'
     'document.querySelectorAll(".tab-panel").forEach(function(p){p.classList.remove("active")});'
     'var t=document.getElementById("tab-experiment");if(t)t.classList.add("active");'
     'document.querySelectorAll(".subpanel").forEach(function(p){p.classList.remove("active")});'
     'document.getElementById("sub-exp_3arm").classList.add("active");'
     'document.querySelectorAll(".tab-btn,.subtab-btn").forEach(function(b){b.classList.remove("active")});'
     'if(window.MathJax&&MathJax.typesetPromise)MathJax.typesetPromise();},500)});</script>')
head,_,tail=h.rpartition('</body>')
(NEW/"_exp3arm_view.html").write_text(head+INJ+'</body>'+tail)
print("spliced POLICY_PROB_3ARM (%d Γ × %d arms × %d methods); var=%d call=%d divwidth=%d; cap≈(1,%.2f,%.2f); viewer regenerated"
      %(len(byGammaArm),K,len(META),n1,n2,n3,cap[1],cap[2]))
