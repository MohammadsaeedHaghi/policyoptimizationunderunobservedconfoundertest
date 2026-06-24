"""Aggregate _multiseed/seed*.json (5 seeds) → mean±SD, and build the interactive-chart JSON (_chartdata.json):
REALIZED (train+test, mean±SD bands), OBJECTIVE (mean±SD), POLICY (seed-0 per-Γ grids → method toggle + Γ selector).
Same structure the index.html renderChart consumes. Run after run_multiseed.py for all 5 seeds."""
import json, glob
import numpy as np
from pathlib import Path
HERE=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1/assets/exp_nonmono")
METH=["RW","RO","IPW","AIPW","RegretO","RegretOW","RW_DR","RO_DR","Kallus"]
LAB={"RW":"R-OW","RO":"R-O","IPW":"IPW","AIPW":"AIPW","RegretO":"Regret-O","RegretOW":"Hajek-OW","RW_DR":"R-OW-DR","RO_DR":"R-O-DR","Kallus":"Kallus"}
COL={"RW":"#d62728","RO":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","RegretO":"#1f77b4","RegretOW":"#17becf","RW_DR":"#a01f1f","RO_DR":"#6d4a7d","Kallus":"#7f7f7f"}
seeds=[json.loads(open(f).read()) for f in sorted(glob.glob(str(HERE/"_multiseed"/"seed*.json")))]
nS=len(seeds); G=[round(float(g),4) for g in seeds[0]["gammas"]]; mi=4; mG=G[mi]
def gkey(g): g=float(g); return str(int(g)) if g==int(g) else str(g)
xticks=[{"x":g,"label":gkey(g)} for g in G]
def ms(arrs):
    a=np.array([[np.nan if v is None else v for v in row] for row in arrs],float); return np.nanmean(a,0),np.nanstd(a,0)
def r4(x): return [None if (v is None or np.isnan(v)) else round(float(v),4) for v in x]
M={};SD={}
for m in METH:
    for fld in ("rt","rt_train","obj"): mean,sd=ms([s["methods"][m][fld] for s in seeds]); M[(m,fld)]=mean; SD[(m,fld)]=sd
cl={k:float(np.mean([s["ceilings"][k] for s in seeds])) for k in seeds[0]["ceilings"]}
out={"realized":{},"objective":{},"policy":{}}
# realized
allhi=[];alllo=[]
for m in METH:
    for fld in ("rt","rt_train"): allhi.append(np.nanmax(M[(m,fld)]+SD[(m,fld)])); alllo.append(np.nanmin(M[(m,fld)]-SD[(m,fld)]))
lo=min(alllo+[cl["best_means"],cl["best_means_train"]]); hi=max(allhi+[cl["best_means"],cl["best_means_train"]]); pad=0.03*(hi-lo)
ymin=round(lo-pad,4); ymax=round(hi+pad,4)
def realized_chart(fld,fi,bm,ttl):
    return {"x":G,"xmin":min(G),"xmax":max(G),"xlabel":"Γ (assumed sensitivity)","ylabel":"realised E[Y]","ymin":ymin,"ymax":ymax,
            "xticks":xticks,"gammaLine":mG,"hlines":[{"y":round(float(bm),4),"label":f"best-means {float(bm):.3f}","color":"#2ca02c","dash":"5,4"}],
            "note":f"{ttl} · mean ± SD over {nS} seeds · oracle (max E[Y]) = {float(fi):.3f} (off scale) · matched Γ={mG}",
            "series":[{"id":m,"label":LAB[m],"color":COL[m],"y":r4(M[(m,fld)]),"ysd":r4(SD[(m,fld)])} for m in METH]}
out["realized"]["nm"]={"train":realized_chart("rt_train",cl["full_info_train"],cl["best_means_train"],"Train (in-sample)"),
                       "test": realized_chart("rt",      cl["full_info"],      cl["best_means"],      "Test (deployed)")}
ov_hi=max(np.nanmax(M[(m,"obj")]+SD[(m,"obj")]) for m in METH); ov_lo=min(np.nanmin(M[(m,"obj")]-SD[(m,"obj")]) for m in METH); op=0.06*(ov_hi-ov_lo)
out["objective"]["nm"]={"x":G,"xmin":min(G),"xmax":max(G),"xlabel":"Γ (assumed sensitivity)","ylabel":"worst-case objective",
    "ymin":round(min(ov_lo-op,-0.02),4),"ymax":round(ov_hi+op,4),"xticks":xticks,"gammaLine":mG,
    "hlines":[{"y":0,"label":"0","color":"#999","dash":"2,3"}],"note":f"worst-case in-sample objective · mean ± SD over {nS} seeds · matched Γ={mG}",
    "series":[{"id":m,"label":LAB[m],"color":COL[m],"y":r4(M[(m,"obj")]),"ysd":r4(SD[(m,"obj")])} for m in METH]}
# policy (seed 0 per-Γ grids)
s0=next(s for s in seeds if s["seed"]==0); xs=s0["xs"]
seriesByGamma={}
for gi,g in enumerate(G):
    seriesByGamma[gkey(g)]=[{"id":m,"label":LAB[m],"color":COL[m],"y":(s0["methods"][m]["grids"][gi] if s0["methods"][m].get("grids") else None)} for m in METH]
out["policy"]["nm"]={"x":xs,"xmin":-1.0,"xmax":1.0,"xlabel":"X","ylabel":"π(treat | x)","ymin":0.0,"ymax":1.0,
    "gammas":G,"defaultGamma":mG,"matchedGamma":mG,"seriesByGamma":seriesByGamma,
    "note":f"π(treat | x) over the 21-X grid · seed 0 · pick Γ and methods","series":seriesByGamma[gkey(mG)]}
(HERE/"_chartdata.json").write_text(json.dumps(out,ensure_ascii=False))
rows=sorted(((LAB[m],M[(m,"rt")][mi],SD[(m,"rt")][mi]) for m in METH),key=lambda r:-r[1])
print(f"{nS} seeds · realized y∈[{ymin},{ymax}] · ceilings control? best-means {cl['best_means']:.3f} oracle {cl['full_info']:.3f}")
print("5-seed mean±SD @matched Γ (TEST): "+" > ".join(f"{n} {mu:.3f}±{sd:.3f}" for n,mu,sd in rows))
print("wrote _chartdata.json")
