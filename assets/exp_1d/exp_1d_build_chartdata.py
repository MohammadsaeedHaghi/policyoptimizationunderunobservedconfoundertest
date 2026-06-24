"""Build the interactive-chart JSON for the 1-D experiment from the saved NPZs + the per-Γ grid captures:
  • REALIZED  (per mode, train+test charts vs Γ, method toggle)  — from NPZ
  • OBJECTIVE (per mode, vs Γ, method toggle)                     — from NPZ
  • POLICY    (per mode, π(treat|x), method toggle + Γ selector)  — matched-Γ from NPZ + per-Γ from _bygamma/*.json
Writes assets/exp_1d/_chartdata.json (spliced into index.html). Run AFTER exp_1d_capture_grid.py for all methods."""
import json
import numpy as np
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
DIR={"bern":NEW/"assets"/"exp_1d","uni":NEW/"assets"/"exp_1d_uniformS"}
NPZ={"bern":"exp_1d_data.npz","uni":"exp_1d_uniformS_data.npz"}
METH=["RW","RO","IPW","AIPW","RegretO","RegretOW","RW_DR","RO_DR","Kallus"]
LAB={"RW":"R-OW","RO":"R-O","IPW":"IPW","AIPW":"AIPW","RegretO":"Regret-O","RegretOW":"Hajek-OW","RW_DR":"R-OW-DR","RO_DR":"R-O-DR","Kallus":"Kallus"}
COL={"RW":"#d62728","RO":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","RegretO":"#1f77b4","RegretOW":"#17becf","RW_DR":"#a01f1f","RO_DR":"#6d4a7d","Kallus":"#7f7f7f"}
def r4(a): return [None if (v is None or (isinstance(v,float) and np.isnan(v))) else round(float(v),4) for v in a]
out={"realized":{},"objective":{},"policy":{}}
for mode in ("bern","uni"):
    d=dict(np.load(DIR[mode]/NPZ[mode])); G=[round(float(g),4) for g in d["GAMMAS"]]; mG=round(float(d["matched_Gamma"]),4)
    def gkey(g): g=float(g); return str(int(g)) if g==int(g) else str(g)   # match JS String(Number) exactly
    xticks=[{"x":g,"label":gkey(g)} for g in G]
    # ---- realized: shared focused y across train+test ----
    allv=np.concatenate([d[m+s] for m in METH for s in ("_rt","_rt_train")]+[[d["best_means"],d["best_means_train"]]])
    lo=float(np.nanmin(allv)); hi=float(np.nanmax(allv)); pad=0.02*(hi-lo); ymin=round(lo-pad,4); ymax=round(hi+pad,4)
    def realized_chart(suf,fi,bm,ttl):
        return {"x":G,"xmin":min(G),"xmax":max(G),"xlabel":"Γ (assumed sensitivity)","ylabel":"realised E[Y]",
                "ymin":ymin,"ymax":ymax,"xticks":xticks,"gammaLine":mG,
                "hlines":[{"y":round(float(bm),4),"label":f"best-means {float(bm):.3f}","color":"#2ca02c","dash":"5,4"}],
                "note":f"{ttl} · oracle (max realisable E[Y]) = {float(fi):.3f} (off scale) · matched Γ={mG}",
                "series":[{"id":m,"label":LAB[m],"color":COL[m],"y":r4(d[m+suf])} for m in METH]}
    out["realized"][mode]={"train":realized_chart("_rt_train",d["full_info_train"],d["best_means_train"],"Train (in-sample)"),
                            "test": realized_chart("_rt",      d["full_info"],      d["best_means"],      "Test (deployed)")}
    # ---- objective: dynamic y ----
    ov=np.concatenate([d[m+"_obj"] for m in METH]); olo=float(np.nanmin(ov)); ohi=float(np.nanmax(ov)); op=0.06*(ohi-olo)
    out["objective"][mode]={"x":G,"xmin":min(G),"xmax":max(G),"xlabel":"Γ (assumed sensitivity)","ylabel":"worst-case objective",
        "ymin":round(min(olo-op,-0.02),4),"ymax":round(ohi+op,4),"xticks":xticks,"gammaLine":mG,
        "hlines":[{"y":0,"label":"0","color":"#999","dash":"2,3"}],"note":f"worst-case in-sample objective · matched Γ={mG}",
        "series":[{"id":m,"label":LAB[m],"color":COL[m],"y":r4(d[m+"_obj"])} for m in METH]}
    # ---- policy: matched-Γ series (from NPZ) + per-Γ from _bygamma ----
    xs=[round(float(v),4) for v in d["xs"]]
    byg={}
    for m in METH:
        f=DIR[mode]/"_bygamma"/f"{mode}_{m}.json"
        byg[m]=json.loads(f.read_text())["grids"] if f.exists() else None
    seriesByGamma={}
    for gi,g in enumerate(G):
        seriesByGamma[gkey(g)]=[{"id":m,"label":LAB[m],"color":COL[m],
                                "y":(byg[m][gi] if byg[m] is not None else None)} for m in METH]
    out["policy"][mode]={"x":xs,"xmin":-1.0,"xmax":1.0,"xlabel":"X","ylabel":"π(treat | x)","ymin":0.0,"ymax":1.0,
        "gammas":G,"defaultGamma":mG,"matchedGamma":mG,"seriesByGamma":seriesByGamma,
        "series":[{"id":m,"label":LAB[m],"color":COL[m],"y":(d[m+"_treat_grid"].tolist() if (m+"_treat_grid") in d else None)} for m in METH]}
    miss=[m for m in METH if byg[m] is None]
    print(f"[{mode}] realized y∈[{ymin},{ymax}] · objective y∈[{out['objective'][mode]['ymin']},{out['objective'][mode]['ymax']}] · "
          f"policy per-Γ grids: {len(METH)-len(miss)}/{len(METH)}"+(f" MISSING {miss}" if miss else ""))
(NEW/"assets"/"exp_1d"/"_chartdata.json").write_text(json.dumps(out))
print(f"wrote assets/exp_1d/_chartdata.json ({(NEW/'assets'/'exp_1d'/'_chartdata.json').stat().st_size} bytes)")
