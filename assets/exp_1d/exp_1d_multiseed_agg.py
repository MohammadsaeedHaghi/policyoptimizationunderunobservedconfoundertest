"""Aggregate the multi-seed runs (assets/exp_1d{,_uniformS}/_multiseed/{mode}_seed*.json) into mean ± SD per method
per Γ, for BOTH regimes (uncap, cap), and write the regime-nested REALIZED + OBJECTIVE chart data into _chartdata.json
as cd["realized"][mode][regime] = {train,test} and cd["objective"][mode][regime], each series carrying `y` (mean) and
`ysd` (SD across seeds). Run AFTER exp_1d_multiseed.py for all seeds; then build_policy_chart.py splices into index.html.
Usage: python3 exp_1d_multiseed_agg.py"""
import json, glob
import numpy as np
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
DIR={"bern":NEW/"assets"/"exp_1d","uni":NEW/"assets"/"exp_1d_uniformS"}
METH=["RW","RO","IPW","AIPW","RegretO","RegretOW","RW_DR","RO_DR","Kallus"]
LAB={"RW":"R-OW","RO":"R-O","IPW":"IPW","AIPW":"AIPW","RegretO":"Regret-O","RegretOW":"Hajek-OW","RW_DR":"R-OW-DR","RO_DR":"R-O-DR","Kallus":"Kallus"}
COL={"RW":"#d62728","RO":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","RegretO":"#1f77b4","RegretOW":"#17becf","RW_DR":"#a01f1f","RO_DR":"#6d4a7d","Kallus":"#7f7f7f"}
REGIMES=["uncap","cap"]; REGLAB={"uncap":"uncapped (treat ≤ 100%)","cap":"capped (treat ≤ 50%)"}
cdpath=NEW/"assets"/"exp_1d"/"_chartdata.json"
cd=json.loads(cdpath.read_text()) if cdpath.exists() else {}
for top in ("realized","objective"): cd.setdefault(top,{})
def ms(arrs):  # list of per-seed arrays (may contain None) -> (mean, sd) per index, nan-aware
    a=np.array([[np.nan if v is None else v for v in row] for row in arrs],float)
    return np.nanmean(a,0), np.nanstd(a,0)
def r4(x): return [None if (v is None or np.isnan(v)) else round(float(v),4) for v in x]
def series_arr(seeds,regime,m,fld,ng):
    return [[seeds[si]["regimes"][regime]["methods"][m][str(gi)][fld] for gi in range(ng)] for si in range(len(seeds))]
for mode in ("bern","uni"):
    files=sorted(glob.glob(str(DIR[mode]/"_multiseed"/f"{mode}_seed*.json")))
    if not files: continue
    seeds=[json.loads(open(f).read()) for f in files]; NS=len(seeds)
    if "regimes" not in seeds[0]: print(f"[{mode}] skipped (old-schema multiseed files, no 'regimes')"); continue
    G=[round(float(g),4) for g in seeds[0]["gammas"]]; ng=len(G); mi=4; mG=G[mi]
    def gk(g): g=float(g); return str(int(g)) if g==int(g) else str(g)
    xticks=[{"x":g,"label":gk(g)} for g in G]
    cd["realized"][mode]={}; cd["objective"][mode]={}  # reset this mode (drop any stale flat keys); regimes filled below
    for regime in REGIMES:
        if regime not in seeds[0]["regimes"]: continue
        M={}; SD={}
        for m in METH:
            for fld in ("rt_test","rt_train","obj"):
                mean,sd=ms(series_arr(seeds,regime,m,fld,ng)); M[(m,fld)]=mean; SD[(m,fld)]=sd
        cl={k:float(np.mean([s["regimes"][regime]["ceilings"][k] for s in seeds])) for k in seeds[0]["regimes"][regime]["ceilings"]}
        # realized: shared focused y across train+test, widened to include mean±SD and best-means ceiling
        allhi=[]; alllo=[]
        for m in METH:
            for fld in ("rt_test","rt_train"):
                allhi.append(np.nanmax(M[(m,fld)]+SD[(m,fld)])); alllo.append(np.nanmin(M[(m,fld)]-SD[(m,fld)]))
        lo=min(alllo+[cl["best_means_test"],cl["best_means_train"]]); hi=max(allhi+[cl["best_means_test"],cl["best_means_train"]])
        pad=0.03*(hi-lo); ymin=round(lo-pad,4); ymax=round(hi+pad,4)
        def realized_chart(fld,fi,bm,ttl):
            return {"x":G,"xmin":min(G),"xmax":max(G),"xlabel":"Γ (assumed sensitivity)","ylabel":"realised E[Y]",
                    "ymin":ymin,"ymax":ymax,"xticks":xticks,"gammaLine":mG,
                    "hlines":[{"y":round(float(bm),4),"label":f"best-means {float(bm):.3f}","color":"#15803d","dash":"5,4"}],
                    "note":f"{ttl} · {REGLAB[regime]} · mean ± SD over {NS} seeds · oracle (Full-info) = {float(fi):.3f} (off scale) · matched Γ={mG}",
                    "series":[{"id":m,"label":LAB[m],"color":COL[m],"y":r4(M[(m,fld)]),"ysd":r4(SD[(m,fld)])} for m in METH]}
        cd["realized"][mode][regime]={"train":realized_chart("rt_train",cl["full_info_train"],cl["best_means_train"],"Train (in-sample)"),
                                      "test": realized_chart("rt_test", cl["full_info_test"], cl["best_means_test"], "Test (deployed)")}
        ov_hi=max(np.nanmax(M[(m,"obj")]+SD[(m,"obj")]) for m in METH if not np.all(np.isnan(M[(m,"obj")])))
        ov_lo=min(np.nanmin(M[(m,"obj")]-SD[(m,"obj")]) for m in METH if not np.all(np.isnan(M[(m,"obj")]))); op=0.06*(ov_hi-ov_lo)
        cd["objective"][mode][regime]={"x":G,"xmin":min(G),"xmax":max(G),"xlabel":"Γ (assumed sensitivity)","ylabel":"worst-case objective",
            "ymin":round(min(ov_lo-op,-0.02),4),"ymax":round(ov_hi+op,4),"xticks":xticks,"gammaLine":mG,
            "hlines":[{"y":0,"label":"0","color":"#999","dash":"2,3"}],"note":f"worst-case in-sample objective · {REGLAB[regime]} · mean ± SD over {NS} seeds · matched Γ={mG}",
            "series":[{"id":m,"label":LAB[m],"color":COL[m],"y":r4(M[(m,"obj")]),"ysd":r4(SD[(m,"obj")])} for m in METH]}
        print(f"[{mode}/{regime}] {NS} seeds · realized y∈[{ymin},{ymax}] · @matched Γ test mean±SD: "
              +" ".join(f"{LAB[m]}={M[(m,'rt_test')][mi]:.3f}±{SD[(m,'rt_test')][mi]:.3f}" for m in ["RW","AIPW","RW_DR","Kallus"]))
cdpath.write_text(json.dumps(cd,ensure_ascii=False))
print("updated _chartdata.json (regime-nested realized/objective)")
