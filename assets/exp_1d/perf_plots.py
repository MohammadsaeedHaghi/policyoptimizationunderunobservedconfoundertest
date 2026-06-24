"""Aggregate the multi-seed runs of the 2-arm 1-D experiment and draw per-regime static performance plots, for each
S-variant present (bern -> assets/exp_1d, uni -> assets/exp_1d_uniformS):
  realized_vs_gamma_{uncap,cap}.png  — realised test E[Y] vs Γ, all methods + best-means ceiling, ±SD
  realized_bar_{uncap,cap}.png       — realised test E[Y] at matched Γ, bar chart, ±SD
  objective_vs_gamma_{uncap,cap}.png — worst-case in-sample objective vs Γ
  treat_fraction_{uncap,cap}.png     — fraction treated vs Γ (+ cap line)
Also writes exp_1d_results.json (aggregated means) into each mode dir. Run AFTER exp_1d_multiseed.py for all seeds."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
DIR={"bern":NEW/"assets"/"exp_1d","uni":NEW/"assets"/"exp_1d_uniformS"}
META=[("RW","R-OW","#d62728"),("RW_DR","R-OW-DR","#a01f1f"),("RO","R-O","#9467bd"),("RO_DR","R-O-DR","#6d4a7d"),
      ("IPW","IPW","#2ca02c"),("AIPW","AIPW","#ff7f0e"),("RegretO","Regret-O","#1f77b4"),("RegretOW","Hajek-OW","#17becf"),("Kallus","Kallus","#7f7f7f")]
for mode in ("bern","uni"):
    seeds=sorted((DIR[mode]/"_multiseed").glob(f"{mode}_seed*.json"))
    if not seeds: continue
    D=[json.loads(p.read_text()) for p in seeds]
    if "regimes" not in D[0]: print(f"[{mode}] skipped (old-schema multiseed files)"); continue
    G=D[0]["gammas"]; MG=D[0]["matched_gamma"]; mi=G.index(MG) if MG in G else len(G)-1
    def series(regime,field):
        out={}
        for sid,_,_ in META:
            vals=[]
            for d in D:
                row=d["regimes"][regime]["methods"][sid]
                vals.append([row[str(gi)][field] for gi in range(len(G))])
            A=np.array([[np.nan if v is None else v for v in r] for r in vals],float)
            out[sid]=(np.nanmean(A,0),np.nanstd(A,0))
        return out
    def ceil_mean(regime,key): return float(np.mean([d["regimes"][regime]["ceilings"][key] for d in D]))
    agg={"mode":mode,"gammas":G,"matched_gamma":MG,"regimes":{}}
    for regime,rlabel in (("uncap","UNCAPPED  (treat ≤ 100%)"),("cap","CAPPED  (treat ≤ 50%)")):
        cap=D[0]["regimes"][regime]["cap"]
        rt=series(regime,"rt_test"); trf=series(regime,"treat"); ob=series(regime,"obj")
        fi=ceil_mean(regime,"full_info_test"); bm=ceil_mean(regime,"best_means_test")
        agg["regimes"][regime]={"cap":cap,"full_info":fi,"best_means":bm,
                                "rt_test_matched":{lab:float(rt[sid][0][mi]) for sid,lab,_ in META}}
        # realised vs Γ
        fig,ax=plt.subplots(figsize=(9.2,5.0))
        for sid,lab,c in META:
            y,sd=rt[sid]; ax.plot(G,y,"-o",color=c,ms=4,lw=2.1,label=lab); ax.fill_between(G,y-sd,y+sd,color=c,alpha=0.10)
        ax.axhline(bm,color="#15803d",lw=1.4,ls="--",label="best-means %.3f"%bm)
        ax.axvline(MG,color="#c9ccd6",lw=1.1,ls="--"); ax.text(MG,ax.get_ylim()[0]," matched Γ",fontsize=8,color="#64748b",va="bottom")
        ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_ylabel("realised test E[Y]")
        ax.set_title("2-arm 1-D (%s) — realised value vs Γ · %s\nmean ± SD over %d seeds · full-info=%.3f (off-scale)"%(mode,rlabel,len(D),fi),fontsize=10.4)
        ax.legend(fontsize=8.2,ncol=2,loc="lower right"); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(DIR[mode]/("realized_vs_gamma_%s.png"%regime),dpi=120); plt.close(fig)
        # realised bar at matched Γ
        fig,ax=plt.subplots(figsize=(9.2,4.6))
        names=[lab for _,lab,_ in META]; ys=[rt[sid][0][mi] for sid,_,_ in META]; es=[rt[sid][1][mi] for sid,_,_ in META]; cs=[c for _,_,c in META]
        ax.bar(range(len(names)),ys,yerr=es,color=cs,alpha=.9,capsize=3)
        ax.axhline(bm,color="#15803d",lw=1.4,ls="--",label="best-means %.3f"%bm)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=30,ha="right",fontsize=9)
        ax.set_ylim(min(ys)-0.03,max(max(ys),bm)+0.02); ax.set_ylabel("realised test E[Y]")
        ax.set_title("Realised value at matched Γ=%.2f · %s · %s · mean ± SD over %d seeds"%(MG,mode,rlabel,len(D)),fontsize=10.2)
        ax.legend(fontsize=9,loc="lower right"); ax.grid(alpha=.2,axis="y")
        fig.tight_layout(); fig.savefig(DIR[mode]/("realized_bar_%s.png"%regime),dpi=120); plt.close(fig)
        # objective vs Γ
        fig,ax=plt.subplots(figsize=(9.2,4.8))
        for sid,lab,c in META:
            if sid in ("IPW","AIPW"): continue
            y,sd=ob[sid]
            if np.all(np.isnan(y)): continue
            ax.plot(G,y,"-o",color=c,ms=4,lw=2.1,label=lab); ax.fill_between(G,y-sd,y+sd,color=c,alpha=0.10)
        ax.axhline(0,color="#999",lw=1,ls=":"); ax.axvline(MG,color="#c9ccd6",lw=1.1,ls="--")
        ax.set_xlabel("Γ"); ax.set_ylabel("worst-case in-sample objective")
        ax.set_title("Worst-case objective vs Γ · %s · %s · mean ± SD over %d seeds"%(mode,rlabel,len(D)),fontsize=10.2)
        ax.legend(fontsize=8.4,ncol=2,loc="best"); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(DIR[mode]/("objective_vs_gamma_%s.png"%regime),dpi=120); plt.close(fig)
        # treat fraction vs Γ
        fig,ax=plt.subplots(figsize=(9.2,4.6))
        for sid,lab,c in META:
            y,sd=trf[sid]; ax.plot(G,y,"-o",color=c,ms=4,lw=2.1,label=lab)
        ax.axhline(cap[1],color="#dc2626",lw=1.3,ls="--",label="cap = %.2f"%cap[1]); ax.axvline(MG,color="#c9ccd6",lw=1.1,ls="--")
        ax.set_xlabel("Γ"); ax.set_ylabel("fraction treated  (1/n)Σ π(treat|x)"); ax.set_ylim(-0.03,1.03)
        ax.set_title("Fraction treated vs Γ · %s · %s · mean over %d seeds"%(mode,rlabel,len(D)),fontsize=10.2)
        ax.legend(fontsize=8.2,ncol=2,loc="best"); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(DIR[mode]/("treat_fraction_%s.png"%regime),dpi=120); plt.close(fig)
        print("[%s/%s] R-OW@mΓ=%.3f IPW=%.3f AIPW=%.3f RW_DR=%.3f best-means=%.3f"%(mode,regime,rt["RW"][0][mi],rt["IPW"][0][mi],rt["AIPW"][0][mi],rt["RW_DR"][0][mi],bm))
    (DIR[mode]/"exp_1d_results.json").write_text(json.dumps(agg,indent=1))
    print("[%s] perf plots done; wrote exp_1d_results.json"%mode)
