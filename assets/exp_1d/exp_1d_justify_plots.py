"""Justification plots for the 1-D 2-arm experiment (for the advisor write-up):
  (1) RESCALED realised-outcome-vs-Γ (train|test) — y-axis focused on the METHODS so differences are visible
      (the clairvoyant Full-info ceiling is annotated off-scale instead of compressing the axis).
  (2) ABLATION bars — IPW → AIPW → R-O-DR → R-OW-DR, isolating the outcome-model rung and the robustness rung
      (box-O vs +Wasserstein-W), bern vs uni side by side.
  (3) ROBUSTNESS-vs-TYPICAL-Γ — realised test vs Γ with the worst-case matched Γ AND the typical (median per-unit)
      Γ marked: the robust optimum peaks near the typical confounding, not the worst-case (the over-hedging story).
Reads the saved NPZs (no method re-run). Run once: python3 exp_1d_justify_plots.py"""
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
DIR={"bern":NEW/"assets"/"exp_1d","uni":NEW/"assets"/"exp_1d_uniformS"}
NPZ={"bern":"exp_1d_data.npz","uni":"exp_1d_uniformS_data.npz"}
STITLE={"bern":"Bernoulli S","uni":"uniform-11 S"}
GRID=np.round(np.linspace(-1,1,21),6); S_GRID=np.round(np.arange(0,1.0001,0.1),6); gamma=5.0; n_tr=800; seed=0
mG=float(np.exp(gamma/2))                                   # worst-case matched Γ = 12.18
METH=["RW","RO","IPW","AIPW","RegretO","RegretOW","RW_DR","RO_DR","Kallus"]
LAB={"RW":"R-OW","RO":"R-O","IPW":"IPW","AIPW":"AIPW","RegretO":"Regret-O","RegretOW":"Hajek-OW","RW_DR":"R-OW-DR","RO_DR":"R-O-DR","Kallus":"Kallus"}
COL={"RW":"#d62728","RO":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","RegretO":"#1f77b4","RegretOW":"#17becf","RW_DR":"#a01f1f","RO_DR":"#6d4a7d","Kallus":"#7f7f7f"}
MK ={"RW":"o","RO":"s","IPW":"^","AIPW":"p","RegretO":"v","RegretOW":"D","RW_DR":"P","RO_DR":"*","Kallus":"X"}
D={k:dict(np.load(DIR[k]/NPZ[k])) for k in ("bern","uni")}
def typ_gamma(mode):
    rng=np.random.default_rng(seed); rng.choice(GRID,size=n_tr)
    S=(rng.uniform(size=n_tr)<0.5).astype(float) if mode=="bern" else rng.choice(S_GRID,size=n_tr)
    return float(np.median(np.exp(gamma*np.abs(S-0.5))))
TYP={k:typ_gamma(k) for k in ("bern","uni")}

# ---------- (1) RESCALED realised train|test (per mode) ----------
for mode in ("bern","uni"):
    d=D[mode]; G=d["GAMMAS"]
    allv=np.concatenate([d[m+s] for m in METH for s in ("_rt","_rt_train")])
    lo=float(np.nanmin(allv)); hi=float(np.nanmax([np.nanmax(allv),d["best_means"],d["best_means_train"]]))
    pad=0.012*(hi-lo); ylo,yhi=lo-pad,hi+pad
    fig,(axL,axR)=plt.subplots(1,2,figsize=(13.0,5.6),dpi=140,sharey=True)
    for ax,suf,ttl,fi,bm in [(axL,"_rt_train","Train (in-sample)",float(d["full_info_train"]),float(d["best_means_train"])),
                             (axR,"_rt","Test (deployed)",float(d["full_info"]),float(d["best_means"]))]:
        for m in METH: ax.plot(G,d[m+suf],'-',marker=MK[m],color=COL[m],lw=2,ms=6,label=LAB[m])
        ax.axhline(bm,ls='--',lw=1.5,color="#2ca02c",alpha=.9,label=f"Best true-means (deployable) {bm:.3f}")
        ax.axvline(mG,ls=':',color="#888",lw=1.2); ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_title(ttl,fontsize=11); ax.grid(alpha=.25)
        ax.set_ylim(ylo,yhi)
        ax.annotate(f"↑ Full-info (clairvoyant) ceiling = {fi:.3f}  (off scale)",xy=(0.5,0.985),xycoords="axes fraction",
                    ha="center",va="top",fontsize=8.5,color="#444",bbox=dict(boxstyle="round",fc="#f0f0f0",ec="#bbb",alpha=.9))
    axL.set_ylabel("realised outcome  E[Y]   (higher = better)")
    h,l=axL.get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=6,fontsize=8.5,frameon=False,bbox_to_anchor=(0.5,-0.05))
    fig.suptitle(f"1-D 2-arm · {STITLE[mode]} · realised outcome vs Γ — matched Γ={mG:.2f}  ·  "
                 f"ORACLE (max realisable E[Y]) = {float(d['full_info_train']):.3f} train / {float(d['full_info']):.3f} test",y=0.99,fontsize=11.5)
    fig.tight_layout(rect=[0,0.08,1,1]); fig.savefig(DIR[mode]/"realized_train_test.png",bbox_inches="tight"); plt.close(fig)
    print(f"[{mode}] rescaled realized_train_test.png  y∈[{ylo:.3f},{yhi:.3f}]  (was ~[0.60,0.87])",flush=True)

# ---------- (2) ABLATION bars (bern | uni) ----------
LADDER=["IPW","AIPW","RO_DR","RW_DR"]   # propensity → +outcome model → +O box → +W balance
fig,axes=plt.subplots(1,2,figsize=(13.0,5.4),dpi=140,sharey=True)
for ax,mode in zip(axes,("bern","uni")):
    d=D[mode]; mi=4; vals=[float(d[m+"_rt"][mi]) for m in LADDER]
    xpos=np.arange(len(LADDER))
    bars=ax.bar(xpos,vals,width=.62,color=[COL[m] for m in LADDER],edgecolor="white",zorder=3)
    for x,v in zip(xpos,vals): ax.text(x,v+0.002,f"{v:.3f}",ha="center",va="bottom",fontsize=9.5,fontweight="bold")
    aipw=float(d["AIPW_rt"][mi]); ipw=float(d["IPW_rt"][mi]); bm=float(d["best_means"])
    ax.axhline(aipw,ls='--',lw=1.4,color="#ff7f0e",alpha=.8,zorder=1); ax.axhline(bm,ls='--',lw=1.4,color="#2ca02c",alpha=.7,zorder=1)
    ax.text(3.46,aipw,"no-robustness AIPW",fontsize=7.5,color="#cc7000",va="center",ha="right")
    ax.text(3.46,bm,f"best-means {bm:.3f}",fontsize=7.5,color="#2ca02c",va="bottom",ha="right")
    # delta arrows between consecutive rungs
    labels=["+ outcome model μ̂","+ O box (robust)","+ W balance"]
    for i,lab in enumerate(labels):
        v0,v1=vals[i],vals[i+1]; dv=v1-v0; mid=(i+i+1)/2
        ax.annotate("",xy=(i+1,v1),xytext=(i,v0),arrowprops=dict(arrowstyle="->",color="#333",lw=1.3))
        ax.text(mid,max(v0,v1)+0.006,f"{lab}\n{dv:+.3f}",ha="center",va="bottom",fontsize=8,color=("#197d19" if dv>0 else "#b22222"))
    ax.set_xticks(xpos); ax.set_xticklabels([LAB[m] for m in LADDER],fontsize=9.5)
    lo=min(min(vals),ipw)-0.015; hi=max(max(vals),bm,aipw)+0.045
    ax.set_ylim(lo,hi); ax.set_title(f"{STITLE[mode]}",fontsize=11); ax.grid(alpha=.2,axis="y",zorder=0)
axes[0].set_ylabel("realised TEST outcome  E[Y]  at matched Γ")
fig.suptitle("Ablation — what each ingredient adds: propensity (IPW) → +outcome model (AIPW) → +confounding robustness (R-O-DR box, R-OW-DR box∩Wasserstein)",y=0.995,fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(NEW/"assets"/"exp_1d"/"ablation_bars.png",bbox_inches="tight"); plt.close(fig)
import shutil; shutil.copy(NEW/"assets"/"exp_1d"/"ablation_bars.png",NEW/"assets"/"exp_1d_uniformS"/"ablation_bars.png")
print("ablation_bars.png done",flush=True)

# ---------- (3) ROBUSTNESS vs TYPICAL Γ (bern | uni) ----------
SHOW=["IPW","AIPW","RW","RW_DR"]
fig,axes=plt.subplots(1,2,figsize=(13.0,5.2),dpi=140,sharey=True)
for ax,mode in zip(axes,("bern","uni")):
    d=D[mode]; G=d["GAMMAS"]; typ=TYP[mode]
    sub=np.concatenate([d[m+"_rt"] for m in SHOW]+[[d["best_means"]]])
    ax.axvspan(typ,float(max(G)),color="#d62728",alpha=.06,zorder=0)
    for m in SHOW: ax.plot(G,d[m+"_rt"],'-',marker=MK[m],color=COL[m],lw=2.2,ms=6,label=LAB[m])
    ax.axhline(d["best_means"],ls='--',lw=1.4,color="#2ca02c",alpha=.8,label=f"best-means {d['best_means']:.3f}")
    ax.axvline(mG,ls='--',lw=1.8,color="#d62728",label=f"worst-case matched Γ={mG:.1f}")
    ax.axvline(typ,ls='-',lw=1.8,color="#1f77b4",label=f"typical (median) Γ={typ:.1f}")
    ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_title(f"{STITLE[mode]}",fontsize=11); ax.grid(alpha=.25)
    lo=float(np.nanmin(sub))-0.01; hi=float(np.nanmax(sub))+0.012; ax.set_ylim(lo,hi)
    if mode=="uni": ax.text(0.5*(typ+max(G)),lo+0.012,"over-hedging zone\n(Γ above typical)",fontsize=8,color="#b22222",ha="center")
    ax.legend(fontsize=8,loc="lower left")
axes[0].set_ylabel("realised TEST outcome  E[Y]")
fig.suptitle("Where robustness pays vs over-hedges — the robust optimum tracks the TYPICAL confounding, not the worst case",y=0.99,fontsize=11.5)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(NEW/"assets"/"exp_1d"/"robustness_vs_typical.png",bbox_inches="tight"); plt.close(fig)
shutil.copy(NEW/"assets"/"exp_1d"/"robustness_vs_typical.png",NEW/"assets"/"exp_1d_uniformS"/"robustness_vs_typical.png")
print("robustness_vs_typical.png done",flush=True)
print("\nABLATION (test @ matched Γ):")
for mode in ("bern","uni"):
    d=D[mode]; v={m:float(d[m+"_rt"][4]) for m in LADDER}
    print(f"  {mode}: IPW {v['IPW']:.3f} →(+μ̂) AIPW {v['AIPW']:.3f} →(+O) R-O-DR {v['RO_DR']:.3f} →(+W) R-OW-DR {v['RW_DR']:.3f}"
          f"   | μ̂:{v['AIPW']-v['IPW']:+.3f}  O:{v['RO_DR']-v['AIPW']:+.3f}  W:{v['RW_DR']-v['RO_DR']:+.3f}  net-robust:{v['RW_DR']-v['AIPW']:+.3f}")
print("DONE")
