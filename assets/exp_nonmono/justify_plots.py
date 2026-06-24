"""Justification + static plots for the non-monotone experiment. Reproduces seed-0 data (imports dgp), reads
_multiseed (mean±SD) and _chartdata (seed-0 per-Γ policy grids). Produces:
  data_diagnostics.png  — 4 panels: (A) who's treated + counts, (B) E[S|X,T] edge-selection, (C) observed vs true
                          vs μ̂ treatment mean (non-monotone band), (D) naive vs true vs μ̂ CATE (the trap).
  policy_vs_x.png        — deployed π(treat|x) at matched Γ for all methods vs the true treat-band (the headline:
                          R-OW/R-OW-DR treat the band; Kallus flat; IPW/AIPW mis-target the edges).
  ablation_bars.png      — IPW→AIPW→R-O-DR→R-OW-DR (mean±SD), the Wasserstein term as the differentiator.
  realized_train_test.png— static mean±SD realised plot (for slides).
Run after run_multiseed.py (all seeds) + agg_build.py."""
import sys, json, glob
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1/assets/exp_nonmono")
sys.path.insert(0,str(HERE)); import dgp
GRID=dgp.GRID; mG=dgp.GAMMAS[dgp.mi]
LAB={"RW":"R-OW","RO":"R-O","IPW":"IPW","AIPW":"AIPW","RegretO":"Regret-O","RegretOW":"Hajek-OW","RW_DR":"R-OW-DR","RO_DR":"R-O-DR","Kallus":"Kallus"}
COL={"RW":"#d62728","RO":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","RegretO":"#1f77b4","RegretOW":"#17becf","RW_DR":"#a01f1f","RO_DR":"#6d4a7d","Kallus":"#7f7f7f"}
MK ={"RW":"o","RO":"s","IPW":"^","AIPW":"p","RegretO":"v","RegretOW":"D","RW_DR":"P","RO_DR":"*","Kallus":"X"}
METH=["RW","RO","IPW","AIPW","RegretO","RegretOW","RW_DR","RO_DR","Kallus"]
# reproduce seed-0 train data + μ̂
rng=np.random.default_rng(0); tr=dgp.generate(dgp.n_tr,rng); te=dgp.generate(dgp.n_te,rng)
X=tr.X.ravel(); S=tr.S; T=tr.T; Y=tr.Y; xs=np.sort(np.unique(np.round(X,6)))
mu0_true=np.array([0.5*dgp.pa(x,0,0)+0.5*dgp.pa(x,1,0) for x in xs]); mu1_true=np.array([0.5*dgp.pa(x,0,1)+0.5*dgp.pa(x,1,1) for x in xs])
muh={}
for k in range(2):
    idx=(T==k); m=LogisticRegression(max_iter=2000).fit(X[idx].reshape(-1,1),Y[idx].astype(int)); muh[k]=m.predict_proba(xs.reshape(-1,1))[:,1]
cells={x:(np.round(X,6)==x) for x in xs}
ntreat=np.array([int((cells[x]&(T==1)).sum()) for x in xs]); nctrl=np.array([int((cells[x]&(T==0)).sum()) for x in xs])
phat=np.array([(T[cells[x]]==1).mean() if cells[x].any() else np.nan for x in xs])
ES1=np.array([S[cells[x]&(T==1)].mean() if (cells[x]&(T==1)).any() else np.nan for x in xs])
ES0=np.array([S[cells[x]&(T==0)].mean() if (cells[x]&(T==0)).any() else np.nan for x in xs])
obs1=np.array([Y[cells[x]&(T==1)].mean() if (cells[x]&(T==1)).any() else np.nan for x in xs])
obs0=np.array([Y[cells[x]&(T==0)].mean() if (cells[x]&(T==0)).any() else np.nan for x in xs])
naive=obs1-obs0; truec=mu1_true-mu0_true; hatc=muh[1]-muh[0]
# ---- data_diagnostics ----
fig,ax=plt.subplots(2,2,figsize=(13.2,8.6),dpi=140)
axA=ax[0,0]; axA.bar(xs-0.022,ntreat,width=0.044,color="#d62728",alpha=.55,label="# treated"); axA.bar(xs+0.022,nctrl,width=0.044,color="#9aa0a6",alpha=.55,label="# control")
axA.set_xlabel("X"); axA.set_ylabel("# train units"); axt=axA.twinx(); axt.plot(xs,phat,'-o',color="#111",lw=2,ms=3,label="P(T=1|X)"); axt.set_ylim(0,1); axt.set_ylabel("P(T=1|X)")
axA.set_title("(A) Who is treated, and data per cell",fontsize=11); h1,l1=axA.get_legend_handles_labels(); h2,l2=axt.get_legend_handles_labels(); axA.legend(h1+h2,l1+l2,fontsize=8,loc="upper center")
axB=ax[0,1]; axB.plot(xs,ES1,'-o',color="#d62728",lw=2,ms=4,label="E[S | X, T=1] (treated)"); axB.plot(xs,ES0,'-s',color="#9467bd",lw=2,ms=4,label="E[S | X, T=0]")
axB.axhline(0.5,ls='--',color="#2ca02c",lw=1.4,label="E[S]=0.5"); axB.fill_between(xs,ES0,ES1,where=ES1>=ES0,color="#d62728",alpha=.10)
axB.set_xlabel("X"); axB.set_ylabel("E[unobserved S | X, arm]"); axB.set_ylim(0,1); axB.set_title("(B) Confounding: treated units are high-S, concentrated at the EDGES",fontsize=10.5); axB.grid(alpha=.25); axB.legend(fontsize=8)
axC=ax[1,0]; axC.plot(xs,obs1,'o',color="#d62728",ms=4,alpha=.8,label="observed treated mean"); axC.plot(xs,mu1_true,'-',color="#d62728",lw=2.4,label="TRUE treat μ₁(X) (a middle BUMP)")
axC.plot(xs,muh[1],'--',color="#ff7f0e",lw=2.2,label="ESTIMATED μ̂₁(X) — logistic, monotone"); axC.plot(xs,mu0_true,'-',color="#555",lw=1.8,label="control μ₀≈0.5")
axC.set_xlabel("X"); axC.set_ylabel("outcome E[Y]"); axC.set_ylim(0,1.02); axC.set_title("(C) Treatment helps a MIDDLE band; μ̂ (logistic) can't bend to it",fontsize=10.5); axC.grid(alpha=.25); axC.legend(fontsize=7.5,loc="upper right")
axD=ax[1,1]; axD.plot(xs,naive,'-o',color="#111",lw=2,ms=3,label="NAIVE CATE (obs treat − obs control)"); axD.plot(xs,truec,'-',color="#2ca02c",lw=2.6,label="TRUE CATE = μ₁−μ₀"); axD.plot(xs,hatc,'--',color="#ff7f0e",lw=2.2,label="μ̂ CATE (monotone)")
axD.axhline(0,color="#888",lw=1); axD.fill_between(xs,0,truec,where=truec>0,color="#2ca02c",alpha=.10)
axD.set_xlabel("X"); axD.set_ylabel("CATE"); axD.set_title("(D) Optimal = treat where true CATE>0 (the band); μ̂/naive point the wrong way",fontsize=10); axD.grid(alpha=.25); axD.legend(fontsize=7.5,loc="upper right")
fig.suptitle(f"What the methods see — non-monotone DGP training data (seed 0, n={dgp.n_tr})",y=0.995,fontsize=13)
fig.tight_layout(rect=[0,0,1,0.98]); fig.savefig(HERE/"data_diagnostics.png",bbox_inches="tight"); plt.close(fig)
print("data_diagnostics.png done")
# ---- policy_vs_x (deployed π(treat|x) at matched Γ, from _chartdata seed-0 grids) ----
cd=json.loads((HERE/"_chartdata.json").read_text()); pol=cd["policy"]["nm"]; gx=pol["x"]
sm={s["id"]:s["y"] for s in pol["series"]}
fig,ax=plt.subplots(figsize=(8.8,5.0),dpi=140)
band=(mu1_true>mu0_true); ax.fill_between(xs,-0.05,1.05,where=band,color="#2ca02c",alpha=.08,label="true treat-band (optimal)")
for m in METH:
    if sm.get(m) is None: continue
    ax.plot(gx,sm[m],'-',marker=MK[m],color=COL[m],lw=2,ms=4.5,label=LAB[m],alpha=.95)
ax.set_xlabel("X"); ax.set_ylabel("π(treat | x)"); ax.set_ylim(-.05,1.05)
ax.set_title(f"Deployed policy π(treat|x) at matched Γ={mG} — R-OW/R-OW-DR treat the band; Kallus flat; IPW/AIPW mis-target")
ax.grid(alpha=.25); ax.legend(fontsize=7.5,ncol=2,loc="upper right")
fig.tight_layout(); fig.savefig(HERE/"policy_vs_x.png",bbox_inches="tight"); plt.close(fig); print("policy_vs_x.png done")
# ---- ablation bars + realized (mean±SD from _multiseed) ----
seeds=[json.loads(open(f).read()) for f in sorted(glob.glob(str(HERE/"_multiseed"/"seed*.json")))]; nS=len(seeds); mi=dgp.mi
def ms(m,f): a=np.array([[np.nan if v is None else v for v in s["methods"][m][f]] for s in seeds]); return np.nanmean(a,0),np.nanstd(a,0)
cl={k:float(np.mean([s["ceilings"][k] for s in seeds])) for k in seeds[0]["ceilings"]}
LADDER=["IPW","AIPW","RO_DR","RW_DR"]
fig,ax=plt.subplots(figsize=(7.6,5.2),dpi=140)
vals=[ms(m,"rt")[0][mi] for m in LADDER]; sds=[ms(m,"rt")[1][mi] for m in LADDER]; xp=np.arange(4)
ax.bar(xp,vals,yerr=sds,width=.6,color=[COL[m] for m in LADDER],edgecolor="white",capsize=4,zorder=3)
for x,v in zip(xp,vals): ax.text(x,v+0.004,f"{v:.3f}",ha="center",fontsize=10,fontweight="bold")
aipw=ms("AIPW","rt")[0][mi]; ax.axhline(aipw,ls='--',lw=1.3,color="#ff7f0e",alpha=.8); ax.axhline(cl["best_means"],ls='--',lw=1.3,color="#2ca02c",alpha=.7)
ax.text(3.45,aipw,"AIPW (no robustness)",fontsize=7.5,color="#cc7000",va="bottom",ha="right"); ax.text(3.45,cl["best_means"],f"best-means {cl['best_means']:.3f}",fontsize=7.5,color="#2ca02c",va="bottom",ha="right")
for i,labd in enumerate(["+ μ̂","+ O box","+ W balance"]):
    ax.annotate("",xy=(i+1,vals[i+1]),xytext=(i,vals[i]),arrowprops=dict(arrowstyle="->",color="#333",lw=1.2))
    ax.text((2*i+1)/2,max(vals[i],vals[i+1])+0.012,f"{labd}\n{vals[i+1]-vals[i]:+.3f}",ha="center",fontsize=8,color=("#197d19" if vals[i+1]>vals[i] else "#b22222"))
ax.set_xticks(xp); ax.set_xticklabels([LAB[m] for m in LADDER]); ax.set_ylabel("realised TEST E[Y] (mean±SD, matched Γ)")
ax.set_ylim(min(vals)-0.02,max(vals+[cl["best_means"]])+0.04); ax.set_title("Ablation — the W (Wasserstein) balance is the winning ingredient"); ax.grid(alpha=.2,axis="y")
fig.tight_layout(); fig.savefig(HERE/"ablation_bars.png",bbox_inches="tight"); plt.close(fig); print("ablation_bars.png done")
# realized static
G=[round(float(g),4) for g in seeds[0]["gammas"]]
fig,(axL,axR)=plt.subplots(1,2,figsize=(13.0,5.6),dpi=140,sharey=True)
allv=np.concatenate([ms(m,f)[0] for m in METH for f in("rt","rt_train")])
ylo=float(np.nanmin(allv))-0.02; yhi=max(float(np.nanmax(allv)),cl["best_means"],cl["best_means_train"])+0.02
for axx,fld,ttl,fi,bm in [(axL,"rt_train","Train",cl["full_info_train"],cl["best_means_train"]),(axR,"rt","Test",cl["full_info"],cl["best_means"])]:
    for m in METH:
        mu,sd=ms(m,fld); axx.plot(G,mu,'-',marker=MK[m],color=COL[m],lw=2,ms=5,label=LAB[m]); axx.fill_between(G,mu-sd,mu+sd,color=COL[m],alpha=.12)
    axx.axhline(bm,ls='--',lw=1.4,color="#2ca02c",label=f"best-means {bm:.3f}"); axx.axvline(mG,ls=':',color="#888",lw=1)
    axx.set_xlabel("Γ"); axx.set_title(f"{ttl} · oracle {fi:.3f} (off scale)",fontsize=10.5); axx.grid(alpha=.25); axx.set_ylim(ylo,yhi)
axL.set_ylabel("realised E[Y] (mean ± SD over %d seeds)"%nS); h,l=axL.get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=6,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.04))
fig.suptitle("Non-monotone DGP · realised outcome vs Γ (mean ± SD) — R-OW & R-OW-DR on top",y=0.99,fontsize=12)
fig.tight_layout(rect=[0,0.07,1,1]); fig.savefig(HERE/"realized_train_test.png",bbox_inches="tight"); plt.close(fig); print("realized_train_test.png done")
print("DONE")
