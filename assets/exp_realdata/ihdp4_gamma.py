"""IHDP 4-binary-feature experiment swept over Γ (assumed sensitivity). Same setup as ihdp_run4.py (observe x7,x9,x14,x21;
hide the other 21; no cap; 10 reps; realised value via known mu0/mu1; scale-free fraction-of-oracle-gain). The robust
methods R-OW, R-O, R-OW-DR and Kallus are recomputed at each Γ; IPW (nominal value) and AIPW (non-robust DR, =R-O-DR at Γ=1)
are Γ-independent references. Saves ihdp4_gamma_results.json + ihdp4_gamma.png. Usage: python3 ihdp4_gamma.py"""
import sys, json, importlib.util
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
row=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
ro =load("ro","methods/IPW-O-X/Capped/ipw_o_x_capped.py").solve_ipw_o_x_capped
ipw=load("ipw","methods/IPW-X-X/Capped/ipw_x_x_capped.py").solve_ipw_x_x_capped
rowdr=load("rowdr","methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py").solve_doublyrobust_o_w_capped
rodr =load("rodr","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py").solve_doublyrobust_o_x_capped
kal=load("kal","methods/Kallus/kallus.py")
DATA=NEW/"assets"/"exp_realdata"/"data"; K=2; CAP=(1.0,1.0)
GAMMAS=[1.0,2.0,3.0,4.0,6.0,8.0]
cols=['treat','yf','ycf','mu0','mu1']+['x%d'%i for i in range(1,26)]
OBS=[7,9,14,21]                                                      # the 4 observed binary features (from ihdp_run4.py)
GVAR=["R-OW","R-O","R-OW-DR","Kallus"]; GFIX=["IPW","AIPW"]; METHODS=GVAR+GFIX
COL={"R-OW":"#d62728","R-O":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","R-OW-DR":"#a01f1f","Kallus":"#7f7f7f"}
# frac[m][gi] = list over reps
frac={m:{gi:[] for gi in range(len(GAMMAS))} for m in METHODS}
ta_frac=[]
for rep in range(1,11):
    d=pd.read_csv(DATA/f"ihdp_{rep}.csv",header=None,names=cols)
    X=d[['x%d'%j for j in OBS]].values.astype(float)
    T=d['treat'].values.astype(int); Y=d['yf'].values.astype(float)
    mu0=d['mu0'].values.astype(float); mu1=d['mu1'].values.astype(float)
    rng=np.random.default_rng(rep); idx=rng.permutation(len(d)); ntr=int(0.6*len(d)); tri,tei=idx[:ntr],idx[ntr:]
    Xtr,Ttr,Ytr=X[tri],T[tri],Y[tri]; Xte=X[tei]; m0,m1=mu0[tei],mu1[tei]
    c=float(m0.mean()); o=float(np.maximum(m0,m1).mean()); ta=float(m1.mean())
    ta_frac.append((ta-c)/(o-c) if o>c else np.nan)
    w,_=common.ipw_weights_from_data(Xtr,Ttr,K); Dm=common.pairwise_distance_matrix(Xtr)
    eps=tuple(common.tight_epsilon(Dm,Ttr,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(Xtr,Ttr,Ytr,n_arms=K,cross_fit=True)
    seen={}
    for j,r in enumerate(np.round(Xtr,6)): seen.setdefault(tuple(r),j)
    keys=np.array(list(seen.keys())); kcol=np.array([seen[tuple(k)] for k in keys])
    nn=np.array([int(kcol[np.argmin(((keys-r)**2).sum(1))]) for r in np.round(Xte,6)])
    fv=lambda v:(v-c)/(o-c) if o>c else np.nan                       # value -> fraction of oracle gain
    def fr(pi): pi=np.asarray(pi,float); cc=pi[:,nn]; return fv(float((cc[0]*m0+cc[1]*m1).mean()))
    # Γ-independent references
    f_ipw=fr(ipw(Xtr,Ttr,Ytr,w,n_arms=K,cap=CAP,discretize=False).pi)
    f_aipw=fr(rodr(Xtr,Ttr,Ytr,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi)
    for gi,g in enumerate(GAMMAS):
        frac["IPW"][gi].append(f_ipw); frac["AIPW"][gi].append(f_aipw)
        frac["R-OW"][gi].append(fr(row(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=g,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi))
        frac["R-O"][gi].append(fr(ro(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=g,cap=CAP,discretize=False).pi))
        frac["R-OW-DR"][gi].append(fr(rowdr(Xtr,Ttr,Ytr,w,muhat,n_arms=K,Gamma=g,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi))
        th=kal.fit_kallus(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=g,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
        frac["Kallus"][gi].append(fv(float((kal.predict_kallus(th.theta,Xte,("affine",))*np.column_stack([m0,m1])).sum(1).mean())))
    print(f"rep{rep} done",flush=True)
out={"gammas":GAMMAS,"treatall_frac":float(np.nanmean(ta_frac)),
     "methods":{m:{"mean":[float(np.nanmean(frac[m][gi]))*100 for gi in range(len(GAMMAS))],
                   "sd":[float(np.nanstd(frac[m][gi]))*100 for gi in range(len(GAMMAS))]} for m in METHODS}}
(NEW/"assets"/"exp_realdata"/"ihdp4_gamma_results.json").write_text(json.dumps(out,indent=1))
# plot
fig,ax=plt.subplots(figsize=(8.6,4.9))
for m in ["R-OW","R-OW-DR","R-O","Kallus","AIPW","IPW"]:
    y=np.array(out["methods"][m]["mean"]); e=np.array(out["methods"][m]["sd"])
    ls="--" if m in GFIX else "-"
    ax.plot(GAMMAS,y,ls+"o",color=COL[m],label=m+(" (Γ-indep.)" if m in GFIX else ""),lw=1.9,ms=4)
    if m in GVAR: ax.fill_between(GAMMAS,y-e,y+e,color=COL[m],alpha=.10)
ax.axhline(out["treatall_frac"]*100,ls=":",c="#444",lw=1.3,label="treat-everyone (%.0f%%)"%(out["treatall_frac"]*100))
ax.axhline(100,ls=":",c="green",lw=1.0,label="oracle (100%)")
ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_ylabel("% of oracle gain captured (mean ± SD, 10 reps)")
ax.set_title("IHDP, 4 binary features observed — performance vs Γ\nrobust methods hold near treat-everyone across the whole Γ sweep (mild confounding)",fontsize=10.5)
ax.legend(fontsize=8.5,ncol=2,loc="lower left"); ax.grid(alpha=.25); fig.tight_layout()
fig.savefig(NEW/"assets"/"exp_realdata"/"ihdp4_gamma.png",dpi=120); plt.close(fig)
print("\n=== IHDP 4-feature Γ-sweep (fraction of oracle gain %) ===")
print("Γ:        "+"  ".join("%6.1f"%g for g in GAMMAS))
for m in ["R-OW","R-O","R-OW-DR","Kallus","AIPW","IPW"]:
    print("%-9s "%m+"  ".join("%6.1f"%v for v in out["methods"][m]["mean"]))
print("saved ihdp4_gamma_results.json + ihdp4_gamma.png")
