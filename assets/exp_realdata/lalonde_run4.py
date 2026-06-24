"""LaLonde variant — observe FOUR covariates {age, education, married, nodegree}; hide ONLY {re74, re75, black, hispanic}
as the unobserved confounders (per user request). Age & education are binned to 3 levels each; married/nodegree are binary
-> up to 3*3*2*2 = 36 observed cells. Trained on the confounded observational pairing (185 NSW trainees + 420 PSID controls),
evaluated on the randomised experimental NSW by IPW. Swept over Γ. Saves lalonde4_results.json + lalonde4_realized.png +
lalonde4_confounding.png. Usage: python3 lalonde_run4.py"""
import sys, json, importlib.util
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
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
DATA=NEW/"assets"/"exp_realdata"/"data"; K=2; NCTRL=420; CAP=(1.0,1.0); SC=1000.0
GAMMAS=[1.0,1.5,2.0,3.0,5.0,8.0]; SEEDS=list(range(5))
OBSF=['age','education','married','nodegree']; HIDF=['re74','re75','black','hispanic']
exp=pd.read_stata(DATA/"nsw_dw.dta"); psid=pd.read_stata(DATA/"psid_controls.dta")
pooled_age=np.r_[exp.age.values,psid.age.values]; AGE_EDGES=np.quantile(pooled_age,[1/3,2/3]); EDU_EDGES=np.array([10.5,12.5])
def X4(df):
    ab=np.clip(np.digitize(df.age.values.astype(float),AGE_EDGES),0,2)/2.0
    eb=np.clip(np.digitize(df.education.values.astype(float),EDU_EDGES),0,2)/2.0
    return np.column_stack([ab,eb,df.married.values.astype(float),df.nodegree.values.astype(float)])
eX=X4(exp); eXr=np.round(eX,6); eT=exp.treat.values.astype(int); eY=exp.re78.values/SC; e1=float((eT==1).mean())
def realized(pt):
    pt=np.asarray(pt,float); pol=np.where(eT==1,pt,1-pt); prop=np.where(eT==1,e1,1-e1); return float(np.mean(pol*eY/prop))
ctrl=realized(np.zeros(len(exp))); treatall=realized(np.ones(len(exp)))
# best deployable on the 4 observed cells: treat a cell iff its within-cell experimental ATE>0
from collections import defaultdict
cr=defaultdict(list)
for i,r in enumerate(eXr): cr[tuple(r)].append(i)
bestpt=np.zeros(len(exp))
for c,rows in cr.items():
    rows=np.array(rows); t=rows[eT[rows]==1]; ct=rows[eT[rows]==0]
    if len(t) and len(ct) and (eY[t].mean()-eY[ct].mean())>0: bestpt[rows]=1.0
best=realized(bestpt)
diag={c:[float(exp[exp.treat==1][c].mean()),float(psid[c].mean())] for c in OBSF+HIDF}
exp_ate=float((exp[exp.treat==1].re78.mean()-exp[exp.treat==0].re78.mean())/SC)
def deploy_pt(pi,Xtr_r):
    pi=np.asarray(pi,float); seen={}
    for j,r in enumerate(Xtr_r): seen.setdefault(tuple(r),j)
    cells=np.array(list(seen.keys())); reps=np.array([seen[tuple(c)] for c in cells])
    return np.array([pi[1,reps[int(np.argmin(((cells-r)**2).sum(1)))]] for r in eXr])
METHODS=["R-OW","R-O","R-OW-DR","Kallus","IPW","AIPW"]; GVAR={"R-OW","R-O","R-OW-DR","Kallus"}
acc={m:{g:[] for g in GAMMAS} for m in METHODS}; naive=[]; ncells=[]
for seed in SEEDS:
    rng=np.random.default_rng(seed); ci=rng.choice(len(psid),NCTRL,replace=False)
    tr=pd.concat([exp[exp.treat==1],psid.iloc[ci]],ignore_index=True)
    Tt=np.r_[np.ones((exp.treat==1).sum(),int),np.zeros(NCTRL,int)]; Yt=tr.re78.values/SC
    Xtr=X4(tr); Xtr_r=np.round(Xtr,6); naive.append(float(Yt[Tt==1].mean()-Yt[Tt==0].mean())); ncells.append(len(set(map(tuple,Xtr_r))))
    w,_=common.ipw_weights_from_data(Xtr,Tt,K); Dm=common.pairwise_distance_matrix(Xtr)
    eps=tuple(common.tight_epsilon(Dm,Tt,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(Xtr,Tt,Yt,n_arms=K,cross_fit=True)
    v_ipw=realized(deploy_pt(ipw(Xtr,Tt,Yt,w,n_arms=K,cap=CAP,discretize=False).pi,Xtr_r))
    v_aipw=realized(deploy_pt(rodr(Xtr,Tt,Yt,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi,Xtr_r))
    for g in GAMMAS:
        acc["IPW"][g].append(v_ipw); acc["AIPW"][g].append(v_aipw)
        acc["R-OW"][g].append(realized(deploy_pt(row(Xtr,Tt,Yt,w,n_arms=K,Gamma=g,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi,Xtr_r)))
        acc["R-O"][g].append(realized(deploy_pt(ro(Xtr,Tt,Yt,w,n_arms=K,Gamma=g,cap=CAP,discretize=False).pi,Xtr_r)))
        acc["R-OW-DR"][g].append(realized(deploy_pt(rowdr(Xtr,Tt,Yt,w,muhat,n_arms=K,Gamma=g,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi,Xtr_r)))
        th=kal.fit_kallus(Xtr,Tt,Yt,w,n_arms=K,Gamma=g,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
        acc["Kallus"][g].append(realized(kal.predict_kallus(th.theta,eX,("affine",))[:,1]))
    print(f"seed{seed} naiveATE={naive[-1]:.2f} cells={ncells[-1]} | "+" ".join(f"{m}@Γ3={np.mean(acc[m][3.0]):.3f}" for m in METHODS),flush=True)
out={"gammas":GAMMAS,"ctrl":ctrl,"treatall":treatall,"best":best,"exp_ate":exp_ate,
     "naive_obs_ate":float(np.mean(naive)),"n_cells":float(np.mean(ncells)),"diag":diag,"obs":OBSF,"hidden":HIDF,
     "methods":{m:{"mean":[float(np.mean(acc[m][g])) for g in GAMMAS],"sd":[float(np.std(acc[m][g])) for g in GAMMAS]} for m in METHODS}}
(NEW/"assets"/"exp_realdata"/"lalonde4_results.json").write_text(json.dumps(out,indent=1))
# --- realized vs Γ plot ---
COL={"R-OW":"#d62728","R-O":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","R-OW-DR":"#a01f1f","Kallus":"#7f7f7f"}
fig,ax=plt.subplots(figsize=(8.6,4.9))
for m in ["R-OW","R-OW-DR","R-O","Kallus","AIPW","IPW"]:
    y=np.array(out["methods"][m]["mean"]); e=np.array(out["methods"][m]["sd"]); ls="--" if m in("IPW","AIPW") else "-"
    ax.plot(GAMMAS,y,ls+"o",color=COL[m],label=m+(" (Γ-indep.)" if m in("IPW","AIPW") else ""),lw=1.9,ms=4)
    if m in GVAR: ax.fill_between(GAMMAS,y-e,y+e,color=COL[m],alpha=.10)
ax.axhline(treatall,ls="--",c="#2ca02c",lw=1.4,label="treat-everyone (truth-optimal ≈ %.2f)"%treatall)
ax.axhline(ctrl,ls=":",c="#444",lw=1.4,label="never-treat (ctrl = %.2f)"%ctrl)
ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_ylabel("realised policy value on randomised NSW ($1000s)")
ax.set_title("LaLonde — observe {age, education, married, nodegree}; hide {re74, re75, black, hispanic}\nrealised value vs Γ (5 seeds); earnings+race still hidden ⇒ methods stay below truth-optimal",fontsize=10)
ax.legend(fontsize=8.5,ncol=2,loc="best"); ax.grid(alpha=.25); fig.tight_layout()
fig.savefig(NEW/"assets"/"exp_realdata"/"lalonde4_realized.png",dpi=120); plt.close(fig)
# --- confounding diagnostic: hidden vs observed ---
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))
a1.bar([0,1],[diag['re74'][0],diag['re75'][0]],.36,label="treated (NSW)",color="#d62728",edgecolor="#333")
a1.bar([0.4,1.4],[diag['re74'][1],diag['re75'][1]],.36,label="control (PSID)",color="#7f7f7f",edgecolor="#333")
a1.set_xticks([0.2,1.2]); a1.set_xticklabels(["re74 (HIDDEN)","re75 (HIDDEN)"]); a1.set_ylabel("mean prior earnings ($)")
a1.set_title("Hidden earnings confounders (≈10× gap)"); a1.legend(fontsize=9)
binv=["black","hispanic","married","nodegree"]; xb=np.arange(4)
a2.bar(xb-.18,[diag[c][0] for c in binv],.36,label="treated (NSW)",color="#d62728",edgecolor="#333")
a2.bar(xb+.18,[diag[c][1] for c in binv],.36,label="control (PSID)",color="#7f7f7f",edgecolor="#333")
a2.set_xticks(xb); a2.set_xticklabels(["black\n(HIDDEN)","hispanic\n(HIDDEN)","married\n(observed)","nodegree\n(observed)"],fontsize=9)
a2.set_ylabel("proportion"); a2.set_title("Binary covariates: 2 hidden + 2 observed"); a2.legend(fontsize=9)
fig.tight_layout(); fig.savefig(NEW/"assets"/"exp_realdata"/"lalonde4_confounding.png",dpi=120); plt.close(fig)
print("\n=== LaLonde 4-observed (hide re74,re75,black,hispanic), realised value vs Γ ($1000s) ===")
print("refs: never-treat=%.3f  treat-all(truth-opt)=%.3f  best-by-cell=%.3f  | naive obs ATE=%.2f  exp ATE=%.3f  ~cells=%.0f"%(ctrl,treatall,best,out["naive_obs_ate"],exp_ate,np.mean(ncells)))
print("Γ:        "+"  ".join("%6.1f"%g for g in GAMMAS))
for m in ["R-OW","R-O","R-OW-DR","Kallus","AIPW","IPW"]:
    print("%-9s "%m+"  ".join("%6.3f"%v for v in out["methods"][m]["mean"]))
print("saved lalonde4_results.json + lalonde4_realized.png + lalonde4_confounding.png")
