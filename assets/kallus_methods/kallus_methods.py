"""Run OUR method suite on the Kallus & Zhou §7.1.1 DGP and compare, in the 2-arm 1-D experiment plot structure.

Methods: R-OW, R-O, IPW, Regret-O, Hajek-OW, R-OW-DoublyRobust, R-O-DoublyRobust, and our parametric Kallus.
The two DoublyRobust methods use an ESTIMATED cross-fitted outcome model μ̂_k(X) (LinearRegression per arm on the
observed (X,T,reward) — NEVER the oracle risk); everything else is unchanged. Free-π LP methods are solved on the
n=400 training points (uncapped) and deployed to the 5-D-continuous test set by 1-NN; policy plots project onto the
true CATE axis. Single replication (seed 0). Authoritative generator for assets/kallus_methods/* (all 5 figures,
the NPZ, the CSV, and the kalcate_data.json / summary.json the index.html table + interactive chart read)."""
import sys, time, importlib.util, csv, json
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
ASSETS=NEW/"assets"/"kallus_methods"; ASSETS.mkdir(parents=True,exist_ok=True)
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
row=load("row","methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py").solve_ipw_o_w_uncapped
ro =load("ro","methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py").solve_ipw_o_x_uncapped
ipw=load("ipw","methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py").solve_ipw_x_x_uncapped
rgo=load("rgo","methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py").solve_hajek_o_x_uncapped
rgw=load("rgw","methods/Hajek-O-W/Uncapped/hajek_o_w_uncapped.py").solve_hajek_o_w_uncapped
rowdr=load("rowdr","methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py").solve_doublyrobust_o_w_uncapped
rodr =load("rodr","methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py").solve_doublyrobust_o_x_uncapped
kal=load("kal","methods/Kallus/kallus.py"); fit_kallus=kal.fit_kallus; predict_kallus=kal.predict_kallus
# ---- Kallus §7.1.1 DGP (verbatim from assets/kallus_repro/kallus_repro.py) ----
BETA_CONS=2.5; BETA_X=np.array([0,.5,-0.5,0,0,0.0]); BETA_X_T=np.array([-1.5,1,-1.5,1.,0.5,0.0])
BETA_TC=np.array([0,.75,-.5,0,-1,0.0]); MU_X=np.array([-1,.5,-1,0,-1.0]); ALPHA=-2.0; WCOEF=1.5; LG_TRUE=1.5
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def get_bnds(Q,LG): Q=np.asarray(Q,float); eL=np.exp(LG); eLm=np.exp(-LG); return 1.0/(eL*Q/(1-Q+eL*Q)),1.0/(eLm*Q/(1-Q+eLm*Q))
def real_risk(T,xa,u): T=np.asarray(T,float); return T*BETA_CONS+xa@BETA_X+(xa@BETA_X_T)*T+ALPHA*u*(2*T-1)+WCOEF*u
def generate(n,rng):
    u=(rng.random(n)>0.5).astype(float); x=rng.standard_normal((n,5))+MU_X*(2*u-1)[:,None]; xa=np.hstack([x,np.ones((n,1))])
    nominal=sig(xa@BETA_TC); optT=(real_risk(np.ones(n),xa,u)<real_risk(np.zeros(n),xa,u)).astype(int)
    a_bnd,b_bnd=get_bnds(nominal,LG_TRUE); trueQ=np.where(optT==1,1/a_bnd,1/b_bnd)
    T=(rng.random(n)<trueQ).astype(int); Y=real_risk(T,xa,u)+2*rng.standard_normal(n)
    return x,xa,u,T,Y
K=2; n_tr=400; n_te=4000; seed=0
GAMMAS=[1.0,2.0,3.0,round(float(np.exp(LG_TRUE)),4),6.0,9.0]; mi=3   # matched Gamma ~ e^{1.5}=4.48 (odds ratio)
t0=time.time(); rng=np.random.default_rng(seed)
xtr,xatr,utr,Ttr,Ytr=generate(n_tr,rng); xte,xate,ute,Tte,Yte=generate(n_te,rng)
Yrew=-Ytr                                                  # reward = -risk (all methods maximise value)
w,_=common.ipw_weights_from_data(xtr,Ttr,K)
# ESTIMATED outcome model μ̂_k(X) in REWARD units — honest cross-fit, from observed (X,T,reward) only (NO oracle)
muhat_tr=common.outcome_means(xtr,Ttr,Yrew,n_arms=K,cross_fit=True)   # (n_tr, K)
# 1-NN deploy (z-scored Euclidean in 5-D)
mu=xtr.mean(0); sd=xtr.std(0)+1e-9; ztr=(xtr-mu)/sd; zte=(xte-mu)/sd
nn=np.array([int(np.argmin(((ztr-zte[j])**2).sum(1))) for j in range(n_te)])
risk1_tr=real_risk(np.ones(n_tr),xatr,utr); risk0_tr=real_risk(np.zeros(n_tr),xatr,utr); cate_tr=risk1_tr-risk0_tr
risk1_te=real_risk(np.ones(n_te),xate,ute); risk0_te=real_risk(np.zeros(n_te),xate,ute); cate_te=risk1_te-risk0_te
base_tr=float(risk0_tr.mean()); base_te=float(risk0_te.mean())                       # never-treat risk level
orc_tr=float(np.mean(np.where(cate_tr<0,risk1_tr,risk0_tr)))                          # oracle risk level (treat iff CATE<0)
orc_te=float(np.mean(np.where(cate_te<0,risk1_te,risk0_te)))
oracle_te=orc_te-base_te                                                              # oracle regret (test)
def lvl_tr(pi): pi=np.asarray(pi); return float(np.mean(pi[0]*risk0_tr+pi[1]*risk1_tr))     # realised risk level (train)
def lvl_te(pi): pi=np.asarray(pi); pe=pi[:,nn]; return float(np.mean(pe[0]*risk0_te+pe[1]*risk1_te))
def treat_te(pi):pi=np.asarray(pi); return pi[1,nn]                                         # deployed P(treat|x) on test
print(f"[{time.time()-t0:.0f}s] setup n_tr={n_tr} arms={np.bincount(Ttr).tolist()} ATE={cate_te.mean():+.3f} "
      f"oracle_regret={oracle_te:+.3f} base_te={base_te:.3f} orc_te={orc_te:.3f}",flush=True)
# μ̂-implied CATE direction vs true CATE (diagnostic: how well the estimated outcome model orders benefit)
mucate_tr=muhat_tr[:,1]-muhat_tr[:,0]   # reward-CATE = -(risk-CATE); >0 ⇒ μ̂ says treat helps
print(f"  μ̂ reward-CATE sign-agreement with true benefit (train): "
      f"{float(np.mean((mucate_tr>0)==(cate_tr<0))):.3f}",flush=True)
METH=["RW","RO","IPW","RegretO","RegretOW","RW_DR","RO_DR","Kallus"]
LAB={"RW":"R-OW","RO":"R-O","IPW":"IPW","RegretO":"Regret-O","RegretOW":"Hajek-OW",
     "RW_DR":"R-OW-DR","RO_DR":"R-O-DR","Kallus":"Kallus"}
COL={"RW":"#d62728","RO":"#9467bd","IPW":"#2ca02c","RegretO":"#1f77b4","RegretOW":"#17becf",
     "RW_DR":"#a01f1f","RO_DR":"#6d4a7d","Kallus":"#7f7f7f"}
MK ={"RW":"o","RO":"s","IPW":"^","RegretO":"v","RegretOW":"D","RW_DR":"P","RO_DR":"*","Kallus":"X"}
res={m:dict(tr=[],te=[],otr=[],ote=[],obj=[],treat=None) for m in METH}
def safe(fn):
    try: return fn()
    except Exception as e: print("  FAIL:",str(e)[:90]); return None
di=ipw(xtr,Ttr,Yrew,w,n_arms=K,discretize=False)            # IPW is Gamma-independent
for gi,G in enumerate(GAMMAS):
    sv={"RW":safe(lambda:row(xtr,Ttr,Yrew,w,n_arms=K,Gamma=G,discretize=False,zscore=True)),
        "RO":safe(lambda:ro(xtr,Ttr,Yrew,w,n_arms=K,Gamma=G,discretize=False)),"IPW":di,
        "RegretO":safe(lambda:rgo(xtr,Ttr,Yrew,w,n_arms=K,Gamma=G,maximize=True,discretize=False)),
        "RegretOW":safe(lambda:rgw(xtr,Ttr,Yrew,w,n_arms=K,Gamma=G,maximize=True,discretize=False,zscore=True)),
        "RW_DR":safe(lambda:rowdr(xtr,Ttr,Yrew,w,muhat_tr,n_arms=K,Gamma=G,discretize=False,zscore=True)),
        "RO_DR":safe(lambda:rodr(xtr,Ttr,Yrew,w,muhat_tr,n_arms=K,Gamma=G,discretize=False))}
    kth=safe(lambda:fit_kallus(xtr,Ttr,Yrew,w,n_arms=K,Gamma=G,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0).theta)
    kpi_tr=predict_kallus(kth,xtr,("affine",)).T if kth is not None else None
    kpi_te=predict_kallus(kth,xte,("affine",)).T if kth is not None else None
    for m in METH:
        if m=="Kallus":
            if kpi_te is None: [res[m][k].append(np.nan) for k in ("tr","te","otr","ote","obj")]; continue
            otr=float(np.mean(kpi_tr[0]*risk0_tr+kpi_tr[1]*risk1_tr)); ote=float(np.mean(kpi_te[0]*risk0_te+kpi_te[1]*risk1_te))
            res[m]["otr"].append(otr); res[m]["ote"].append(ote); res[m]["tr"].append(otr-base_tr); res[m]["te"].append(ote-base_te)
            res[m]["obj"].append(np.nan)
            if gi==mi: res[m]["treat"]=kpi_te[1]
        else:
            if sv[m] is None: [res[m][k].append(np.nan) for k in ("tr","te","otr","ote","obj")]; continue
            pi=sv[m].pi; otr=lvl_tr(pi); ote=lvl_te(pi)
            res[m]["otr"].append(otr); res[m]["ote"].append(ote); res[m]["tr"].append(otr-base_tr); res[m]["te"].append(ote-base_te)
            res[m]["obj"].append(float(sv[m].objective_value))
            if gi==mi: res[m]["treat"]=treat_te(pi)
    print(f"[{time.time()-t0:.0f}s] Γ={G} done  "
          +" ".join(f"{LAB[m]}={res[m]['te'][-1]:+.3f}" for m in METH if not np.isnan(res[m]['te'][-1])),flush=True)
mG=GAMMAS[mi]
# ---- Fig A: realised regret vs Γ, train | test ----
fig,(axL,axR)=plt.subplots(1,2,figsize=(12.6,5.4),dpi=140,sharey=True)
for ax,key,ttl,orc in [(axL,"tr","Train (in-sample)",None),(axR,"te","Test (1-NN deployed)",oracle_te)]:
    for m in METH: ax.plot(GAMMAS,res[m][key],'-',marker=MK[m],color=COL[m],lw=2,ms=5.5,label=LAB[m])
    ax.axhline(0,ls='--',lw=1.4,color="#8c564b",label="never-treat control")
    if orc is not None: ax.axhline(orc,ls='--',lw=1.3,color="#111",label="Oracle (true CATE)")
    ax.axvline(mG,ls=':',color="#888",lw=1); ax.set_xlabel("Γ (marginal-sensitivity box)"); ax.set_title(ttl); ax.grid(alpha=.25)
axL.set_ylabel("policy regret vs control  (lower = better)")
h,l=axR.get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=10,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.03))
fig.suptitle(f"Our methods on the Kallus §7.1.1 DGP — policy regret vs Γ (train vs test) · n={n_tr} · matched Γ≈{mG} (=e^1.5)",y=0.99,fontsize=11.5)
fig.tight_layout(rect=[0,0.07,1,1]); fig.savefig(ASSETS/"realized_train_test.png",bbox_inches="tight"); plt.close(fig)
# ---- Fig B: mean realised outcome (absolute risk level) vs Γ, train | test ----
fig,(axL,axR)=plt.subplots(1,2,figsize=(12.6,5.4),dpi=140,sharey=True)
for ax,key,ttl,bse,orc in [(axL,"otr","Train (in-sample)",base_tr,orc_tr),(axR,"ote","Test (1-NN deployed)",base_te,orc_te)]:
    for m in METH: ax.plot(GAMMAS,res[m][key],'-',marker=MK[m],color=COL[m],lw=2,ms=5.5,label=LAB[m])
    ax.axhline(bse,ls='--',lw=1.4,color="#8c564b",label=f"never-treat control ({bse:.2f})")
    ax.axhline(orc,ls='--',lw=1.3,color="#111",label=f"Oracle (true CATE, {orc:.2f})")
    ax.axvline(mG,ls=':',color="#888",lw=1); ax.set_xlabel("Γ (marginal-sensitivity box)"); ax.set_title(ttl); ax.grid(alpha=.25)
axL.set_ylabel("mean realised risk  (lower = better)")
h,l=axR.get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=6,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.03))
fig.suptitle(f"Our methods on the Kallus §7.1.1 DGP — mean realised outcome vs Γ (train vs test) · n={n_tr} · matched Γ≈{mG}",y=0.99,fontsize=11.5)
fig.tight_layout(rect=[0,0.07,1,1]); fig.savefig(ASSETS/"realized_outcome_vs_gamma.png",bbox_inches="tight"); plt.close(fig)
# ---- binned treat-vs-CATE curves at matched Γ ----
edges=np.quantile(cate_te,np.linspace(0,1,17)); ctr=0.5*(edges[:-1]+edges[1:]); idx=np.clip(np.digitize(cate_te,edges)-1,0,len(ctr)-1)
treat_curves={}
for m in METH:
    if res[m]["treat"] is None: treat_curves[m]=np.full(len(ctr),np.nan); continue
    treat_curves[m]=np.array([res[m]["treat"][idx==b].mean() if (idx==b).any() else np.nan for b in range(len(ctr))])
# ---- Fig C: π(treat) vs true CATE, binned, at matched Γ ----
fig,ax=plt.subplots(figsize=(8.4,5.0),dpi=140)
for m in METH:
    if np.all(np.isnan(treat_curves[m])): continue
    ax.plot(ctr,treat_curves[m],'-',marker=MK[m],color=COL[m],lw=2,ms=5.5,label=LAB[m],alpha=.95)
ax.axvline(0,ls=':',color="#444",lw=1.2); ax.text(0.02,1.02,"treat iff CATE<0 (oracle)",fontsize=8,color="#444")
ax.set_xlabel("true CATE = risk(1) − risk(0)   (← treatment helps | hurts →)"); ax.set_ylabel("π(treat | x)"); ax.set_ylim(-.05,1.05)
ax.set_title(f"Deployed treatment policy vs true CATE at matched Γ={mG}"); ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2,loc="upper right")
fig.tight_layout(); fig.savefig(ASSETS/"policy_treat_cate.png"); plt.close(fig)
# ---- Fig D: per-arm policy π(T=a|x) vs CATE, control | treat, all methods ----
fig,(axC,axT)=plt.subplots(1,2,figsize=(12.6,5.2),dpi=140,sharey=True)
for ax,arm,ttl in [(axC,0,"π(T = control | x)"),(axT,1,"π(T = treat | x)")]:
    for m in METH:
        if np.all(np.isnan(treat_curves[m])): continue
        y=treat_curves[m] if arm==1 else 1.0-treat_curves[m]
        ax.plot(ctr,y,'-',marker=MK[m],color=COL[m],lw=2,ms=5,label=LAB[m],alpha=.95)
    orc=(ctr<0).astype(float) if arm==1 else (ctr>=0).astype(float)
    ax.step(ctr,orc,where='mid',ls='--',lw=1.3,color="#111",label="Oracle (true CATE)")
    ax.axvline(0,ls=':',color="#444",lw=1.1); ax.set_xlabel("true CATE = risk(1) − risk(0)"); ax.set_title(ttl); ax.set_ylim(-.05,1.05); ax.grid(alpha=.25)
axC.set_ylabel("deployed assignment probability")
h,l=axT.get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=9,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.suptitle(f"Per-arm deployed policy π(T=a | x) vs true CATE at matched Γ={mG} — every arm, every method",y=0.99,fontsize=11.5)
fig.tight_layout(rect=[0,0.06,1,1]); fig.savefig(ASSETS/"policy_per_arm_cate.png",bbox_inches="tight"); plt.close(fig)
# ---- Fig E: worst-case objective vs Γ ----
fig,ax=plt.subplots(figsize=(8.0,4.9),dpi=140)
for m in METH:
    if not np.all(np.isnan(res[m]["obj"])): ax.plot(GAMMAS,res[m]["obj"],'-',marker=MK[m],color=COL[m],lw=2,ms=5.5,label=LAB[m])
ax.axhline(0,color="#999",lw=.8); ax.set_xlabel("Γ"); ax.set_ylabel("worst-case objective (reward units)")
ax.set_title("Our methods on Kallus §7.1.1 — worst-case objective vs Γ"); ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2)
fig.tight_layout(); fig.savefig(ASSETS/"objective_vs_gamma.png"); plt.close(fig)
# ---- save NPZ + CSV ----
np.savez(ASSETS/"kallus_methods_data.npz",GAMMAS=np.array(GAMMAS),matched_Gamma=mG,oracle_te=oracle_te,
         base_tr=base_tr,base_te=base_te,orc_tr=orc_tr,orc_te=orc_te,cate_bins=ctr,
         **{f"{m}_tr":np.array(res[m]["tr"]) for m in METH},**{f"{m}_te":np.array(res[m]["te"]) for m in METH},
         **{f"{m}_out_tr":np.array(res[m]["otr"]) for m in METH},**{f"{m}_out_te":np.array(res[m]["ote"]) for m in METH},
         **{f"{m}_obj":np.array(res[m]["obj"]) for m in METH},**{f"{m}_treatcate":treat_curves[m] for m in METH})
with open(ASSETS/"kallus_methods.csv","w",newline="") as f:
    wri=csv.writer(f); wri.writerow(["method","Gamma","regret_train","regret_test","risk_train","risk_test","objective"])
    for m in METH:
        for gi,G in enumerate(GAMMAS): wri.writerow([LAB[m],G,res[m]["tr"][gi],res[m]["te"][gi],res[m]["otr"][gi],res[m]["ote"][gi],res[m]["obj"][gi]])
# ---- kalcate_data.json (interactive chart) + summary.json (table/prose numbers) ----
kc={"x":[round(float(c),3) for c in ctr],"xmin":round(float(ctr.min()),2),"xmax":round(float(ctr.max()),2),
    "xlabel":"true CATE = risk(1)−risk(0)","gamma":mG,
    "series":[{"id":m,"label":LAB[m],"color":COL[m],
               "y":[None if np.isnan(v) else round(float(v),4) for v in treat_curves[m]]} for m in METH]}
(ASSETS/"kalcate_data.json").write_text(json.dumps(kc))
def tmean(treat,msk): t=np.asarray(treat); return float(t[msk].mean()) if msk.any() else float("nan")
neg=cate_te<0; pos=cate_te>=0
summ={"matched_Gamma":mG,"base_tr":base_tr,"base_te":base_te,"orc_tr":orc_tr,"orc_te":orc_te,"oracle_te":oracle_te,
      "mucate_signagree":float(np.mean((mucate_tr>0)==(cate_tr<0))),"GAMMAS":GAMMAS,
      "methods":{LAB[m]:{
          "regret_tr":round(res[m]["tr"][mi],3),"regret_te":round(res[m]["te"][mi],3),
          "risk_tr":round(res[m]["otr"][mi],3),"risk_te":round(res[m]["ote"][mi],3),
          "treat_neg":round(tmean(res[m]["treat"],neg),2) if res[m]["treat"] is not None else None,
          "treat_pos":round(tmean(res[m]["treat"],pos),2) if res[m]["treat"] is not None else None,
          "te_sweep":[None if np.isnan(v) else round(float(v),3) for v in res[m]["te"]],
          "ote_sweep":[None if np.isnan(v) else round(float(v),3) for v in res[m]["ote"]],
      } for m in METH}}
(ASSETS/"summary.json").write_text(json.dumps(summ,indent=1))
print(f"[{time.time()-t0:.0f}s] DONE -> {ASSETS}")
print("== regret @ matched Γ (TEST) ==",{LAB[m]:round(res[m]['te'][mi],3) for m in METH},"| oracle",round(oracle_te,3))
print("== π(treat) | CATE<0 ==",{LAB[m]:summ['methods'][LAB[m]]['treat_neg'] for m in METH})
print("== π(treat) | CATE>0 ==",{LAB[m]:summ['methods'][LAB[m]]['treat_pos'] for m in METH})
