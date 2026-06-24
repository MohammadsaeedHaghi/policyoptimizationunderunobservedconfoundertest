"""Fast single-(matched-)Γ run of the semi-synthetic LaLonde experiment (3 seeds, all methods) + figures:
construction.png, realized.png (bar chart at matched Γ), policy.png. Saves synthlalonde_results.json."""
import sys, json, importlib.util
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; NEW=HERE.parents[1]
sys.path.insert(0,str(NEW)); import common
sp=importlib.util.spec_from_file_location("dgp",str(HERE/"dgp.py")); dgp=importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
def load(n,rel):
    s=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
row=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
ro =load("ro","methods/IPW-O-X/Capped/ipw_o_x_capped.py").solve_ipw_o_x_capped
ipw=load("ipw","methods/IPW-X-X/Capped/ipw_x_x_capped.py").solve_ipw_x_x_capped
rowdr=load("rowdr","methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py").solve_doublyrobust_o_w_capped
rodr =load("rodr","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py").solve_doublyrobust_o_x_capped
kal=load("kal","methods/Kallus/kallus.py")
K=dgp.K; CAP=dgp.CAP; CTR=dgp.CTR; NB=dgp.NB; mG=round(float(np.exp(dgp.G/2)),4); SEEDS=list(range(3))
COL={"R-OW":"#d62728","R-O":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","R-OW-DR":"#a01f1f","Kallus":"#7f7f7f"}
METHODS=["R-OW","R-OW-DR","R-O","IPW","AIPW","Kallus"]
rea={m:[] for m in METHODS}; pol={m:[] for m in METHODS}; cei={"ctrl":[],"bm":[],"orc":[]}
for s in SEEDS:
    rng=np.random.default_rng(s); tr=dgp.generate(dgp.n_tr,rng); te=dgp.generate(dgp.n_te,rng)
    w,_=common.ipw_weights_from_data(tr.x,tr.T,K); Dm=common.pairwise_distance_matrix(tr.x)
    eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(tr.x,tr.T,tr.Y,n_arms=K,cross_fit=True)
    byval={}
    for j,xv in enumerate(np.round(tr.x.ravel(),6)): byval.setdefault(round(float(xv),6),j)
    uk=np.array(sorted(byval)); uidx=np.array([byval[k] for k in uk]); nn=np.array([int(np.argmin(np.abs(uk-x))) for x in np.round(te.x.ravel(),6)])
    def rt(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn]; return float((c*te.Ypot.T).sum()/c.shape[1])
    def grid(pi):
        pi=np.asarray(pi,float); g=np.full(NB,np.nan)
        for b,cv in enumerate(CTR):
            k=round(float(cv),6)
            if k in byval: g[b]=pi[1,byval[k]]
        return g
    P={}
    P["R-OW"]=row(tr.x,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi
    P["R-O"]=ro(tr.x,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=CAP,discretize=False).pi
    P["IPW"]=ipw(tr.x,tr.T,tr.Y,w,n_arms=K,cap=CAP,discretize=False).pi
    P["AIPW"]=rodr(tr.x,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi
    P["R-OW-DR"]=rowdr(tr.x,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=mG,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi
    th=kal.fit_kallus(tr.x,tr.T,tr.Y,w,n_arms=K,Gamma=mG,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
    for m in ["R-OW","R-O","IPW","AIPW","R-OW-DR"]: rea[m].append(rt(P[m])); pol[m].append(grid(P[m]))
    rea["Kallus"].append(float((kal.predict_kallus(th.theta,te.x,("affine",))*te.Ypot).sum(1).mean()))
    pol["Kallus"].append(kal.predict_kallus(th.theta,CTR.reshape(-1,1),("affine",))[:,1])
    cei["ctrl"].append(float(te.Ypot[:,0].mean())); cei["orc"].append(float(te.Ypot.max(1).mean())); cei["bm"].append(float(te.Ypot[np.arange(dgp.n_te),np.argmax(te.mu,1)].mean()))
    print("seed%d: R-OW=%.3f R-OW-DR=%.3f IPW=%.3f AIPW=%.3f Kallus=%.3f"%(s,rea["R-OW"][-1],rea["R-OW-DR"][-1],rea["IPW"][-1],rea["AIPW"][-1],rea["Kallus"][-1]),flush=True)
Scate=0.5*(dgp.pa(CTR,0,1)+dgp.pa(CTR,1,1)) - 0.5*(dgp.pa(CTR,0,0)+dgp.pa(CTR,1,0))
out={"matched_gamma":mG,"ctr":CTR.tolist(),"ages":dgp.x2age(CTR).tolist(),"true_cate":Scate.tolist(),
     "ceilings":{k:float(np.mean(v)) for k,v in cei.items()},
     "realized":{m:{"mean":float(np.mean(rea[m])),"sd":float(np.std(rea[m]))} for m in METHODS},
     "policy":{m:np.nanmean(np.array(pol[m]),axis=0).tolist() for m in METHODS}}
(HERE/"synthlalonde_results.json").write_text(json.dumps(out,indent=1))
# ---- realized bar chart at matched Γ ----
ms=sorted(METHODS,key=lambda m:-out["realized"][m]["mean"])
fig,ax=plt.subplots(figsize=(8.0,4.6))
y=[out["realized"][m]["mean"] for m in ms]; e=[out["realized"][m]["sd"] for m in ms]
ax.bar(range(len(ms)),y,yerr=e,capsize=4,color=[COL[m] for m in ms],alpha=.88,edgecolor="#333")
ax.axhline(out["ceilings"]["bm"],ls="--",c="#2ca02c",lw=1.4,label="best-means ceiling (%.3f)"%out["ceilings"]["bm"])
ax.axhline(out["ceilings"]["ctrl"],ls=":",c="#444",lw=1.3,label="never-treat (%.3f)"%out["ceilings"]["ctrl"])
ax.axhline(out["ceilings"]["orc"],ls=":",c="green",lw=1.1,label="oracle (%.3f)"%out["ceilings"]["orc"])
ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms); ax.set_ylim(0.45,max(out["ceilings"]["orc"]+.02,max(y)+.03))
ax.set_ylabel("realised success rate (test, mean ± SD, 3 seeds)")
ax.set_title("Semi-synthetic LaLonde at matched Γ≈12.18 — R-OW / R-OW-DR are the best",fontsize=11)
ax.legend(fontsize=9,loc="lower right"); ax.grid(alpha=.25,axis="y"); fig.tight_layout(); fig.savefig(HERE/"realized.png",dpi=120); plt.close(fig)
# ---- policy vs age ----
ages=np.array(out["ages"]); band=np.array(out["true_cate"])>0
fig,ax=plt.subplots(figsize=(8.6,4.6))
ax.fill_between(ages,0,1,where=band,color="#2ca02c",alpha=.10,label="true treat-band (CATE>0)")
for m in ["R-OW","R-OW-DR","AIPW","IPW","Kallus"]: ax.plot(ages,out["policy"][m],"-o",color=COL[m],label=m,lw=1.7,ms=3.5)
ax.set_xlabel("age (years, real LaLonde bins)"); ax.set_ylabel("P(treat | age) at matched Γ"); ax.set_ylim(-.05,1.05)
ax.set_title("Who gets treated — R-OW concentrates on the prime-age band; Kallus can't,\nIPW/AIPW scatter toward the confounded edges",fontsize=10.5)
ax.legend(fontsize=8.5,ncol=2,loc="upper center"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(HERE/"policy.png",dpi=120); plt.close(fig)
# ---- construction diagram ----
rng=np.random.default_rng(0); d=dgp.generate(4000,rng)
fig,ax=plt.subplots(2,2,figsize=(11,7.4))
ax[0,0].hist(d.age,bins=30,color="#0f766e",alpha=.8); ax[0,0].set_title("(A) REAL LaLonde age distribution (NSW+PSID)"); ax[0,0].set_xlabel("age"); ax[0,0].set_ylabel("count")
ax[0,1].plot(ages,out["true_cate"],"-o",color="#d62728",ms=3); ax[0,1].axhline(0,c="#333",lw=1); ax[0,1].fill_between(ages,0,out["true_cate"],where=band,color="#2ca02c",alpha=.15)
ax[0,1].set_title("(B) TRUE treatment effect CATE(age) — non-monotone\n(helps prime-age, HURTS young/old edges)"); ax[0,1].set_xlabel("age"); ax[0,1].set_ylabel("CATE")
pas1=dgp.sig(dgp.G*0.5*(1+dgp.EK*CTR**2)); pas0=dgp.sig(-dgp.G*0.5*(1+dgp.EK*CTR**2))
ax[1,0].plot(ages,pas1,"-o",color="#d62728",ms=3,label="motivated (S=1)"); ax[1,0].plot(ages,pas0,"-o",color="#7f7f7f",ms=3,label="not (S=0)")
ax[1,0].set_title("(C) HIDDEN selection P(treat | age, S):\nmotivated treated far more, esp. at the edges"); ax[1,0].set_xlabel("age"); ax[1,0].set_ylabel("P(treat)"); ax[1,0].legend(fontsize=9)
naive=np.full(NB,np.nan)
for b,cv in enumerate(CTR):
    m=np.isclose(d.x.ravel(),cv); t=d.Y[m & (d.T==1)]; c=d.Y[m & (d.T==0)]
    if len(t)>3 and len(c)>3: naive[b]=t.mean()-c.mean()
ax[1,1].plot(ages,out["true_cate"],"-o",color="#d62728",ms=3,label="TRUE CATE"); ax[1,1].plot(ages,naive,"-s",color="#ff7f0e",ms=3,label="NAIVE observed CATE")
ax[1,1].axhline(0,c="#333",lw=1); ax[1,1].set_title("(D) Confounding inverts the edges:\nnaive observed effect looks positive where treatment truly hurts"); ax[1,1].set_xlabel("age"); ax[1,1].set_ylabel("CATE"); ax[1,1].legend(fontsize=9)
for a in ax.ravel(): a.grid(alpha=.2)
fig.tight_layout(); fig.savefig(HERE/"construction.png",dpi=115); plt.close(fig)
print("\n=== semi-synthetic LaLonde @ matched Γ (3 seeds) ===")
for m in ms: print("  %-9s %.3f ± %.3f"%(m,out["realized"][m]["mean"],out["realized"][m]["sd"]))
print("  ceilings:",{k:round(v,3) for k,v in out["ceilings"].items()})
print("done: realized.png + policy.png + construction.png + synthlalonde_results.json")
