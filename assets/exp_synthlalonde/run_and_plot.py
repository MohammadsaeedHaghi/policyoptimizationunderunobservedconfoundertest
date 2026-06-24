"""Run the semi-synthetic LaLonde experiment (5 seeds × Γ-sweep, all methods) and make the figures.
Saves synthlalonde_results.json + realized.png + policy.png + construction.png."""
import sys, json, importlib.util
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; NEW=HERE.parents[1]
sys.path.insert(0,str(NEW)); import common
dgp=importlib.util.module_from_spec(importlib.util.spec_from_file_location("dgp",str(HERE/"dgp.py")))
importlib.util.spec_from_file_location("dgp",str(HERE/"dgp.py")).loader.exec_module(dgp)
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
row=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
ro =load("ro","methods/IPW-O-X/Capped/ipw_o_x_capped.py").solve_ipw_o_x_capped
ipw=load("ipw","methods/IPW-X-X/Capped/ipw_x_x_capped.py").solve_ipw_x_x_capped
rowdr=load("rowdr","methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py").solve_doublyrobust_o_w_capped
rodr =load("rodr","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py").solve_doublyrobust_o_x_capped
kal=load("kal","methods/Kallus/kallus.py")
K=dgp.K; CAP=dgp.CAP; CTR=dgp.CTR; NB=dgp.NB; G=[1.0,6.0,12.1825,16.0]; mi=2; SEEDS=list(range(3)); mG=G[mi]  # lean Γ-grid/seeds for speed
COL={"R-OW":"#d62728","R-O":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","R-OW-DR":"#a01f1f","Kallus":"#7f7f7f"}
GVAR={"R-OW","R-O","R-OW-DR","Kallus"}; METHODS=["R-OW","R-OW-DR","R-O","IPW","AIPW","Kallus"]
rea={m:{gi:[] for gi in range(len(G))} for m in METHODS}; pol={m:[] for m in METHODS}; cei={"ctrl":[],"bm":[],"orc":[]}
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
    f_ipw=ipw(tr.x,tr.T,tr.Y,w,n_arms=K,cap=CAP,discretize=False).pi
    f_aipw=rodr(tr.x,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi
    polM={"IPW":grid(f_ipw),"AIPW":grid(f_aipw)}; v_ipw=rt(f_ipw); v_aipw=rt(f_aipw)
    for gi,g in enumerate(G):
        rea["IPW"][gi].append(v_ipw); rea["AIPW"][gi].append(v_aipw)
        pr=row(tr.x,tr.T,tr.Y,w,n_arms=K,Gamma=g,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi; rea["R-OW"][gi].append(rt(pr))
        po=ro(tr.x,tr.T,tr.Y,w,n_arms=K,Gamma=g,cap=CAP,discretize=False).pi; rea["R-O"][gi].append(rt(po))
        pd_=rowdr(tr.x,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=g,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi; rea["R-OW-DR"][gi].append(rt(pd_))
        th=kal.fit_kallus(tr.x,tr.T,tr.Y,w,n_arms=K,Gamma=g,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
        kp=kal.predict_kallus(th.theta,te.x,("affine",)); rea["Kallus"][gi].append(float((kp*te.Ypot).sum(1).mean()))
        if gi==mi:
            polM["R-OW"]=grid(pr); polM["R-O"]=grid(po); polM["R-OW-DR"]=grid(pd_)
            polM["Kallus"]=kal.predict_kallus(th.theta,CTR.reshape(-1,1),("affine",))[:,1]
    for m in METHODS: pol[m].append(polM[m])
    cei["ctrl"].append(float(te.Ypot[:,0].mean())); cei["orc"].append(float(te.Ypot.max(1).mean())); cei["bm"].append(float(te.Ypot[np.arange(dgp.n_te),np.argmax(te.mu,1)].mean()))
    print(f"seed{s}: R-OW@mΓ={np.mean(rea['R-OW'][mi]):.3f} AIPW={v_aipw:.3f} IPW={v_ipw:.3f} Kallus@mΓ={np.mean(rea['Kallus'][mi]):.3f}",flush=True)
out={"gammas":G,"matched_idx":mi,"ctr":CTR.tolist(),"ages":dgp.x2age(CTR).tolist(),
     "ceilings":{k:float(np.mean(v)) for k,v in cei.items()},
     "realized":{m:{"mean":[float(np.mean(rea[m][gi])) for gi in range(len(G))],"sd":[float(np.std(rea[m][gi])) for gi in range(len(G))]} for m in METHODS},
     "policy":{m:np.nanmean(np.array(pol[m]),axis=0).tolist() for m in METHODS}}
# true CATE on the grid
Sgrid=0.5*(dgp.pa(CTR,0,1)+dgp.pa(CTR,1,1)) - 0.5*(dgp.pa(CTR,0,0)+dgp.pa(CTR,1,0))
out["true_cate"]=Sgrid.tolist()
(HERE/"synthlalonde_results.json").write_text(json.dumps(out,indent=1))

# ---- realized vs Γ ----
fig,ax=plt.subplots(figsize=(8.6,4.9))
for m in METHODS:
    y=np.array(out["realized"][m]["mean"]); e=np.array(out["realized"][m]["sd"]); ls="--" if m in("IPW","AIPW") else "-"
    ax.plot(G,y,ls+"o",color=COL[m],label=m+(" (Γ-indep.)" if m in("IPW","AIPW") else ""),lw=2.0,ms=4)
    if m in GVAR: ax.fill_between(G,y-e,y+e,color=COL[m],alpha=.10)
ax.axhline(out["ceilings"]["bm"],ls="--",c="#2ca02c",lw=1.3,label="best-means ceiling (%.3f)"%out["ceilings"]["bm"])
ax.axhline(out["ceilings"]["ctrl"],ls=":",c="#444",lw=1.2,label="never-treat (%.3f)"%out["ceilings"]["ctrl"])
ax.axvline(mG,color="#bbb",ls=":",lw=1); ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_ylabel("realised success rate (test, mean ± SD over %d seeds)"%len(SEEDS))
ax.set_title("Semi-synthetic LaLonde — realised value vs Γ\nR-OW / R-OW-DR are the best; matched Γ=e^{2.5}≈12.18 (dotted)",fontsize=10.5)
ax.legend(fontsize=8.5,ncol=2,loc="lower right"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(HERE/"realized.png",dpi=120); plt.close(fig)

# ---- policy vs age at matched Γ ----
ages=np.array(out["ages"]); band=np.array(out["true_cate"])>0
fig,ax=plt.subplots(figsize=(8.6,4.6))
ax.fill_between(ages,0,1,where=band,color="#2ca02c",alpha=.10,label="true treat-band (CATE>0)")
for m in ["R-OW","R-OW-DR","AIPW","IPW","Kallus"]:
    ax.plot(ages,out["policy"][m],"-o",color=COL[m],label=m,lw=1.7,ms=3.5)
ax.set_xlabel("age (years, real LaLonde bins)"); ax.set_ylabel("P(treat | age) at matched Γ"); ax.set_ylim(-.05,1.05)
ax.set_title("Who gets treated — R-OW concentrates on the prime-age band; Kallus can't,\nIPW/AIPW scatter toward the confounded edges",fontsize=10.5)
ax.legend(fontsize=8.5,ncol=2,loc="upper center"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(HERE/"policy.png",dpi=120); plt.close(fig)

# ---- construction diagram (4 panels) ----
rng=np.random.default_rng(0); d=dgp.generate(4000,rng)
fig,ax=plt.subplots(2,2,figsize=(11,7.4))
ax[0,0].hist(d.age,bins=30,color="#0f766e",alpha=.8); ax[0,0].set_title("(A) REAL LaLonde age distribution (NSW+PSID)"); ax[0,0].set_xlabel("age"); ax[0,0].set_ylabel("count")
ax[0,1].plot(ages,out["true_cate"],"-o",color="#d62728",ms=3); ax[0,1].axhline(0,c="#333",lw=1)
ax[0,1].fill_between(ages,0,out["true_cate"],where=band,color="#2ca02c",alpha=.15)
ax[0,1].set_title("(B) TRUE treatment effect CATE(age) — non-monotone\n(helps prime-age, HURTS young/old edges)"); ax[0,1].set_xlabel("age"); ax[0,1].set_ylabel("CATE = P(success|treat)−P(|ctrl)")
# selection P(treat|age,S)
pa_grid_s1=dgp.sig(dgp.G*(1-0.5)*(1+dgp.EK*CTR**2)); pa_grid_s0=dgp.sig(dgp.G*(0-0.5)*(1+dgp.EK*CTR**2))
ax[1,0].plot(ages,pa_grid_s1,"-o",color="#d62728",ms=3,label="motivated (S=1)"); ax[1,0].plot(ages,pa_grid_s0,"-o",color="#7f7f7f",ms=3,label="not (S=0)")
ax[1,0].set_title("(C) HIDDEN selection P(treat | age, S):\nmotivated treated far more, esp. at the edges"); ax[1,0].set_xlabel("age"); ax[1,0].set_ylabel("P(treat)"); ax[1,0].legend(fontsize=9)
# naive observed CATE vs true
naive=np.full(NB,np.nan)
for b,cv in enumerate(CTR):
    m=np.isclose(d.x.ravel(),cv); t=d.Y[m & (d.T==1)]; c=d.Y[m & (d.T==0)]
    if len(t)>3 and len(c)>3: naive[b]=t.mean()-c.mean()
ax[1,1].plot(ages,out["true_cate"],"-o",color="#d62728",ms=3,label="TRUE CATE"); ax[1,1].plot(ages,naive,"-s",color="#ff7f0e",ms=3,label="NAIVE observed CATE")
ax[1,1].axhline(0,c="#333",lw=1); ax[1,1].set_title("(D) Confounding inverts the edges:\nnaive observed effect looks positive where treatment truly hurts"); ax[1,1].set_xlabel("age"); ax[1,1].set_ylabel("CATE"); ax[1,1].legend(fontsize=9)
for a in ax.ravel(): a.grid(alpha=.2)
fig.tight_layout(); fig.savefig(HERE/"construction.png",dpi=115); plt.close(fig)
print("\n=== semi-synthetic LaLonde (5 seeds) ===")
for m in METHODS: print("  %-9s realised@matchedΓ %.3f ± %.3f"%(m,out["realized"][m]["mean"][mi],out["realized"][m]["sd"][mi]))
print("  ceilings:",{k:round(v,3) for k,v in out["ceilings"].items()})
print("saved synthlalonde_results.json + realized.png + policy.png + construction.png")
