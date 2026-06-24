"""Build the UNCAPPED non-monotone experiment (same v1 DGP as exp_nonmono, cap=(1,1) — fair to Kallus). Runs all methods,
and makes the figures that justify 'R-OW basically treats everyone uncapped':
  propensity.png   — P(T=1|X,S): the selection / near-positivity-violation that fools everyone
  whats_seen.png   — observed treated vs observed control vs the TRUTH (the data 'says treat' everywhere)
  realized_uncap.png — realised value per method at matched Γ (R-OW ≈ never-treat)
  policy_uncap.png — who each method treats (value methods ≈ treat-all)
Saves uncap_results.json."""
import sys, json, importlib.util
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
NM=NEW/"assets"/"exp_nonmono"; HERE=NEW/"assets"/"exp_uncap"
sp=importlib.util.spec_from_file_location("dgp",str(NM/"dgp.py")); dgp=importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
def load(n,rel):
    s=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
row=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
ro =load("ro","methods/IPW-O-X/Capped/ipw_o_x_capped.py").solve_ipw_o_x_capped
ipw=load("ipw","methods/IPW-X-X/Capped/ipw_x_x_capped.py").solve_ipw_x_x_capped
rowdr=load("rowdr","methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py").solve_doublyrobust_o_w_capped
rodr =load("rodr","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py").solve_doublyrobust_o_x_capped
regow=load("regow","methods/Hajek-O-W/Capped/hajek_o_w_capped.py").solve_hajek_o_w_capped
kal=load("kal","methods/Kallus/kallus.py")
K=dgp.K; NOCAP=(1.0,1.0); mG=dgp.GAMMAS[dgp.mi]; X=np.sort(dgp.GRID); xband=float(np.sqrt(dgp.BT))
COL={"R-OW":"#d62728","R-O":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","R-OW-DR":"#a01f1f","Kallus":"#7f7f7f","Hajek-OW":"#17becf"}

# ---------------- (A) propensity P(T=1|X,S) ----------------
pt_s1=dgp.sig(dgp.G*0.5*(1+dgp.EK*X**2)); pt_s0=dgp.sig(-dgp.G*0.5*(1+dgp.EK*X**2))
fig,ax=plt.subplots(figsize=(8.8,4.6))
ax.plot(X,pt_s1,"-o",color="#d62728",ms=4,lw=2,label="P(treat | X, S=1)  — motivated")
ax.plot(X,pt_s0,"-o",color="#7f7f7f",ms=4,lw=2,label="P(treat | X, S=0)  — not")
ax.axhline(0.5,ls=":",c="#999",lw=1); ax.fill_between(X,0,1,where=np.abs(X)>0.85,color="#d62728",alpha=.06)
ax.set_xlabel("X"); ax.set_ylabel("P(treat | X, S)"); ax.set_ylim(-.03,1.03)
ax.set_title("Treatment-assignment probability — selection is concentrated at the edges\nat |X|→1: S=1 is ALWAYS treated, S=0 is NEVER treated (near positivity violation)",fontsize=10.5)
ax.annotate("treated here = all S=1\ncontrol here = all S=0",xy=(0.95,0.5),xytext=(0.45,0.5),fontsize=9.5,color="#333",
            arrowprops=dict(arrowstyle="->",color="#555"),ha="center")
ax.legend(fontsize=9.5,loc="center left"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(HERE/"propensity.png",dpi=120); plt.close(fig); print("propensity.png")

# ---------------- (B) what the methods SEE (observed) vs the TRUTH ----------------
rng=np.random.default_rng(0); d=dgp.generate(40000,rng)
obs_t=np.full(len(X),np.nan); obs_c=np.full(len(X),np.nan)
for i,xv in enumerate(X):
    m=np.isclose(d.X.ravel(),xv); t=d.Y[m&(d.T==1)]; c=d.Y[m&(d.T==0)]
    if len(t): obs_t[i]=t.mean()
    if len(c): obs_c[i]=c.mean()
true_t=0.5*(dgp.pa(X,0,1)+dgp.pa(X,1,1)); true_c=0.5*(dgp.pa(X,0,0)+dgp.pa(X,1,0))
band=true_t>true_c
fig,ax=plt.subplots(figsize=(9.4,5.2))
ax.fill_between(X,0,1,where=band,color="#2ca02c",alpha=.06,label="TRUE optimal: treat (|X|<0.6)")
ax.plot(X,obs_t,"-o",color="#d62728",lw=2.4,ms=4,label="OBSERVED treated mean  (what the methods see)")
ax.plot(X,obs_c,"-o",color="#7f7f7f",lw=2.4,ms=4,label="OBSERVED control mean  (what the methods see)")
ax.plot(X,true_t,"--",color="#7a1212",lw=1.8,label="TRUE E[Y(1)|X]  (marginal over S)")
ax.plot(X,true_c,"--",color="#444",lw=1.8,label="TRUE E[Y(0)|X] = 0.50  (marginal over S)")
ax.set_xlabel("X"); ax.set_ylabel("mean outcome"); ax.set_ylim(-.02,1.02)
ax.set_title("Why every method 'wants' to treat: the OBSERVED data says treat ≳ control EVERYWHERE\n— but the TRUTH is that treatment HURTS at the edges (control 0.50 ≫ treat there)",fontsize=10.3)
ax.annotate("observed control is an\nunrepresentative low-S subset\n(true control = 0.50)",xy=(0.93,obs_c[-1]),xytext=(0.30,0.78),fontsize=9,color="#333",
            arrowprops=dict(arrowstyle="->",color="#555"),ha="center")
ax.legend(fontsize=8.6,loc="upper center",ncol=2); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(HERE/"whats_seen.png",dpi=120); plt.close(fig); print("whats_seen.png")

# ---------------- (C,D) run methods UNCAPPED at matched Γ ----------------
METHODS=["R-OW","R-OW-DR","R-O","Hajek-OW","IPW","AIPW","Kallus"]; SEEDS=list(range(8))
rea={m:[] for m in METHODS}; fr={m:[] for m in METHODS}; pol={m:[] for m in METHODS}; cei={"ctrl":[],"bm":[],"orc":[]}
for s in SEEDS:
    rng=np.random.default_rng(s); tr=dgp.generate(dgp.n_tr,rng); te=dgp.generate(dgp.n_te,rng)
    w,_=common.ipw_weights_from_data(tr.X,tr.T,K); Dm=common.pairwise_distance_matrix(tr.X)
    eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(tr.X,tr.T,tr.Y,n_arms=K,cross_fit=True)
    byv={}
    for j,xv in enumerate(np.round(tr.X.ravel(),6)): byv.setdefault(round(float(xv),6),j)
    uk=np.array(sorted(byv)); uidx=np.array([byv[k] for k in uk]); nn=np.array([int(np.argmin(np.abs(uk-x))) for x in np.round(te.X.ravel(),6)])
    def rt(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn]; return float((c*te.Ypot.T).sum()/c.shape[1]), float(pi[1].mean())
    def grid(pi):
        pi=np.asarray(pi,float); g=np.full(len(X),np.nan)
        for b,cv in enumerate(X):
            k=round(float(cv),6)
            if k in byv: g[b]=pi[1,byv[k]]
        return g
    P={"R-OW":row(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=NOCAP,discretize=False,zscore=False,epsilon=eps).pi,
       "R-OW-DR":rowdr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=mG,cap=NOCAP,discretize=False,zscore=False,epsilon=eps).pi,
       "R-O":ro(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=NOCAP,discretize=False).pi,
       "Hajek-OW":regow(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=NOCAP,discretize=False,zscore=False,epsilon=eps).pi,
       "IPW":ipw(tr.X,tr.T,tr.Y,w,n_arms=K,cap=NOCAP,discretize=False).pi,
       "AIPW":rodr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=NOCAP,discretize=False).pi}
    for m in ["R-OW","R-OW-DR","R-O","Hajek-OW","IPW","AIPW"]:
        v,f=rt(P[m]); rea[m].append(v); fr[m].append(f); pol[m].append(grid(P[m]))
    th=kal.fit_kallus(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
    kp=kal.predict_kallus(th.theta,te.X,("affine",)); rea["Kallus"].append(float((kp*te.Ypot).sum(1).mean())); fr["Kallus"].append(float(kp[:,1].mean()))
    pol["Kallus"].append(kal.predict_kallus(th.theta,X.reshape(-1,1),("affine",))[:,1])
    cei["ctrl"].append(float(te.Ypot[:,0].mean())); cei["orc"].append(float(te.Ypot.max(1).mean())); cei["bm"].append(float(te.Ypot[np.arange(dgp.n_te),np.argmax(te.mu,1)].mean()))
    print("seed%d done"%s,flush=True)
out={"gamma":mG,"x":X.tolist(),"true_cate":(true_t-true_c).tolist(),
     "ceilings":{k:float(np.mean(v)) for k,v in cei.items()},
     "methods":{m:{"realised":float(np.mean(rea[m])),"sd":float(np.std(rea[m])),"treat_frac":float(np.mean(fr[m])),"policy":np.nanmean(np.array(pol[m]),0).tolist()} for m in METHODS}}
(HERE/"uncap_results.json").write_text(json.dumps(out,indent=1))

# realised bar chart
ms=sorted(METHODS,key=lambda m:-out["methods"][m]["realised"])
fig,ax=plt.subplots(figsize=(8.4,4.7))
y=[out["methods"][m]["realised"] for m in ms]; e=[out["methods"][m]["sd"] for m in ms]
ax.bar(range(len(ms)),y,yerr=e,capsize=4,color=[COL[m] for m in ms],alpha=.88,edgecolor="#333")
ax.axhline(out["ceilings"]["bm"],ls="--",c="#2ca02c",lw=1.5,label="best-deployable (with targeting) = %.3f"%out["ceilings"]["bm"])
ax.axhline(out["ceilings"]["ctrl"],ls=":",c="#444",lw=1.4,label="never-treat = %.3f"%out["ceilings"]["ctrl"])
ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms,rotation=12); ax.set_ylim(0.45,out["ceilings"]["bm"]+.02)
ax.set_ylabel("realised success rate (test, mean ± SD, 8 seeds)")
ax.set_title("UNCAPPED, matched Γ≈12.18 — no method recovers the band; R-OW ≈ never-treat\n(value methods treat ~everyone → harmful edges cancel the good middle)",fontsize=10.3)
ax.legend(fontsize=9,loc="upper right"); ax.grid(alpha=.25,axis="y"); fig.tight_layout(); fig.savefig(HERE/"realized_uncap.png",dpi=120); plt.close(fig); print("realized_uncap.png")

# policy chart
fig,ax=plt.subplots(figsize=(9.0,4.7))
ax.fill_between(X,0,1,where=band,color="#2ca02c",alpha=.08,label="true treat-band (CATE>0)")
for m in ["R-OW","R-O","AIPW","Hajek-OW","Kallus"]:
    ax.plot(X,out["methods"][m]["policy"],"-o",color=COL[m],lw=1.7,ms=3.3,label="%s (treats %.0f%%)"%(m,100*out["methods"][m]["treat_frac"]))
ax.set_xlabel("X"); ax.set_ylabel("P(treat | X) at matched Γ (avg over 8 seeds)"); ax.set_ylim(-.05,1.05)
ax.set_title("Who gets treated UNCAPPED — the value methods (R-OW, R-O) treat ≈ everyone,\nincluding the harmful edges; only the regret method / AIPW pull back",fontsize=10.3)
ax.legend(fontsize=8.6,ncol=2,loc="lower center"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(HERE/"policy_uncap.png",dpi=120); plt.close(fig); print("policy_uncap.png")
print("\n=== UNCAPPED realised @ matched Γ (8 seeds) ===")
for m in ms: print("  %-10s %.3f ± %.3f   (treats %.0f%%)"%(m,out["methods"][m]["realised"],out["methods"][m]["sd"],100*out["methods"][m]["treat_frac"]))
print("  ceilings:",{k:round(v,3) for k,v in out["ceilings"].items()})
