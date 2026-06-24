"""'What the methods see' — training-data diagnostics for the 1-D 2-arm experiment, to JUSTIFY the results.
Reproduces the exact seed-0 training sample and shows, per X grid cell: (A) who is treated + how much data;
(B) the hidden-confounder selection E[S|X,T]; (C) observed vs TRUE vs ESTIMATED(μ̂) outcome means per arm;
(D) the decision trap — naive (confounded) CATE vs true CATE vs μ̂-CATE, with the 'data says treat but it hurts'
region shaded. These four panels are the mechanism behind every result (IPW over-treats, AIPW/R-OW escape).
Usage: python3 exp_1d_data_diagnostics.py {bern|uni}"""
import sys, importlib.util
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
MODE=sys.argv[1] if len(sys.argv)>1 else "bern"
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
ASSETS=(NEW/"assets"/"exp_1d") if MODE=="bern" else (NEW/"assets"/"exp_1d_uniformS")
STITLE="Bernoulli S" if MODE=="bern" else "uniform-11 S"
A,C,B=4.0,5.0,1.0; theta0,theta1=0.5,-1.2; K=2; GRID=np.round(np.linspace(-1,1,21),6)
S_GRID=np.round(np.arange(0,1.0001,0.1),6); n_tr=800; seed=0; gt=5.0
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def pa(X,S,k): return np.clip(sig(theta0+0*np.asarray(X,float)) if k==0 else sig(theta1+A*np.asarray(X,float)+C*np.asarray(S,float)),1e-3,1-1e-3)
def generate(nn,gamma,rng):
    X=rng.choice(GRID,size=nn); S=(rng.uniform(size=nn)<0.5).astype(float) if MODE=="bern" else rng.choice(S_GRID,size=nn)
    Yp=np.column_stack([(rng.uniform(size=nn)<pa(X,S,0)).astype(float),(rng.uniform(size=nn)<pa(X,S,1)).astype(float)])
    U1=-B*X+gamma*(S-0.5); p1=np.exp(U1)/(1+np.exp(U1)); T=(rng.uniform(size=nn)<p1).astype(int)
    return X,S,T,Yp[np.arange(nn),T]
rng=np.random.default_rng(seed); X,S,T,Y=generate(n_tr,gt,rng)
xs=GRID
# true marginal means (over S) on the grid
Ssup=np.array([0.0,1.0]) if MODE=="bern" else S_GRID
mu0_true=np.full_like(xs,sig(theta0)); mu1_true=np.array([np.mean([pa(x,s,1) for s in Ssup]) for x in xs])
# ESTIMATED outcome model μ̂_k(x) the AIPW/DR direct term uses (per-arm logistic on observed data)
mu_hat={}
for k in range(K):
    idx=(T==k); m=LogisticRegression(max_iter=2000).fit(X[idx].reshape(-1,1),Y[idx].astype(int))
    mu_hat[k]=m.predict_proba(xs.reshape(-1,1))[:,1]
# per-X-cell empirical aggregates
def cell(mask_extra=None):
    out={}
    for x in xs:
        m=(np.round(X,6)==x); out[x]=m
    return out
cells={x:(np.round(X,6)==x) for x in xs}
n_tr_arm=np.array([[int(((np.round(X,6)==x)&(T==k)).sum()) for x in xs] for k in range(K)])  # (2,21)
phat=np.array([ (T[cells[x]]==1).mean() if cells[x].any() else np.nan for x in xs])             # empirical P(T=1|X)
ES_t1=np.array([ S[cells[x]&(T==1)].mean() if (cells[x]&(T==1)).any() else np.nan for x in xs])  # E[S|X,T=1]
ES_t0=np.array([ S[cells[x]&(T==0)].mean() if (cells[x]&(T==0)).any() else np.nan for x in xs])
obsY1=np.array([ Y[cells[x]&(T==1)].mean() if (cells[x]&(T==1)).any() else np.nan for x in xs])  # observed treated mean
obsY0=np.array([ Y[cells[x]&(T==0)].mean() if (cells[x]&(T==0)).any() else np.nan for x in xs])
naive_cate=obsY1-obsY0; true_cate=mu1_true-mu0_true; hat_cate=mu_hat[1]-mu_hat[0]
# ---------- figure ----------
fig,ax=plt.subplots(2,2,figsize=(13.2,8.4),dpi=140)
# (A) assignment + data counts
axA=ax[0,0]; wdt=0.045
axA.bar(xs-wdt,n_tr_arm[1],width=2*wdt,color="#d62728",alpha=.55,label="# treated")
axA.bar(xs+wdt,n_tr_arm[0],width=2*wdt,color="#9aa0a6",alpha=.55,label="# control")
axA.set_ylabel("# training units per X"); axA.set_xlabel("X")
axt=axA.twinx(); axt.plot(xs,phat,'-o',color="#111",lw=2,ms=4,label="empirical P(T=1 | X)")
axt.axhline(0.5,ls=':',color="#888",lw=1); axt.set_ylabel("P(T=1 | X)"); axt.set_ylim(0,1)
axA.set_title("(A) Who is treated, and how much data per cell",fontsize=11)
h1,l1=axA.get_legend_handles_labels(); h2,l2=axt.get_legend_handles_labels(); axA.legend(h1+h2,l1+l2,fontsize=8,loc="upper center")
# (B) hidden confounder selection
axB=ax[0,1]
axB.plot(xs,ES_t1,'-o',color="#d62728",lw=2,ms=4,label="E[S | X, T=1]  (treated)")
axB.plot(xs,ES_t0,'-s',color="#9467bd",lw=2,ms=4,label="E[S | X, T=0]  (control)")
axB.axhline(0.5,ls='--',color="#2ca02c",lw=1.4,label="marginal E[S]=0.5 (no confounding)")
axB.fill_between(xs,ES_t0,ES_t1,where=ES_t1>=ES_t0,color="#d62728",alpha=.10)
axB.set_xlabel("X"); axB.set_ylabel("E[unobserved S | X, arm]"); axB.set_ylim(0,1)
axB.set_title("(B) The confounding: treated units are selected toward HIGH S",fontsize=11); axB.grid(alpha=.25); axB.legend(fontsize=8,loc="best")
# (C) observed vs true vs estimated outcome means
axC=ax[1,0]
axC.plot(xs,obsY1,'o',color="#d62728",ms=5,alpha=.85,label="observed treated mean (confounded)")
axC.plot(xs,mu1_true,'-',color="#d62728",lw=2.4,label="TRUE treat marginal μ₁(X)")
axC.plot(xs,mu_hat[1],'--',color="#ff7f0e",lw=2.2,label="ESTIMATED μ̂₁(X) (AIPW uses this)")
axC.plot(xs,obsY0,'s',color="#9aa0a6",ms=4,alpha=.8,label="observed control mean")
axC.plot(xs,mu0_true,'-',color="#555",lw=1.8,label="TRUE control μ₀(X)≈0.62")
axC.set_xlabel("X"); axC.set_ylabel("outcome  E[Y]"); axC.set_ylim(0,1.02)
axC.set_title("(C) Observed treated outcome is INFLATED vs truth (confounding); μ̂ is closer",fontsize=11); axC.grid(alpha=.25); axC.legend(fontsize=7.5,loc="upper left")
# (D) the decision trap: naive vs true vs μ̂ CATE
axD=ax[1,1]
axD.plot(xs,naive_cate,'-o',color="#111",lw=2,ms=4,label="NAIVE CATE = obs treat − obs control")
axD.plot(xs,true_cate,'-',color="#2ca02c",lw=2.6,label="TRUE CATE = μ₁−μ₀")
axD.plot(xs,hat_cate,'--',color="#ff7f0e",lw=2.2,label="μ̂ CATE (estimated)")
axD.axhline(0,color="#888",lw=1)
trap=(naive_cate>0)&(true_cate<0)
axD.fill_between(xs,0,naive_cate,where=trap,color="#d62728",alpha=.18,label="TRAP: data says treat, truth says it hurts")
axD.set_xlabel("X"); axD.set_ylabel("CATE = effect of treat on E[Y]")
axD.set_title("(D) The trap: naive CATE says treat low-X (wrong); truth & μ̂ say treat high-X",fontsize=11); axD.grid(alpha=.25); axD.legend(fontsize=7.5,loc="upper left")
fig.suptitle(f"What the methods see — 1-D 2-arm training data ({STITLE}, n={n_tr}, seed 0)",y=0.995,fontsize=13)
fig.tight_layout(rect=[0,0,1,0.98]); fig.savefig(ASSETS/"data_diagnostics.png",bbox_inches="tight"); plt.close(fig)
np.savez(ASSETS/"data_diagnostics.npz",xs=xs,n_tr_arm=n_tr_arm,phat=phat,ES_t1=ES_t1,ES_t0=ES_t0,
         obsY1=obsY1,obsY0=obsY0,mu1_true=mu1_true,mu0_true=mu0_true,mu_hat1=mu_hat[1],mu_hat0=mu_hat[0],
         naive_cate=naive_cate,true_cate=true_cate,hat_cate=hat_cate)
print(f"[{MODE}] DONE -> {ASSETS/'data_diagnostics.png'}")
print(f"[{MODE}] confounding gap E[S|T=1]-E[S|T=0] (mean over X) = {np.nanmean(ES_t1-ES_t0):+.3f}  (bern≈large, uni≈small)")
print(f"[{MODE}] trap cells (naive>0 but true<0): {int(np.nansum(trap))}/21   true-CATE crossover near X=0")
print(f"[{MODE}] mean|naive-true CATE|={np.nanmean(np.abs(naive_cate-true_cate)):.3f}  mean|μ̂-true CATE|={np.nanmean(np.abs(hat_cate-true_cate)):.3f}")
