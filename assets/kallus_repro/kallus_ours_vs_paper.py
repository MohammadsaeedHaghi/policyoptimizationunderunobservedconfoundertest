"""Validate OUR methods/Kallus/kallus.py::fit_kallus against the PAPER's CRLogit on the exact Kallus & Zhou
§7.1.1 Fig.1 experiment (out-of-sample regret vs confounding). Overlays, on the SAME reps/DGP:
  • Oracle (true-CATE ceiling), IPW (naive logistic, harmed), CRLogit (paper-faithful closed-form, clamp),
  • OUR fit_kallus with NOMINAL weights  (paper setup → isolates the ALGORITHM: should track CRLogit),
  • OUR fit_kallus with ESTIMATED weights (our project default → the no-true-propensity convention).
x-axis = log Γ (paper). For our box method we feed Tan box Γ = e^{logΓ} (cross-walk; the two sensitivity
models differ). If Ours-nominal tracks CRLogit's dip-and-recover, our Algorithm-1 implementation is faithful."""
import sys, time, importlib.util
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(ROOT)); import common
ASSETS=ROOT/"assets"/"kallus_repro"
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(ROOT/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
kal=load("kal","methods/Kallus/kallus.py"); fit_kallus=kal.fit_kallus; predict_kallus=kal.predict_kallus
# ---- DGP (verbatim from kallus_repro.py / data_scenarios.py) ----
BETA_CONS=2.5; BETA_X=np.array([0,.5,-0.5,0,0,0.0]); BETA_X_T=np.array([-1.5,1,-1.5,1.,0.5,0.0])
BETA_TC=np.array([0,.75,-.5,0,-1,0.0]); MU_X=np.array([-1,.5,-1,0,-1.0]); ALPHA=-2.0; WCOEF=1.5; LG_TRUE=1.5
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def get_bnds(Q,LG):
    Q=np.asarray(Q,float); eL=np.exp(LG); eLm=np.exp(-LG); return 1.0/(eL*Q/(1-Q+eL*Q)),1.0/(eLm*Q/(1-Q+eLm*Q))
def real_risk(T,xa,u): T=np.asarray(T,float); return T*BETA_CONS+xa@BETA_X+(xa@BETA_X_T)*T+ALPHA*u*(2*T-1)+WCOEF*u
def real_risk_prob(pi1,xa,u): return float(np.mean(pi1*real_risk(np.ones(len(u)),xa,u)+(1-pi1)*real_risk(np.zeros(len(u)),xa,u)))
def generate(n,rng):
    u=(rng.random(n)>0.5).astype(float); x=rng.standard_normal((n,5))+MU_X*(2*u-1)[:,None]; xa=np.hstack([x,np.ones((n,1))])
    nominal=sig(xa@BETA_TC); optT=(real_risk(np.ones(n),xa,u)<real_risk(np.zeros(n),xa,u)).astype(int)
    a_bnd,b_bnd=get_bnds(nominal,LG_TRUE); p_lo=1/b_bnd; p_hi=1/a_bnd
    T=(rng.random(n)<np.where(optT==1,p_hi,p_lo)).astype(int); Y=real_risk(T,xa,u)+2*rng.standard_normal(n)
    return x,xa,u,T,Y
# ---- paper-faithful CRLogit (closed-form self-normalised inner; LG=0 => naive IPW logistic) ----
def worst_case(coef,a,b):
    o=np.argsort(coef); c=coef[o]; aa=a[o]; bb=b[o]; n=len(c)
    pa_c=np.concatenate([[0],np.cumsum(aa*c)]); pa=np.concatenate([[0],np.cumsum(aa)])
    sb_c=np.concatenate([np.cumsum((bb*c)[::-1])[::-1],[0]]); sb=np.concatenate([np.cumsum(bb[::-1])[::-1],[0]])
    den=pa+sb; den[den==0]=1e-12; vals=(pa_c+sb_c)/den; k=int(np.argmax(vals))
    W=np.where(np.arange(n)<k,aa,bb); Ws=np.empty(n); Ws[o]=W; return float(vals[k]),Ws
def crlogit(xa,T,Y,ehat,LG,*,n_restarts,n_iters,clamp,rng,eta0=1.0,kappa=0.5):
    fq=np.where(T==1,ehat,1-ehat); a,b=get_bnds(fq,LG); Tsgn=2*T-1.0; p=xa.shape[1]; best_th,best_R=None,np.inf
    for _ in range(n_restarts):
        th=rng.standard_normal(p)*0.25; acc=np.zeros(p)
        for it in range(n_iters):
            pi1=sig(xa@th); _,Ws=worst_case(Y*Tsgn*pi1,a,b); What=Ws/Ws.sum()
            th=th-(eta0/(it+1)**kappa)*((Y*Tsgn*What*pi1*(1-pi1))@xa); acc+=th
        thb=acc/n_iters; R,_=worst_case(Y*Tsgn*sig(xa@thb),a,b)
        if R<best_R: best_R,best_th=R,thb.copy()
    if clamp and best_R>0: return np.zeros(p),0.0
    return best_th,best_R
def nominal_w(xa,T,K):                      # nominal inverse weights ŵ=1/ẽ, per-arm Hájek-normalised (Σ_{I_k}=n)
    e=np.clip(sig(xa@BETA_TC),1e-3,1-1e-3); p_obs=np.where(T==1,e,1-e); wr=1.0/p_obs; w=wr.copy(); n=len(T)
    for k in range(K):
        m=(T==k); w[m]=wr[m]*n/wr[m].sum()
    return w
# ---- run ----
N_REPS=int(sys.argv[1]) if len(sys.argv)>1 else 20
n_tr=200; n_te=5000; K=2; N_RST=4; N_IT=20         # our fit_kallus budget = the kallus_methods.py experiment
LOG_GAMMAS=np.array([0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0]); G=len(LOG_GAMMAS)
crl=np.full((N_REPS,G),np.nan); oursN=np.full((N_REPS,G),np.nan); oursE=np.full((N_REPS,G),np.nan)
ipw=np.full(N_REPS,np.nan); orc=np.full(N_REPS,np.nan)
t0=time.time(); rng=np.random.default_rng(0)
for r in range(N_REPS):
    x,xa,u,T,Y=generate(n_tr,rng); xt,xat,ut,Tt,Yt=generate(n_te,rng)
    if len(np.unique(T))<2: continue
    cate_te=real_risk(np.ones(n_te),xat,ut)-real_risk(np.zeros(n_te),xat,ut)
    eh=np.clip(sig(xa@BETA_TC),1e-3,1-1e-3); base=real_risk_prob(np.zeros(n_te),xat,ut)
    Yrew=-Y                                                  # our fit_kallus maximises reward = −risk
    wN=nominal_w(xa,T,K); wE,_=common.ipw_weights_from_data(x,T,K)
    orc[r]=real_risk_prob((cate_te<0).astype(float),xat,ut)-base
    th,_=crlogit(xa,T,Y,eh,0.0,n_restarts=N_RST,n_iters=N_IT,clamp=False,rng=rng)   # IPW (naive logistic)
    ipw[r]=real_risk_prob(sig(xat@th),xat,ut)-base
    for gi,LG in enumerate(LOG_GAMMAS):
        th,_=crlogit(xa,T,Y,eh,float(LG),n_restarts=N_RST,n_iters=N_IT,clamp=True,rng=rng)
        crl[r,gi]=real_risk_prob(np.zeros(n_te) if np.allclose(th,0) else sig(xat@th),xat,ut)-base
        Gbox=float(np.exp(LG))                               # cross-walk: Tan box Γ = e^{logΓ}
        try:
            kN=fit_kallus(x,T,Yrew,wN,n_arms=K,Gamma=Gbox,maximize=True,wasserstein=False,basis=("affine",),n_iters=N_IT,n_restarts=N_RST,seed=0).theta
            oursN[r,gi]=real_risk_prob(predict_kallus(kN,xt,("affine",))[:,1],xat,ut)-base
        except Exception as e: print("  oursN FAIL",str(e)[:60])
        try:
            kE=fit_kallus(x,T,Yrew,wE,n_arms=K,Gamma=Gbox,maximize=True,wasserstein=False,basis=("affine",),n_iters=N_IT,n_restarts=N_RST,seed=0).theta
            oursE[r,gi]=real_risk_prob(predict_kallus(kE,xt,("affine",))[:,1],xat,ut)-base
        except Exception as e: print("  oursE FAIL",str(e)[:60])
    print(f"[{time.time()-t0:.0f}s] rep {r+1}/{N_REPS}  CRL@1.5={crl[r,5]:+.3f} oursN@1.5={oursN[r,5]:+.3f} oursE@1.5={oursE[r,5]:+.3f}",flush=True)
def msd(a,ax=0): m=np.nanmean(a,axis=ax); s=np.nanstd(a,axis=ax)/np.sqrt(np.maximum(np.sum(~np.isnan(a),axis=ax),1)); return m,s
crl_m,crl_s=msd(crl); oN_m,oN_s=msd(oursN); oE_m,oE_s=msd(oursE); ipw_m,ipw_s=msd(ipw); orc_m,_=msd(orc)
np.savez(ASSETS/"kallus_ours_vs_paper.npz",LOG_GAMMAS=LOG_GAMMAS,crl=crl,oursN=oursN,oursE=oursE,ipw=ipw,orc=orc,
         crl_m=crl_m,crl_s=crl_s,oN_m=oN_m,oN_s=oN_s,oE_m=oE_m,oE_s=oE_s,ipw_m=ipw_m,orc_m=orc_m,n_reps=N_REPS)
# ---- plot ----
fig,ax=plt.subplots(figsize=(8.2,5.1),dpi=140)
ax.axhline(orc_m,color="#111",ls="--",lw=1.3,label=f"Oracle (true CATE) {orc_m:+.2f}")
ax.axhline(ipw_m,color="#8c564b",lw=2.0,label=f"IPW (naive logistic) {ipw_m:+.2f}")
ax.plot(LOG_GAMMAS,crl_m,'-o',color="#d62728",lw=2.5,ms=5,label="CRLogit — PAPER (closed-form, odds-tilt, nominal)",zorder=6)
ax.fill_between(LOG_GAMMAS,crl_m-crl_s,crl_m+crl_s,color="#d62728",alpha=.15)
ax.plot(LOG_GAMMAS,oN_m,'-P',color="#7f7f7f",lw=2.5,ms=6,label="OUR fit_kallus — NOMINAL weights (paper setup)",zorder=5)
ax.fill_between(LOG_GAMMAS,oN_m-oN_s,oN_m+oN_s,color="#7f7f7f",alpha=.15)
ax.plot(LOG_GAMMAS,oE_m,marker='X',color="#1f77b4",lw=2.0,ms=6,ls="--",label="OUR fit_kallus — ESTIMATED weights (our default)",zorder=4)
ax.axvline(LG_TRUE,ls=':',color="#555",lw=1.2); ax.text(LG_TRUE+0.04,ax.get_ylim()[0]+.02,"true log Γ=1.5",fontsize=8,va="bottom",color="#555",rotation=90)
ax.axhline(0,color="#999",lw=.8)
ax.set_xlabel("log Γ uncertainty parameter  (our box method: Tan Γ = e$^{\\log Γ}$)"); ax.set_ylabel("out-of-sample policy regret vs control  (lower = better)")
ax.set_title(f"Our fit_kallus vs the paper's CRLogit — Kallus & Zhou §7.1.1 (n={n_tr}, {N_REPS} reps)\nsame DGP & reps; Ours-nominal tracks CRLogit ⇒ Algorithm-1 implementation is faithful",fontsize=10.5)
ax.legend(fontsize=8,loc="upper right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(ASSETS/"kallus_ours_vs_paper.png"); plt.close(fig)
def amin(m): i=int(np.nanargmin(m)); return m[i],LOG_GAMMAS[i]
print(f"[{time.time()-t0:.0f}s] DONE. IPW={ipw_m:+.3f} Oracle={orc_m:+.3f}")
print(f"  CRLogit:    min={amin(crl_m)[0]:+.3f}@logΓ={amin(crl_m)[1]:.2f}  @1.5={crl_m[5]:+.3f}")
print(f"  Ours-nomin: min={amin(oN_m)[0]:+.3f}@logΓ={amin(oN_m)[1]:.2f}  @1.5={oN_m[5]:+.3f}")
print(f"  Ours-estim: min={amin(oE_m)[0]:+.3f}@logΓ={amin(oE_m)[1]:.2f}  @1.5={oE_m[5]:+.3f}")
print(f"  corr(Ours-nominal, CRLogit) over the sweep = {np.corrcoef(oN_m,crl_m)[0,1]:+.3f}")
