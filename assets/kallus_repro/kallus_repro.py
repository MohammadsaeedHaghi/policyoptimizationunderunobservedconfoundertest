"""Faithful reproduction of Kallus & Zhou (2021) Section 7.1.1 'Binary Treatments' synthetic experiment
(Figure 1: out-of-sample policy regret vs log(Gamma)).  DGP + CRLogit Algorithm-1 ported from the official repo
(paper/kallus/repo/{data_scenarios,subgrad,unconfoundedness_fns}.py). Inner worst-case = closed-form self-normalised
sort (no LP). Methods: IPW (naive logistic), GRF (causal-forest CATE direct rule), CRLogit (confounding-robust logistic).
"""
import sys, time
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ASSETS=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1/assets/kallus_repro")
ASSETS.mkdir(parents=True,exist_ok=True)
# ---- DGP constants (verbatim from data_scenarios.py) ----
BETA_CONS=2.5
BETA_X   =np.array([0,.5,-0.5,0,0,0.0])         # on x_aug=[x,1] (6-dim)
BETA_X_T =np.array([-1.5,1,-1.5,1.,0.5,0.0])    # treatment interaction on x_aug
BETA_TC  =np.array([0,.75,-.5,0,-1,0.0])        # nominal propensity coeffs on x_aug
MU_X     =np.array([-1,.5,-1,0,-1.0])           # covariate mean (5-dim)
ALPHA=-2.0       # repo 'alpha': confounding coef on u*(2T-1)
WCOEF=1.5        # repo 'w': coef on u
LG_TRUE=1.5      # true confounding strength (log odds-ratio); REAL_PROP_LOG Gamma=1.5
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def get_bnds(Q,LG):                              # unconfoundedness_fns.get_bnds
    Q=np.asarray(Q,float); eL=np.exp(LG); eLm=np.exp(-LG)
    p_hi=eL*Q/(1-Q+eL*Q); p_lo=eLm*Q/(1-Q+eLm*Q)
    return 1.0/p_hi, 1.0/p_lo                     # a_bnd, b_bnd  (a<b, bounds on 1/p)
def real_risk(T,xa,u):                            # data_scenarios.real_risk
    T=np.asarray(T,float)
    return T*BETA_CONS + xa@BETA_X + (xa@BETA_X_T)*T + ALPHA*u*(2*T-1) + WCOEF*u
def real_risk_prob(pi1,xa,u):                     # E[ pi1*Y(1) + (1-pi1)*Y(0) ]
    return float(np.mean(pi1*real_risk(np.ones(len(u)),xa,u)+(1-pi1)*real_risk(np.zeros(len(u)),xa,u)))
def generate(n,rng):                              # data_scenarios.generate_log_data (binary)
    u=(rng.random(n)>0.5).astype(float)           # Bernoulli(1/2) unobserved confounder
    x=rng.standard_normal((n,5))+MU_X*(2*u-1)[:,None]   # X | u ~ N((2u-1)mu_x, I_5)  [paper typo'd 2T-1]
    xa=np.hstack([x,np.ones((n,1))])
    nominal=sig(xa@BETA_TC)                        # true nominal propensity (for data generation)
    optT=(real_risk(np.ones(n),xa,u)<real_risk(np.zeros(n),xa,u)).astype(int)   # CATE-optimal arm (lower risk better)
    a_bnd,b_bnd=get_bnds(nominal,LG_TRUE); p_lo=1/b_bnd; p_hi=1/a_bnd
    trueQ=np.where(optT==1,p_hi,p_lo)              # benefit-from-treatment units treated MORE (confounding)
    T=(rng.random(n)<trueQ).astype(int)
    Y=real_risk(T,xa,u)+2*rng.standard_normal(n)   # noise std 2
    return x,xa,u,T,Y
# ---- inner worst-case: max over W in [a,b] of (sum W coef)/(sum W) (self-normalised) ----
def worst_case(coef,a,b):
    o=np.argsort(coef); c=coef[o]; aa=a[o]; bb=b[o]; n=len(c)
    pa_c=np.concatenate([[0],np.cumsum(aa*c)]); pa=np.concatenate([[0],np.cumsum(aa)])
    sb_c=np.concatenate([np.cumsum((bb*c)[::-1])[::-1],[0]]); sb=np.concatenate([np.cumsum(bb[::-1])[::-1],[0]])
    den=pa+sb; den[den==0]=1e-12; vals=(pa_c+sb_c)/den; k=int(np.argmax(vals))
    W=np.where(np.arange(n)<k,aa,bb); Ws=np.empty(n); Ws[o]=W
    return float(vals[k]),Ws
# ---- CRLogit (Algorithm 1); LG=0 => naive IPW logistic ----
def crlogit(xa,T,Y,ehat,LG,*,n_restarts,n_iters,clamp,rng,eta0=1.0,kappa=0.5):
    fq=np.where(T==1,ehat,1-ehat)                  # observed-treatment nominal propensity
    a,b=get_bnds(fq,LG); Tsgn=2*T-1.0; p=xa.shape[1]
    best_th,best_R=None,np.inf
    for _ in range(n_restarts):
        th=rng.standard_normal(p)*0.25; acc=np.zeros(p)
        for it in range(n_iters):
            pi1=sig(xa@th); coef=Y*Tsgn*pi1
            _,Ws=worst_case(coef,a,b); What=Ws/Ws.sum()
            grad=(Y*Tsgn*What*pi1*(1-pi1))@xa       # envelope-theorem subgradient (minimise worst-case regret)
            th=th-(eta0/(it+1)**kappa)*grad; acc+=th
        thb=acc/n_iters; pi1=sig(xa@thb); R,_=worst_case(Y*Tsgn*pi1,a,b)
        if R<best_R: best_R,best_th=R,thb.copy()
    # robust methods fall back to never-treat if they cannot certify improvement (repo lines 1107-1109)
    if clamp and best_R>0: return np.zeros(p),0.0,True
    return best_th,best_R,False
def grf_decision(x,T,Y):
    m1=RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=0)
    m0=RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=0)
    if (T==1).sum()>2: m1.fit(x[T==1],Y[T==1])
    if (T==0).sum()>2: m0.fit(x[T==0],Y[T==0])
    return lambda xt: (m1.predict(xt)<m0.predict(xt)).astype(float)   # treat if CATE lowers risk
# ---- run ----
N_REPS=int(sys.argv[1]) if len(sys.argv)>1 else 50
n_tr=200; n_te=5000; N_RST=6; N_IT=40
LOG_GAMMAS=np.concatenate([np.arange(0.1,2.01,0.1),[2.5,3.0,4.0,5.0]])
t0=time.time(); rng=np.random.default_rng(0)
crl=np.full((N_REPS,len(LOG_GAMMAS)),np.nan); ipw=np.full(N_REPS,np.nan); grf=np.full(N_REPS,np.nan); orc=np.full(N_REPS,np.nan)
for r in range(N_REPS):
    x,xa,u,T,Y=generate(n_tr,rng)
    xt,xat,ut,Tt,Yt=generate(n_te,rng)
    if len(np.unique(T))<2: continue
    cate_te=real_risk(np.ones(n_te),xat,ut)-real_risk(np.zeros(n_te),xat,ut)   # true treatment effect on risk (test)
    # nominal propensity model = sigma(beta_TC . x): the unconfoundedness-ASSUMED propensity (matches the repo's
    # logistic_pol_asgn(beta_T_conf, x_)). The sensitivity analysis is precisely about how wrong this nominal can
    # be (odds-ratio up to e^logGamma). It uses X only (NOT the unobserved confounder u / e(x,u)).
    eh=np.clip(sig(xa@BETA_TC),1e-3,1-1e-3)
    base=real_risk_prob(np.zeros(n_te),xat,ut)
    orc[r]=real_risk_prob((cate_te<0).astype(float),xat,ut)-base   # oracle: treat iff true CATE lowers risk
    # IPW (naive logistic, LG=0, no clamp)
    th,_,_=crlogit(xa,T,Y,eh,0.0,n_restarts=N_RST,n_iters=N_IT,clamp=False,rng=rng)
    ipw[r]=real_risk_prob(sig(xat@th),xat,ut)-base
    # GRF (direct CATE rule)
    dec=grf_decision(x,T,Y); grf[r]=real_risk_prob(dec(xt),xat,ut)-base
    # CRLogit over the Gamma sweep (robust, clamp)
    for gi,LG in enumerate(LOG_GAMMAS):
        th,_,cl=crlogit(xa,T,Y,eh,float(LG),n_restarts=N_RST,n_iters=N_IT,clamp=True,rng=rng)
        pi1=np.zeros(n_te) if cl else sig(xat@th); crl[r,gi]=real_risk_prob(pi1,xat,ut)-base
    if (r+1)%10==0: print(f"[{time.time()-t0:.0f}s] rep {r+1}/{N_REPS}",flush=True)
# aggregate (mean +/- SE)
def msd(a,ax=0): m=np.nanmean(a,axis=ax); s=np.nanstd(a,axis=ax)/np.sqrt(np.sum(~np.isnan(a),axis=ax)); return m,s
crl_m,crl_s=msd(crl); ipw_m,ipw_s=msd(ipw); grf_m,grf_s=msd(grf); orc_m,orc_s=msd(orc)
np.savez(ASSETS/"kallus_repro_data.npz",LOG_GAMMAS=LOG_GAMMAS,crl=crl,ipw=ipw,grf=grf,orc=orc,
         crl_m=crl_m,crl_s=crl_s,ipw_m=ipw_m,ipw_s=ipw_s,grf_m=grf_m,grf_s=grf_s,orc_m=orc_m,orc_s=orc_s,n_reps=N_REPS,lg_true=LG_TRUE)
# ---- plot (reproduce Figure 1) ----
fig,ax=plt.subplots(figsize=(7.6,4.9),dpi=140)
ax.axhline(orc_m,color="#111",ls="--",lw=1.3,label="Oracle (true CATE) — ceiling")
ax.axhline(ipw_m,color="#8c564b",lw=2.2,label="IPW (naive logistic) — harmed")
ax.fill_between(LOG_GAMMAS,ipw_m-ipw_s,ipw_m+ipw_s,color="#8c564b",alpha=.12)
ax.axhline(grf_m,color="#ff7f0e",ls=":",lw=1.6,alpha=.85,label="GRF (direct CATE forest) — differs from paper*")
ax.plot(LOG_GAMMAS,crl_m,'-o',color="#d62728",lw=2.4,ms=4.5,label="CRLogit (confounding-robust)",zorder=5)
ax.fill_between(LOG_GAMMAS,crl_m-crl_s,crl_m+crl_s,color="#d62728",alpha=.16)
ax.axvline(LG_TRUE,ls=':',color="#555",lw=1.2); ax.text(LG_TRUE+0.05,ax.get_ylim()[0]+.02,"true log Γ=1.5",fontsize=8,va="bottom",color="#555",rotation=90)
ax.axhline(0,color="#999",lw=.8)
ax.set_xlabel("log(Γ) uncertainty parameter"); ax.set_ylabel("policy regret against control")
ax.set_title(f"Reproduction of Kallus & Zhou (2021) Fig. 1 — §7.1.1 synthetic\n(n={n_tr}, {N_REPS} reps, true log Γ=1.5; lower=better)")
ax.legend(fontsize=8,loc="center right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(ASSETS/"figure1_regret_vs_loggamma.png"); plt.close(fig)
print(f"[{time.time()-t0:.0f}s] DONE -> {ASSETS}")
print(f"IPW={ipw_m:+.3f}±{ipw_s:.3f}  GRF={grf_m:+.3f}±{grf_s:.3f}  Oracle={orc_m:+.3f}")
bi=int(np.nanargmin(crl_m)); print(f"CRLogit min regret={crl_m[bi]:+.3f} at logΓ={LOG_GAMMAS[bi]:.2f}; @logΓ=1.5 -> {crl_m[np.argmin(np.abs(LOG_GAMMAS-1.5))]:+.3f}")
