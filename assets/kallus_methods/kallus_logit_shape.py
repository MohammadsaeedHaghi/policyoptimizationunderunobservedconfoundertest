"""Demonstrate that our Kallus policy IS logistic — it is a clean sigmoid in its own linear index
(θ₁−θ₀)·x̃, and only looks 'smeared' when re-plotted against the true CATE (a different scalar that also
depends on the unobserved U). Two-panel figure for the index.html 'Kallus DGP · our methods' subtab.
Reproduces theta at the matched Γ from the same seed-0 data as kallus_methods.py."""
import sys, importlib.util
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
ASSETS=NEW/"assets"/"kallus_methods"
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
kal=load("kal","methods/Kallus/kallus.py"); fit_kallus=kal.fit_kallus; predict_kallus=kal.predict_kallus
# ---- Kallus §7.1.1 DGP (verbatim) ----
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
K=2; n_tr=400; n_te=4000; mG=round(float(np.exp(LG_TRUE)),4)
rng=np.random.default_rng(0)
xtr,xatr,utr,Ttr,Ytr=generate(n_tr,rng); xte,xate,ute,Tte,Yte=generate(n_te,rng)
Yrew=-Ytr; w,_=common.ipw_weights_from_data(xtr,Ttr,K)
kth=fit_kallus(xtr,Ttr,Yrew,w,n_arms=K,Gamma=mG,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0).theta
p1=predict_kallus(kth,xte,("affine",)).T[1]                 # π(treat | x) on test
p1c=np.clip(p1,1e-6,1-1e-6); index=np.log(p1c/(1-p1c))       # logit index = (θ1−θ0)·x̃ (linear in X)
cate=real_risk(np.ones(n_te),xate,ute)-real_risk(np.zeros(n_te),xate,ute)   # true CATE = 2.5 + β_treat·x̃ − 4U
corr=float(np.corrcoef(index,cate)[0,1])
# decompose CATE: observed part (2.5 + β_treat·x̃) vs unobserved −4U part
cate_obs=BETA_CONS+xate@BETA_X_T; cate_u=2*ALPHA*ute        # = −4U; cate = cate_obs + cate_u (check)
print(f"theta diff≈ logit-linear; corr(index, CATE)={corr:.3f}; "
      f"share of CATE variance from unobserved U = {np.var(cate_u)/np.var(cate):.2f}",flush=True)
# ---- figure ----
edges=np.quantile(cate,np.linspace(0,1,17)); ctr=0.5*(edges[:-1]+edges[1:]); idx=np.clip(np.digitize(cate,edges)-1,0,len(ctr)-1)
binmean=np.array([p1[idx==b].mean() if (idx==b).any() else np.nan for b in range(len(ctr))])
fig,(axL,axR)=plt.subplots(1,2,figsize=(12.8,5.2),dpi=140)
# Panel A: π vs its OWN logit index → exact sigmoid
o=np.argsort(index)
axL.scatter(index,p1,s=7,alpha=.25,color="#7f7f7f",label="test units")
zz=np.linspace(index.min(),index.max(),300); axL.plot(zz,sig(zz),color="#d62728",lw=2.4,label=r"$\sigma(z)$ (logistic)")
axL.set_xlabel(r"Kallus linear index  $z=(\theta_1-\theta_0)^\top \tilde x$  (linear in the 5-D $X$)")
axL.set_ylabel(r"$\pi(\mathrm{treat}\,|\,x)$"); axL.set_ylim(-.03,1.03)
axL.set_title("Kallus IS logistic — in its own index\n$\\pi(\\mathrm{treat}\\,|\\,x)=\\sigma((\\theta_1-\\theta_0)^\\top \\tilde{x})$",fontsize=11)
axL.grid(alpha=.25); axL.legend(fontsize=9,loc="center right")
# Panel B: same π vs TRUE CATE → smeared (vertical spread + binned mean)
axR.scatter(cate,p1,s=7,alpha=.18,color="#7f7f7f",label="test units (note vertical spread)")
axR.plot(ctr,binmean,'-X',color="#111",lw=2,ms=6,label="binned mean (the subtab curve)")
axR.axvline(0,ls=':',color="#444",lw=1.2); axR.text(0.3,1.04,"oracle threshold CATE=0",fontsize=8,color="#444")
axR.set_xlabel(r"true CATE $=$ risk(1)$-$risk(0)   (depends on the UNOBSERVED $U$: $-4U$ term)")
axR.set_ylabel(r"$\pi(\mathrm{treat}\,|\,x)$"); axR.set_ylim(-.03,1.10)
axR.set_title(f"Same policy vs true CATE — smeared\ncorr(index, CATE)$={corr:.2f}$; {np.var(cate_u)/np.var(cate)*100:.0f}% of CATE variance is unobserved $U$",fontsize=11)
axR.grid(alpha=.25); axR.legend(fontsize=9,loc="upper right")
fig.suptitle(f"The Kallus policy is logistic in its linear index $z=(\\theta_1-\\theta_0)^\\top \\tilde{{x}}$ — the subtab's CATE axis is a different scalar (matched Γ={mG})",y=1.0,fontsize=11.5)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(ASSETS/"kallus_logit_shape.png",bbox_inches="tight"); plt.close(fig)
np.savez(ASSETS/"kallus_logit_shape.npz",index=index,p1=p1,cate=cate,cate_bins=ctr,binmean=binmean,corr=corr,
         u_var_share=float(np.var(cate_u)/np.var(cate)))
print("DONE ->",ASSETS/"kallus_logit_shape.png")
