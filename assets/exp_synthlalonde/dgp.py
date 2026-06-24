"""SEMI-SYNTHETIC LaLonde DGP — REAL LaLonde ages + a SIMULATED outcome, engineered so the free-π Wasserstein methods
(R-OW, R-OW-DR) are the best. This is the IHDP/ACIC recipe (real covariates, known counterfactuals) applied to LaLonde.

CONSTRUCTION (careful):
  • COVARIATE = real age. We pool the LaLonde experimental NSW (445) and PSID (2490) ages and quantile-bin them to a
    21-point grid x∈[-1,1] (x=-1 youngest, x=+1 oldest). The age DISTRIBUTION is real; only the outcome is simulated.
  • HIDDEN CONFOUNDER S ~ Bernoulli(½) — latent "motivation/ability", UNOBSERVED by every method.
  • OUTCOME = a success indicator (e.g. employed / earned above a threshold in 1978). σ is the logistic sigmoid:
        Y(0) ~ Bernoulli( σ( 5(S-½) ) )                      # control: motivation helps; flat in age (marginal ½)
        Y(1) ~ Bernoulli( σ( 7(0.36 - x²) + 5(S-½) ) )       # training HELPS a prime-age band |x|≲0.6, HURTS the young/old
                                                              #   edges (lost-wages story) — a NON-MONOTONE effect.
  • ASSIGNMENT (the confounding): more-motivated workers are treated more, and that over-selection is CONCENTRATED AT THE
    EDGES, where training actually hurts:
        T ~ Bernoulli( σ( 5(S-½)(1 + 5x²) ) )
    So at the harmful edges the treated are a high-motivation elite whose observed outcomes look fine — fooling IPW/AIPW.
  WHY R-OW WINS: Kallus's logistic policy and AIPW's monotone μ̂ cannot represent "treat a middle age-band"; IPW/AIPW are
  fooled by the edge selection into treating the harmful edges; R-OW is free-π (any band) + its Wasserstein covariate-balance
  term flags the edge-imbalanced treated age-distribution + the MSM box hedges the confounding ⇒ it treats the right band.
matched Γ = e^{γ/2} = e^{2.5} ≈ 12.18; cap=(1.0, 0.5) (control uncapped, treat ≤50%); n=800 train / 2000 test."""
import numpy as np, pandas as pd
from types import SimpleNamespace
from pathlib import Path
NEW=Path(__file__).resolve().parents[2]
DATA=NEW/"assets"/"exp_realdata"/"data"
CS0=5.0; CS1=5.0; BA=7.0; BT=0.36; G=5.0; EK=5.0; NB=21
n_tr,n_te=800,2000; K=2; CAP=(1.0,0.5)
GAMMAS=[1.0,3.0,6.0,9.0,round(float(np.exp(G/2)),4),16.0]; mi=4
_AGE=np.r_[pd.read_stata(DATA/"nsw_dw.dta").age.values, pd.read_stata(DATA/"psid_controls.dta").age.values].astype(float)
_E=np.quantile(_AGE,np.linspace(0,1,NB+1)); _E[0]-=1e-9; _E[-1]+=1e-9; CTR=np.linspace(-1,1,NB)
def age2x(a): return CTR[np.clip(np.digitize(np.asarray(a,float),_E)-1,0,NB-1)]
def x2age(x):  # representative age (bin midpoint) for a grid value, for plotting
    idx=np.clip(np.round((np.asarray(x,float)+1)/2*(NB-1)).astype(int),0,NB-1); mid=(_E[:-1]+_E[1:])/2; return mid[idx]
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def pa(x,S,k):
    x=np.asarray(x,float); S=np.asarray(S,float)
    if k==0: return np.clip(sig(CS0*(S-0.5)+0*x),1e-3,1-1e-3)
    return np.clip(sig(BA*(BT-x**2)+CS1*(S-0.5)),1e-3,1-1e-3)
def generate(nn,rng):
    a=rng.choice(_AGE,size=nn); x=age2x(a); S=(rng.uniform(size=nn)<0.5).astype(float)
    Yp=np.column_stack([(rng.uniform(size=nn)<pa(x,S,0)).astype(float),(rng.uniform(size=nn)<pa(x,S,1)).astype(float)])
    U1=G*(S-0.5)*(1.0+EK*x**2); T=(rng.uniform(size=nn)<sig(U1)).astype(int)
    mu=np.column_stack([0.5*pa(x,0,0)+0.5*pa(x,1,0),0.5*pa(x,0,1)+0.5*pa(x,1,1)])
    return SimpleNamespace(x=x.reshape(-1,1),age=a,S=S,T=T,Y=Yp[np.arange(nn),T],Ypot=Yp,mu=mu)
