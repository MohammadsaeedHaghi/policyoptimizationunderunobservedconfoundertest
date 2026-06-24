"""DGP for the 'non-monotone middle-band, edge-confounded' 1-D 2-arm experiment, where the free-π Wasserstein
methods (R-OW, R-OW-DR) beat Kallus, AIPW and IPW.

STORY (simple + explainable):
  • X ~ Uniform on a 21-point grid in [-1,1];  S ~ Bernoulli(½) is UNOBSERVED.
  • Control succeeds with prob σ(5(S−½)) — it DEPENDS on the unobserved S (marginal still ½), so the confounder
    biases BOTH arms' observed outcomes (and hence the outcome model μ̂ on both sides).
  • Treatment succeeds with prob σ(7(0.36 − X²) + 5(S−½)): it HELPS a MIDDLE band |X|≲0.5 and HURTS the edges
    — a NON-MONOTONE optimal policy (treat the middle subgroup).
  • Assignment T ~ Bernoulli(σ(5(S−½)(1+5X²))): high-S units (who do well on treatment) are treated more, and that
    over-treatment is CONCENTRATED AT THE EDGES (the ×(1+5X²) factor). So at the edges — where treatment truly hurts —
    the treated units are a high-S selected elite whose observed outcomes look great.
  WHY OUR METHODS WIN:
   - Kallus (logistic policy) CANNOT represent "treat a middle band" ⇒ collapses to ≈control.
   - IPW (confounded weights) and AIPW (its logistic μ̂ + non-robust correction) are fooled by the edge selection
     into treating the harmful edges.
   - R-OW / R-OW-DR: the Wasserstein covariate-balance term flags the edge-imbalanced treated distribution and the
     MSM box hedges the confounding ⇒ they correctly treat the middle band. (R-OW > R-O and R-OW-DR > R-O-DR, so the
     WASSERSTEIN term is the differentiator.)
matched Γ = e^{γ/2}=e^{2.5}≈12.18; cap=(1.0, 0.5) (control uncapped, treat ≤50%) so the cap binds on the ~52% band."""
import numpy as np
from types import SimpleNamespace
CS0=5.0; BA=7.0; BT=0.36; CS1=5.0; G=5.0; EK=5.0          # locked DGP params (v1: S-dependent control)
GRID=np.round(np.linspace(-1,1,21),6); n_tr,n_te=500,2000; K=2; CAP=(1.0,0.5)
gt=G; GAMMAS=[1.0,3.0,6.0,9.0,round(float(np.exp(gt/2)),4),16.0]; mi=4
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def pa(X,S,k):
    X=np.asarray(X,float); S=np.asarray(S,float)
    if k==0: return np.clip(sig(CS0*(S-0.5)+0*X),1e-3,1-1e-3)           # control DEPENDS on unobserved S (marginal ½)
    return np.clip(sig(BA*(BT-X**2)+CS1*(S-0.5)),1e-3,1-1e-3)
def generate(nn,rng):
    X=rng.choice(GRID,size=nn); S=(rng.uniform(size=nn)<0.5).astype(int)
    Yp=np.column_stack([(rng.uniform(size=nn)<pa(X,S,0)).astype(float),(rng.uniform(size=nn)<pa(X,S,1)).astype(float)])
    U1=G*(S-0.5)*(1.0+EK*X**2); p1=np.exp(U1)/(1+np.exp(U1)); T=(rng.uniform(size=nn)<p1).astype(int)
    mu=np.column_stack([0.5*pa(X,0,0)+0.5*pa(X,1,0),0.5*pa(X,0,1)+0.5*pa(X,1,1)])
    return SimpleNamespace(X=X.reshape(-1,1),S=S,T=T,Y=Yp[np.arange(nn),T],Ypot=Yp,mu=mu)
