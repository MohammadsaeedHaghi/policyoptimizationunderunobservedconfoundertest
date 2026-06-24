"""3-arm discrete-X DGP — a faithful port of the sibling codebase's discrete study `exp_a` (3 treatments) at γ=5,
with the per-arm capacity-from-train constraint. (Source: code/rw_implementation/experiments/discrete/dgps.py
DEFAULTS["exp_a"] + run.py::cap_from_train.)

  X ~ Unif over a 21-pt grid in [-1,1]; S ~ Bernoulli(½) UNOBSERVED.
  Outcomes:   P(Y^k=1 | X,S) = clip( σ(a_k + b_k X + c_k S), 1e-3, 1-1e-3 ),  a=[0,0,0], b=[-1,0,1], c=[0,0.9,1.8]
  Assignment: P(T=k | X,S) = softmax_k( α_k + β_k X + γ (S-½) d_k ),          α=[0,0,0], β=[-.5,0,.5], d=[-1,0,1], γ=5
  μ_k(X) = ½ P(Y^k=1|X,0) + ½ P(Y^k=1|X,1).
  Capacity (cap_from_train): arm 0 (control) uncapped; arm k≥1 capped at its train share P̂(T=k).
matched Γ = e^{γ/2} = e^{2.5} ≈ 12.18 ; n_train=500 / n_test=2000."""
import numpy as np
from types import SimpleNamespace
K=3; GAMMA=5.0
A=[0.0,0.0,0.0]; B=[-1.0,0.0,1.0]; C=[0.0,0.9,1.8]; ALPHA=[0.0,0.0,0.0]; BETA=[-0.5,0.0,0.5]; D=[-1.0,0.0,1.0]
GRID=np.round(np.arange(-1.0,1.05,0.1),6)            # 21 points
n_tr,n_te=500,2000
matched_G=round(float(np.exp(GAMMA/2)),4)            # ≈ 12.1825
GAMMAS=[1.0,1.5,2.0,3.0,5.0,7.0,matched_G]; mi=GAMMAS.index(matched_G)
ARM_LABELS=["arm 0","arm 1","arm 2"]; ARM_COL=["#1f77b4","#ff7f0e","#d62728"]
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def softmax(U): U=U-U.max(1,keepdims=True); E=np.exp(U); return E/E.sum(1,keepdims=True)
def pa(X,S,k):
    X=np.asarray(X,float); S=np.asarray(S,float); return np.clip(sig(A[k]+B[k]*X+C[k]*S),1e-3,1-1e-3)
def mu_arm(X):                                        # marginal E[Y^k|X] over S, shape (len,K)
    X=np.asarray(X,float); return np.column_stack([0.5*pa(X,0,k)+0.5*pa(X,1,k) for k in range(K)])
def propensity(X,S):                                  # P(T=k|X,S), shape (n,K)
    X=np.asarray(X,float); S=np.asarray(S,float)
    U=np.column_stack([ALPHA[k]+BETA[k]*X+GAMMA*(S-0.5)*D[k] for k in range(K)]); return softmax(U)
def generate(nn,rng):
    X=rng.choice(GRID,size=nn); S=(rng.uniform(size=nn)<0.5).astype(int)
    Yp=np.column_stack([(rng.uniform(size=nn)<pa(X,S,k)).astype(float) for k in range(K)])
    P=propensity(X,S); T=np.array([rng.choice(K,p=P[i]) for i in range(nn)])
    return SimpleNamespace(X=X.reshape(-1,1),S=S,T=T,Ypot=Yp,Y=Yp[np.arange(nn),T],mu=mu_arm(X),P=P)
def cap_from_train(T):                                # arm 0 uncapped, arm k>=1 capped at train share
    T=np.asarray(T).astype(int); return tuple([1.0]+[float(np.mean(T==k)) for k in range(1,K)])
