"""DGP ground-truth figures for the 3-arm exp_a (γ=5) experiment (all static):
  outcome_surfaces.png — μ_k(X) per arm + optimal-arm strip
  outcome_heatmaps.png — P(Y^k=1|X,S) per arm (2 S-rows each)
  assignment_rule.png  — P(T=k|X,S) stacked, one panel per S (the treatment-assignment rule)
  whats_seen.png       — observed mean outcome per arm vs the true μ_k (the confounding the methods see)"""
import sys, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("dgp",str(HERE/"dgp.py")); dgp=importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
X=dgp.GRID; K=dgp.K; COL=dgp.ARM_COL; LAB=dgp.ARM_LABELS
mu=dgp.mu_arm(X); best=np.argmax(mu,1)
plt.rcParams.update({"font.size":11})

# ---- (1) outcome surfaces μ_k(X) + optimal-arm strip ----
fig,ax=plt.subplots(figsize=(9.0,4.9))
for k in range(K): ax.plot(X,mu[:,k],"-o",color=COL[k],ms=4,lw=2.2,label=r"$\mu_%d(X)=E[Y^%d|X]$"%(k,k))
ax.set_ylim(-0.06,1.02); ax.set_xlabel("X"); ax.set_ylabel(r"$\mu_k(X)$")
# thin strip: argmax arm colour just below the axes
for i,xi in enumerate(X):
    ax.axvspan(xi-0.05,xi+0.05,ymin=0.0,ymax=0.04,color=COL[best[i]],alpha=.9)
ax.text(-1.0,-0.10,"best arm (NO cap):",fontsize=8,va="center")
ax.set_title("True arm outcome surfaces  μ_k(X)=E[Y^k|X]  —  the best arm shifts with X\n(strip = UNCONSTRAINED argmax arm; under the arm-2 capacity cap the deployable optimum rations arm 2 to high X)",fontsize=10.0)
ax.legend(fontsize=9.5,loc="upper center",ncol=3); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(HERE/"outcome_surfaces.png",dpi=120); plt.close(fig); print("outcome_surfaces.png")

# ---- (2) per-(X,S) heatmaps, one per arm ----
fig,ax=plt.subplots(1,K,figsize=(13.2,3.6))
ext=[X.min(),X.max(),-0.5,1.5]
for k in range(K):
    Ak=np.array([[dgp.pa(np.array([x]),s,k)[0] for x in X] for s in (0.0,1.0)])  # (2, nX)
    im=ax[k].imshow(Ak,origin="lower",aspect="auto",extent=ext,vmin=0,vmax=1,cmap="viridis")
    ax[k].set_yticks([0,1]); ax[k].set_yticklabels(["S=0","S=1"]); ax[k].set_xlabel("X")
    ax[k].set_title("Arm %d:  P(Y=1 | X, S)"%k,fontsize=10.5)
    for s in (0,1):
        for xi in (3,10,17): v=Ak[s,xi]; ax[k].text(X[xi],s,"%.2f"%v,ha="center",va="center",fontsize=8.5,color="white" if v<0.5 else "black",fontweight="bold")
fig.colorbar(im,ax=ax,fraction=0.02,pad=0.02,label="P(Y=1)")
fig.suptitle("Ground-truth success probability per arm — the unobserved S inflates the higher arms (c = [0, 0.9, 1.8])",fontsize=11.5)
fig.savefig(HERE/"outcome_heatmaps.png",dpi=120,bbox_inches="tight"); plt.close(fig); print("outcome_heatmaps.png")

# ---- (3) assignment rule P(T=k|X,S), stacked, one panel per S ----
fig,ax=plt.subplots(1,2,figsize=(12.0,4.4),sharey=True)
for j,s in enumerate((0,1)):
    P=dgp.propensity(X,np.full_like(X,s))            # (nX, K)
    for k in range(K):
        ax[j].plot(X,P[:,k],"-o",color=COL[k],ms=4,lw=2.2,label=r"$P(T{=}%d\mid X,S)$"%k)
    ax[j].set_xlim(X.min(),X.max()); ax[j].set_ylim(-0.03,1.03); ax[j].set_xlabel("X"); ax[j].set_title("S = %d"%s,fontsize=11.5); ax[j].grid(alpha=.25)
ax[0].set_ylabel("P(T=k | X, S)"); ax[1].legend(fontsize=10,loc="center right",framealpha=.9)
fig.suptitle("True treatment-assignment rule  P(T=k|X,S)=softmax(β_k X + γ(S-½)d_k),  γ=5\nS=0 is X-tilted (arm 0 low-X → arm 2 high-X); S=1 is pushed toward the higher arms (d=[-1,0,1])",fontsize=10.8)
fig.tight_layout(); fig.savefig(HERE/"assignment_rule.png",dpi=120); plt.close(fig); print("assignment_rule.png")

# ---- (4) what the methods see: observed per-arm outcome vs true μ_k ----
rng=np.random.default_rng(0); d=dgp.generate(40000,rng)
obs=np.full((len(X),K),np.nan)
for i,xv in enumerate(X):
    m=np.isclose(d.X.ravel(),xv)
    for k in range(K):
        yk=d.Y[m&(d.T==k)]
        if len(yk)>5: obs[i,k]=yk.mean()
fig,ax=plt.subplots(1,K,figsize=(13.2,3.9),sharey=True)
for k in range(K):
    ax[k].plot(X,obs[:,k],"-o",color=COL[k],ms=4,lw=2.2,label="OBSERVED  (arm %d treated units)"%k)
    ax[k].plot(X,mu[:,k],"--",color="#333",lw=1.8,label=r"TRUE $\mu_%d(X)$"%k)
    ax[k].set_xlabel("X"); ax[k].set_ylim(-0.02,1.02); ax[k].set_title("Arm %d"%k,fontsize=10.5); ax[k].grid(alpha=.25); ax[k].legend(fontsize=8.3,loc="lower right")
ax[0].set_ylabel("mean outcome")
fig.suptitle("What the methods see: observed mean outcome per arm (solid) vs the truth μ_k (dashed)\n— high-S units select the higher arms, so their observed outcomes are INFLATED above μ_k (the confounding)",fontsize=10.6)
fig.tight_layout(); fig.savefig(HERE/"whats_seen.png",dpi=120); plt.close(fig); print("whats_seen.png")
print("DGP plots done.")
