"""Ground-truth outcome-surface plots for the NON-MONOTONE DGP (same style as assets/exp_1d/), to explain the data
before any method sees it. Two figures: outcome_means_vs_x.png (E[Y^k|X] per arm + the unobserved-S band) and
outcome_heatmaps_xs.png (P(Y^k=1|X,S) heatmap per arm). KEY contrasts vs exp_1d: BOTH arms are S-confounded (control
is flat in X but split by S), and treatment is NON-MONOTONE in X (a middle band)."""
import sys, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("dgp",str(HERE/"dgp.py")); dgp=importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
X=np.sort(dgp.GRID); BT=dgp.BT; xband=float(np.sqrt(BT))   # treat-band edge: CATE>0 iff |X|<sqrt(BT)=0.6
c0=dgp.pa(X,0,0); c1=dgp.pa(X,1,0); cm=0.5*c0+0.5*c1        # control: S=0, S=1, marginal
t0=dgp.pa(X,0,1); t1=dgp.pa(X,1,1); tm=0.5*t0+0.5*t1        # treatment: S=0, S=1, marginal
plt.rcParams.update({"font.size":11})

# ---------- (1) expected outcome per arm vs X ----------
fig,ax=plt.subplots(figsize=(9.4,5.4))
ax.axvspan(-xband,xband,color="#2ca02c",alpha=.07,zorder=0,label="optimal: TREAT this middle band (CATE>0)")
ax.fill_between(X,t0,t1,color="#d62728",alpha=.12,zorder=1,label="treatment band over unobserved S")
ax.fill_between(X,c0,c1,color="#7f7f7f",alpha=.12,zorder=1,label="control band over unobserved S")
ax.plot(X,t0,"--",color="#f2a0a0",lw=1.3,label=r"treat | S=0  ($\sigma(7(0.36-X^2)-2.5)$)")
ax.plot(X,t1,"--",color="#7a1212",lw=1.3,label=r"treat | S=1  ($\sigma(7(0.36-X^2)+2.5)$)")
ax.plot(X,c0,":",color="#aaaaaa",lw=1.2); ax.plot(X,c1,":",color="#555555",lw=1.2)
ax.plot(X,cm,"-o",color="#7f7f7f",lw=2.4,ms=4,label=r"$E[Y^0|X]$  control (flat $\approx0.50$, but S-split)")
ax.plot(X,tm,"-o",color="#d62728",lw=2.8,ms=4,label=r"$E[Y^1|X]$  treatment (avg over S) — non-monotone")
for xv in (-xband,xband):
    ax.axvline(xv,color="#333",ls=":",lw=1);
ax.annotate("treatment beats control\nonly for |X| < 0.6", xy=(0,0.5), xytext=(0.0,0.30), ha="center",
            arrowprops=dict(arrowstyle="->",color="#444"), fontsize=10, color="#333")
ax.set_xlabel("X"); ax.set_ylabel("expected outcome E[Y | X]"); ax.set_ylim(-.02,1.02); ax.set_xlim(-1.03,1.03)
ax.set_title("True expected outcome per arm vs X  (averaged over the unobserved S) — NON-MONOTONE",fontsize=12)
ax.grid(alpha=.25); ax.legend(fontsize=8.3,loc="upper right",framealpha=.9,ncol=1)
fig.tight_layout(); fig.savefig(HERE/"outcome_means_vs_x.png",dpi=120); plt.close(fig); print("outcome_means_vs_x.png")

# ---------- (2) per-(X,S) heatmaps, one per arm ----------
Slev=np.array([0.0,1.0])
A0=np.array([[dgp.pa(np.array([x]),s,0)[0] for x in X] for s in Slev])   # (2, nX)
A1=np.array([[dgp.pa(np.array([x]),s,1)[0] for x in X] for s in Slev])
fig,ax=plt.subplots(1,2,figsize=(12.4,4.2))
ext=[X.min(),X.max(),-0.5,1.5]
for a,(A,ttl) in zip(ax,[(A0,"Arm 0 — control:  P(Y=1 | X, S)"),(A1,"Arm 1 — treatment:  P(Y=1 | X, S)")]):
    im=a.imshow(A,origin="lower",aspect="auto",extent=ext,vmin=0,vmax=1,cmap="viridis")
    a.set_yticks([0,1]); a.set_yticklabels(["S=0","S=1"]); a.set_xlabel("X"); a.set_title(ttl,fontsize=11)
    for s in (0,1):
        for xi in (3,10,17):
            v=A[s,xi]; a.text(X[xi],s,"%.2f"%v,ha="center",va="center",fontsize=9,color="white" if v<0.5 else "black",fontweight="bold")
fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02,label="P(Y=1)")
fig.suptitle("Ground-truth success probability for each (X, S) — BOTH arms are S-confounded; treatment is NON-MONOTONE in X",fontsize=12)
fig.savefig(HERE/"outcome_heatmaps_xs.png",dpi=120,bbox_inches="tight"); plt.close(fig); print("outcome_heatmaps_xs.png")

# ---------- (3) TRUE treatment-assignment rule P(treat | X, S) ----------
pt_s1=dgp.sig(dgp.G*0.5*(1+dgp.EK*X**2)); pt_s0=dgp.sig(-dgp.G*0.5*(1+dgp.EK*X**2))
fig,ax=plt.subplots(figsize=(9.2,4.9))
ax.axvspan(-xband,xband,color="#2ca02c",alpha=.06,label="(true treat-band, for reference)")
ax.plot(X,pt_s1,"-o",color="#d62728",ms=4.5,lw=2.4,label=r"$P(\mathrm{treat}\mid X,\ S{=}1)=\sigma(2.5(1+5X^2))$")
ax.plot(X,pt_s0,"-o",color="#7f7f7f",ms=4.5,lw=2.4,label=r"$P(\mathrm{treat}\mid X,\ S{=}0)=\sigma(-2.5(1+5X^2))$")
ax.axhline(0.5,ls=":",c="#999",lw=1)
ax.set_xlabel("X"); ax.set_ylabel("P(treat | X, S)"); ax.set_ylim(-.03,1.03)
ax.set_title("True treatment-assignment rule  T ~ Bernoulli( σ(5(S-½)(1+5X²)) )\nthe unobserved S decides who is treated; the gap WIDENS toward the edges",fontsize=11.5)
ax.annotate("at the edges, assignment is\nnearly deterministic in S:\nS=1 → treat, S=0 → control",xy=(0.93,0.5),xytext=(0.32,0.44),fontsize=9.2,color="#333",
            arrowprops=dict(arrowstyle="->",color="#555"),ha="center")
ax.legend(fontsize=9.5,loc="center left"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(HERE/"assignment_rule.png",dpi=120); plt.close(fig); print("assignment_rule.png")
print("CATE>0 band: |X| < %.2f ; treat marginal: edges %.3f, center %.3f ; control marginal %.3f"%(xband,tm[0],tm[len(X)//2],cm[0]))
