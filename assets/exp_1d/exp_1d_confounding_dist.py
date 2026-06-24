"""Confounding-strength distribution — the single plot that JUSTIFIES the bern-vs-uni result (why robustness
pays in Bernoulli-S but over-hedges in uniform-S). Per training unit the unobserved S tilts the treatment odds
by exp(γ(S−½)); the implied per-unit sensitivity is Γ_i = exp(γ|S−½|). The matched (worst-case) Γ=e^{γ/2}=12.18
is calibrated to the EXTREME S=0,1. Bernoulli-S puts EVERY unit at that extreme (robustness pays); uniform-S
puts most units near S≈½ ⇒ Γ_i≈1 (no confounding), so robustifying to 12.18 over-hedges the typical unit."""
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
GRID=np.round(np.linspace(-1,1,21),6); S_GRID=np.round(np.arange(0,1.0001,0.1),6); n_tr=800; seed=0; gamma=5.0
mG=float(np.exp(gamma/2))   # matched worst-case Γ = 12.18
def S_sample(mode,rng,n):
    rng.choice(GRID,size=n)  # consume the X draw to match the training rng order
    return (rng.uniform(size=n)<0.5).astype(float) if mode=="bern" else rng.choice(S_GRID,size=n)
fig,axes=plt.subplots(1,2,figsize=(12.6,4.9),dpi=140,sharey=True)
for ax,mode,ttl in [(axes[0],"bern","Bernoulli S  (S ∈ {0,1})  — robustness PAYS"),
                    (axes[1],"uni","uniform-11 S  (S ∈ {0,0.1,…,1}) — robustness OVER-HEDGES")]:
    rng=np.random.default_rng(seed); S=S_sample(mode,rng,n_tr)
    Gi=np.exp(gamma*np.abs(S-0.5))               # implied per-unit sensitivity Γ_i
    med=float(np.median(Gi)); frac_ext=float(np.mean(Gi>=mG-1e-6))
    ax.hist(Gi,bins=np.linspace(1,13,25),color="#6d4a7d",alpha=.8,edgecolor="white")
    ax.axvline(mG,ls='--',lw=2,color="#d62728",label=f"matched (worst-case) Γ={mG:.2f}")
    ax.axvline(med,ls='-',lw=2,color="#1f77b4",label=f"median (typical) Γ={med:.2f}")
    ax.set_xlabel("per-unit confounding  Γ_i = exp(γ|S−½|)"); ax.set_title(ttl,fontsize=11); ax.grid(alpha=.2)
    ax.text(0.97,0.74,f"{frac_ext*100:.0f}% of units at the\nworst-case Γ={mG:.1f}",transform=ax.transAxes,
            ha="right",fontsize=9.5,color="#d62728",bbox=dict(boxstyle="round",fc="#fdecec",ec="#d62728",alpha=.9))
    ax.legend(fontsize=9,loc="upper center")
axes[0].set_ylabel("# training units")
fig.suptitle("Why robustness pays in Bernoulli-S but over-hedges in uniform-S — the confounding the methods actually face",y=1.0,fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(NEW/"assets"/"exp_1d"/"confounding_dist.png",bbox_inches="tight"); plt.close(fig)
# also drop a copy in the uniformS dir for that tab
import shutil; shutil.copy(NEW/"assets"/"exp_1d"/"confounding_dist.png",NEW/"assets"/"exp_1d_uniformS"/"confounding_dist.png")
for mode in ("bern","uni"):
    rng=np.random.default_rng(seed); S=S_sample(mode,rng,n_tr); Gi=np.exp(gamma*np.abs(S-0.5))
    print(f"{mode}: median Γ_i={np.median(Gi):.2f}  mean Γ_i={np.mean(Gi):.2f}  %at-worst-case={np.mean(Gi>=mG-1e-6)*100:.0f}%  %near-1(Γ_i<2)={np.mean(Gi<2)*100:.0f}%")
print("DONE -> assets/exp_1d/confounding_dist.png (+ copy in exp_1d_uniformS/)")
