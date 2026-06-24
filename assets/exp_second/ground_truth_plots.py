"""DGP ground-truth figures for the 'second experiment' (2-arm, 2-D discrete-X). All static heatmaps over (X1,X2):
  potential.png          — control μ0 (flat) + treatment P(Y1=1|X,S=0) and P(Y1=1|X,S=1)  (the X-benefit + S-inflation)
  average_potential.png  — μ0, μ1, CATE=μ1−μ0 + optimal treat-region (treat where μ1>μ0)
  assignment_rule.png    — propensity P(T=1|X,S=0) and P(T=1|X,S=1)  (mis-targeted: treats where proj is LOW)
  inverse_propensity.png — marginal P(T=1|X) and the IPW weight 1/P(T=1|X)
  whats_seen.png         — observed treated mean vs true μ1 and their gap (the hidden confounding)"""
import importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("dgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
AX = dgp.AX; G = dgp.GRID; EXT = [-1, 1, -1, 1]
plt.rcParams.update({"font.size": 10.5})
def H(vals): return np.asarray(vals, float).reshape(11, 11)        # row-major -> (X2 rows, X1 cols)
def panel(ax, Z, title, vmin=0, vmax=1, cmap="viridis", ylab=False):
    im = ax.imshow(Z, origin="lower", extent=EXT, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel("$X_1$")
    if ylab: ax.set_ylabel("$X_2$")
    ax.set_title(title, fontsize=10.2); return im

# ---- (1) potential outcomes ----
mu0 = dgp.pa(G, np.zeros(len(G)), 0)                                # flat sig(0.5)
y1s0 = dgp.pa(G, np.zeros(len(G)), 1); y1s1 = dgp.pa(G, np.ones(len(G)), 1)
fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.2))
panel(ax[0], H(mu0), "Control  $P(Y^0{=}1)=\\sigma(0.5)$  (flat, clean)", ylab=True)
panel(ax[1], H(y1s0), "Treatment  $P(Y^1{=}1\\mid X,S{=}0)$")
im = panel(ax[2], H(y1s1), "Treatment  $P(Y^1{=}1\\mid X,S{=}1)$")
fig.colorbar(im, ax=ax, fraction=0.012, pad=0.02, label="P(Y=1)")
fig.suptitle("Potential outcomes — control is flat; treatment rises with proj$=X_1+0.6X_2$, and the hidden $S$ inflates it ($+5S$)", fontsize=10.8)
fig.savefig(HERE / "potential.png", dpi=120, bbox_inches="tight"); plt.close(fig); print("potential.png")

# ---- (2) average potential outcomes + optimal region ----
mu = dgp.mu_arm(G); cate = mu[:, 1] - mu[:, 0]; opt = (mu[:, 1] > mu[:, 0]).astype(float)
fig, ax = plt.subplots(1, 4, figsize=(16.4, 4.0))
panel(ax[0], H(mu[:, 0]), "$\\mu_0(X)$ control", ylab=True)
panel(ax[1], H(mu[:, 1]), "$\\mu_1(X)$ treatment")
im2 = panel(ax[2], H(cate), "CATE $=\\mu_1-\\mu_0$", vmin=-0.6, vmax=0.6, cmap="RdBu_r"); fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
panel(ax[3], H(opt), "Optimal: TREAT iff $\\mu_1>\\mu_0$", cmap="Greens")
fig.suptitle("Average potential outcomes — optimal rule is a clean region: treat where proj$\\gtrsim 0.04$ (~47% of cells); $E[Y]$ never=0.622, optimal=0.732", fontsize=10.6)
fig.savefig(HERE / "average_potential.png", dpi=120, bbox_inches="tight"); plt.close(fig); print("average_potential.png")

# ---- (3) assignment rule (propensity) ----
ps0 = dgp.propensity(G, np.zeros(len(G))); ps1 = dgp.propensity(G, np.ones(len(G)))
fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.3))
panel(ax[0], H(ps0), "$P(T{=}1\\mid X,S{=}0)$", ylab=True)
im = panel(ax[1], H(ps1), "$P(T{=}1\\mid X,S{=}1)$")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="P(T=1)")
fig.suptitle("Treatment-assignment rule  logit$\\,P(T{=}1)=-$proj$+\\gamma(S-\\frac{1}{2})$ — MIS-targeted (treats where proj is LOW) and $S$-confounded ($\\gamma=5$)", fontsize=10.4)
fig.savefig(HERE / "assignment_rule.png", dpi=120, bbox_inches="tight"); plt.close(fig); print("assignment_rule.png")

# ---- (4) inverse propensity ----
emarg = 0.5 * ps0 + 0.5 * ps1                                      # marginal P(T=1|X)
fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.3))
panel(ax[0], H(emarg), "marginal $\\hat P(T{=}1\\mid X)$", vmin=0, vmax=1, ylab=True)
im = panel(ax[1], H(1.0 / np.clip(emarg, 1e-3, 1)), "treated IPW weight $1/\\hat P(T{=}1\\mid X)$", vmin=1, vmax=float((1 / np.clip(emarg, 1e-3, 1)).max()), cmap="magma")
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
fig.suptitle("Inverse-propensity scores — the mis-targeting makes treated units rare at high-proj (where treatment helps) ⇒ large IPW weights there", fontsize=10.4)
fig.savefig(HERE / "inverse_propensity.png", dpi=120, bbox_inches="tight"); plt.close(fig); print("inverse_propensity.png")

# ---- (5) what the methods see ----
rng = np.random.default_rng(0); d = dgp.generate(120000, rng)
obs = np.full(len(G), np.nan)
key = {tuple(np.round(G[i], 6)): i for i in range(len(G))}
from collections import defaultdict
acc = defaultdict(list)
for j in range(d.X.shape[0]):
    if d.T[j] == 1: acc[tuple(np.round(d.X[j], 6))].append(d.Y[j])
for kk, i in key.items():
    if len(acc[kk]) > 5: obs[i] = np.mean(acc[kk])
fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.2))
panel(ax[0], H(obs), "OBSERVED treated mean", ylab=True)
panel(ax[1], H(mu[:, 1]), "TRUE $\\mu_1$")
gap = obs - mu[:, 1]
im = panel(ax[2], H(gap), "OBSERVED − TRUE (confounding)", vmin=-0.4, vmax=0.4, cmap="RdBu_r"); fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
fig.suptitle("What the methods see — observed treated outcomes are INFLATED above $\\mu_1$ (the treated are an $S{=}1$-selected elite); R-OW must discount it", fontsize=10.4)
fig.savefig(HERE / "whats_seen.png", dpi=120, bbox_inches="tight"); plt.close(fig); print("whats_seen.png")
print("DGP 2-D plots done.")
