"""DGP ground-truth figures for the 'first experiment' (Case 4, discrete-X, 2-arm). All static:
  potential.png          — P(Y^k=1|X,S) per arm, S=0 vs S=1 (the non-monotone training effect + S boost)
  average_potential.png  — μ_k(X)=E_S[P(Y^k=1|X,S)] + optimal-arm strip (treat iff X>=0)
  assignment_rule.png    — π^1(X,S)=P(T=1|X,S) for S=0,1 (the treatment-assignment / confounding rule)
  inverse_propensity.png — inverse-propensity weights 1/π (treated) and 1/(1-π) (control) per S
  whats_seen.png         — observed mean outcome per arm vs the true μ_k (the confounding the methods see)"""
import importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("dgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
X = dgp.GRID; K = dgp.K; G = dgp.GAMMA_CONF
C0, C1 = "#1f77b4", "#d62728"          # control (arm 0) blue, treatment (arm 1) red
SCOL = {0: "#94a3b8", 1: "#0f172a"}    # S=0 light, S=1 dark
plt.rcParams.update({"font.size": 11})
mu = dgp.mu_arm(X); best = np.argmax(mu, 1)

# ---- (1) potential outcomes per arm, S=0 vs S=1 ----
fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
for k, (a, col, ttl) in enumerate(zip(ax, (C0, C1), ("Arm 0: no training  P(Y⁰=1 | X,S)", "Arm 1: training  P(Y¹=1 | X,S)"))):
    for s in (0, 1):
        a.plot(X, dgp.pa(X, np.full_like(X, s), k), "-o", color=col, ms=4, lw=2.3,
               alpha=1.0 if s == 1 else 0.55, ls="-" if s == 1 else "--", label="S=%d" % s)
    a.set_xlabel("X (employability score)"); a.set_title(ttl, fontsize=10.6); a.set_ylim(-0.03, 1.03); a.grid(alpha=.25); a.legend(fontsize=9.5, loc="center left")
ax[0].set_ylabel("P(Y=1)")
fig.suptitle("Potential outcomes — training HELPS X≥0 (σ(X+5)≈1) and HURTS X<0 (σ(X−5)≈0); the hidden trait S boosts BOTH arms", fontsize=10.8)
fig.tight_layout(); fig.savefig(HERE / "potential.png", dpi=120); plt.close(fig); print("potential.png")

# ---- (2) average potential outcomes μ_k(X) + optimal-arm strip ----
fig, ax = plt.subplots(figsize=(9.0, 4.9))
ax.plot(X, mu[:, 0], "-o", color=C0, ms=4, lw=2.4, label=r"$\mu_0(X)$  control")
ax.plot(X, mu[:, 1], "-o", color=C1, ms=4, lw=2.4, label=r"$\mu_1(X)$  treatment")
ax.axvline(0, color="#64748b", lw=1.2, ls=":")
ax.set_ylim(-0.06, 1.04); ax.set_xlabel("X"); ax.set_ylabel(r"$\mu_k(X)=E_S[\,P(Y^k=1\mid X,S)\,]$")
for i, xi in enumerate(X): ax.axvspan(xi - 0.05, xi + 0.05, ymin=0.0, ymax=0.04, color=C1 if best[i] == 1 else C0, alpha=.9)
ax.text(-1.0, -0.10, "optimal arm:", fontsize=8.5, va="center")
ax.set_title("Average potential outcomes — the optimal rule is a clean threshold: TREAT iff X≥0\n(control μ₀ rises smoothly; treatment μ₁≈1 for X≥0, ≈0.5 for X<0)", fontsize=10.4)
ax.legend(fontsize=10, loc="center right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(HERE / "average_potential.png", dpi=120); plt.close(fig); print("average_potential.png")

# ---- (3) treatment-assignment rule π^1(X,S) ----
fig, ax = plt.subplots(figsize=(9.0, 4.6))
for s in (0, 1):
    ax.plot(X, dgp.propensity(X, np.full_like(X, s)), "-o", color=SCOL[s], ms=4, lw=2.4, label=r"$\pi^1(X,S{=}%d)$" % s)
ax.axhline(0.05, color="#cbd5e1", lw=1, ls="--"); ax.axhline(0.95, color="#cbd5e1", lw=1, ls="--")
ax.set_ylim(-0.03, 1.03); ax.set_xlabel("X"); ax.set_ylabel("P(T=1 | X, S)")
ax.set_title("Treatment-assignment rule  π¹(X,S)=clip(σ(X+γS−2),0.05,0.95),  γ=%.1f\nS=1 (high perseverance) are treated far more — and X still matters (the curves slope) — clipped to [0.05,0.95]" % G, fontsize=10.2)
ax.legend(fontsize=10, loc="upper left"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(HERE / "assignment_rule.png", dpi=120); plt.close(fig); print("assignment_rule.png")

# ---- (4) inverse-propensity weights ----
fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.4))
for s in (0, 1):
    p = dgp.propensity(X, np.full_like(X, s))
    ax[0].plot(X, 1.0 / p, "-o", color=SCOL[s], ms=4, lw=2.3, label="S=%d" % s)
    ax[1].plot(X, 1.0 / (1.0 - p), "-o", color=SCOL[s], ms=4, lw=2.3, label="S=%d" % s)
ax[0].set_title("Treated weight  1/π¹(X,S)", fontsize=10.6); ax[1].set_title("Control weight  1/(1−π¹(X,S))", fontsize=10.6)
for a in ax: a.set_xlabel("X"); a.grid(alpha=.25); a.legend(fontsize=9.5)
ax[0].set_ylabel("inverse-propensity weight")
fig.suptitle("Inverse-propensity scores — S=0 treated units (rare) get huge weight; S=1 control units get large weight (the overlap problem)", fontsize=10.4)
fig.tight_layout(); fig.savefig(HERE / "inverse_propensity.png", dpi=120); plt.close(fig); print("inverse_propensity.png")

# ---- (5) what the methods see: observed per-arm outcome vs true μ ----
rng = np.random.default_rng(0); d = dgp.generate(40000, rng)
obs = np.full((len(X), K), np.nan)
for i, xv in enumerate(X):
    m = np.isclose(d.X.ravel(), xv)
    for k in range(K):
        yk = d.Y[m & (d.T == k)]
        if len(yk) > 5: obs[i, k] = yk.mean()
fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.2), sharey=True)
for k, (a, col, ttl) in enumerate(zip(ax, (C0, C1), ("Arm 0 (control)", "Arm 1 (treatment)"))):
    a.plot(X, obs[:, k], "-o", color=col, ms=4, lw=2.3, label="OBSERVED mean")
    a.plot(X, mu[:, k], "--", color="#333", lw=1.8, label=r"TRUE $\mu_%d$" % k)
    a.set_xlabel("X"); a.set_ylim(-0.02, 1.04); a.set_title(ttl, fontsize=10.6); a.grid(alpha=.25); a.legend(fontsize=9, loc="lower right")
ax[0].set_ylabel("mean outcome")
fig.suptitle("What the methods see — observed treated outcomes are INFLATED above μ₁ (esp. at X<0, where treated are a high-S elite); this is the confounding R-OW must correct", fontsize=10.2)
fig.tight_layout(); fig.savefig(HERE / "whats_seen.png", dpi=120); plt.close(fig); print("whats_seen.png")
print("DGP plots done (γ=%.2f)." % G)
