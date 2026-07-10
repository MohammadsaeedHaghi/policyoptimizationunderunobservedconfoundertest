"""IDEA 04 - Effect-modification SIGN FLIP (confounding by indication, a Simpson reversal).

Story: a hidden frailty F in {0,1}; frail patients (F=1, more common at low fitness X, P(F=1|X)=sigma(-5X))
have worse baseline outcomes AND are HARMED by the aggressive treatment, while robust patients (F=0) are
HELPED. Doctors treat the frail more (confounding by indication: e=sigma(1.2 F - X)). So among the treated
(mostly frail) outcomes look worse -> the naive within-X contrast points the WRONG way (Simpson reversal).
The truth: treat the robust (high X). A method must undo the selection on hidden F to recover this.

  F in {0,1}, P(F=1|X)=sigma(-5X);  e=clip(sigma(1.2 F - X));
  mu0 = H F (frail worse, H<0);  mu1 = mu0 + (F? -BH : +BR).  CATE = BR(1-pF) - BH pF.
"""
import numpy as np
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
CAP = (1.0, 0.5); K = 2; NOISE = 0.6
_Z = np.linspace(-5.0, 5.0, 121); _W = np.exp(-_Z**2 / 2); _W = _W / _W.sum()
def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

GAM, H, BR, BH, C, D = 5.0, -3.0, 2.0, 2.0, 1.2, 1.0

def _components(x):
    x = np.asarray(x, float); pf = _sig(-GAM * x)
    Fv = np.array([0.0, 1.0]); w = np.stack([1 - pf, pf], axis=1)
    Fm = np.broadcast_to(Fv, (x.size, 2)); xb = x[:, None]
    e = np.clip(_sig(C * Fm - D * xb), 0.02, 0.98)
    m0 = H * Fm + 0.0 * xb; eff = np.where(Fm > 0.5, -BH, BR)
    return w, e, m0, m0 + eff

def _sample(X, rng):
    X = np.asarray(X, float); pf = _sig(-GAM * X)
    F = (rng.uniform(size=X.size) < pf).astype(float)
    e = np.clip(_sig(C * F - D * X), 0.02, 0.98)
    m0 = H * F; eff = np.where(F > 0.5, -BH, BR)
    return F, e, m0, m0 + eff

# ---- generic interface (identical across all idea modules) ----
def _eY(x):
    w, e, m0, m1 = _components(np.asarray(x, float)); return (w * m0).sum(1), (w * m1).sum(1)
def _obs(x):
    w, e, m0, m1 = _components(np.asarray(x, float)); s1 = (w * e).sum(1); s0 = (w * (1 - e)).sum(1)
    return (w * e * m1).sum(1) / s1, (w * (1 - e) * m0).sum(1) / s0, (w * e).sum(1)
def grid_truth():
    X = LEVELS; eY0, eY1 = _eY(X); o1, o0, em = _obs(X); cate = eY1 - eY0
    return dict(X=X, eY0=eY0, eY1=eY1, cate=cate, oracle=(cate > 0).astype(float),
                obs1=o1, obs0=o0, obs_contrast=o1 - o0, e_marg=em, naive=((o1 - o0) > 0).astype(float))
def oracle_policy(X):
    eY0, eY1 = _eY(X); return (eY1 > eY0).astype(float)
def naive_policy_grid(): return grid_truth()["naive"]
def exact_value(pi_grid):
    g = grid_truth(); pi = np.asarray(pi_grid, float); return float(np.mean(pi * g["eY1"] + (1 - pi) * g["eY0"]))
def generate(n, seed=0):
    rng = np.random.default_rng(seed); X = rng.choice(LEVELS, size=n)
    hidden, e, mu0, mu1 = _sample(X, rng); T = (rng.uniform(size=n) < e).astype(int)
    Y0 = mu0 + rng.normal(0, NOISE, n); Y1 = mu1 + rng.normal(0, NOISE, n); Y = np.where(T == 1, Y1, Y0)
    obs = {"X": X.reshape(-1, 1), "T": T, "Y": Y}
    full = {**obs, "U": np.asarray(hidden, float), "Y0": Y0, "Y1": Y1, "e_true": e, "mu0": mu0, "mu1": mu1}
    return obs, full
