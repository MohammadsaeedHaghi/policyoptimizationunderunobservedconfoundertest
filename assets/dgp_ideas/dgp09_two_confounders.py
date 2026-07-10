"""IDEA 09 - TWO correlated hidden confounders (separate selection and prognosis drivers).

Story: selection is driven by one latent variable U1 (e.g. a hidden "eagerness/access" factor), while
outcomes are driven by a different latent U2 (a hidden prognosis), and the two are correlated (corr R).
Both are correlated with X. So no single observed proxy explains assignment, yet because U2 (prognosis)
tracks X, covariate balancing still helps. This is a richer, more realistic confounding structure than a
single U. U2 enters both arms with slightly different coefficients (non-cancelling).

  Z1,Z2 ~ N(0,I);  U1 = C1 X + Z1,  U2 = C2 X + R Z1 + sqrt(1-R^2) Z2.
  e=clip(sigma(A U1 - D X));  mu0 = M0 X + L0 U2,  mu1 = M1 X + L1 U2.
  E[CATE|X] = (M1-M0) X + (L1-L0) C2 X.
"""
import numpy as np
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
CAP = (1.0, 0.5); K = 2; NOISE = 0.6
_Z = np.linspace(-4.0, 4.0, 41); _W = np.exp(-_Z**2 / 2); _W = _W / _W.sum()
def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

C1, C2, R, A, D, L0, L1, M0, M1 = 0.8, 0.8, 0.7, 1.0, 1.0, 4.0, 4.4, 0.0, 0.3
_Z1, _Z2 = np.meshgrid(_Z, _Z, indexing="ij")
_z1, _z2 = _Z1.ravel(), _Z2.ravel()
_ww = (_W[:, None] * _W[None, :]).ravel()
_rt = np.sqrt(1.0 - R * R)

def _components(x):
    x = np.asarray(x, float); xb = x[:, None]; w = np.tile(_ww, (x.size, 1))
    U1 = C1 * xb + _z1[None, :]; U2 = C2 * xb + R * _z1[None, :] + _rt * _z2[None, :]
    e = np.clip(_sig(A * U1 - D * xb), 0.02, 0.98)
    return w, e, M0 * xb + L0 * U2, M1 * xb + L1 * U2

def _sample(X, rng):
    X = np.asarray(X, float); z1 = rng.normal(0, 1, X.size); z2 = rng.normal(0, 1, X.size)
    U1 = C1 * X + z1; U2 = C2 * X + R * z1 + _rt * z2
    e = np.clip(_sig(A * U1 - D * X), 0.02, 0.98)
    return U2, e, M0 * X + L0 * U2, M1 * X + L1 * U2

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
