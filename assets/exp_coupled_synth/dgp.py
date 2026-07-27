"""exp_coupled_synth: OUR coupled-confounder synthetic experiment (self-contained DGP).

The paper's second synthetic experiment, complementing the hidden-vitality showcase
(over-treatment, monotone CATE) with the opposite failure direction (under-treatment), an
oscillating heterogeneous effect, and two designed properties:

  KNOWN GAMMA* (the must-keep property). The propensity is logistic in the hidden confounder,
      P(T=1 | x, U) = sigma(0.5 + 1.5 x + 0.8 (2U-1)),
  so the odds ratio between the U-arms is e^{2*0.8} = 4.95 at EVERY x -- by algebra, not
  calibration (the pipeline's clip at [.02,.98] never binds on x in [-1,1]). The fitted
  marginal propensity always lies between the two U-extremals, hence every unit's true
  inverse weight is inside the Gamma = 4.95 odds-box around it: methods run at the SINGLE
  matched Gamma = Gamma* -- no sweep, no tuning knob.

  COUPLING DIAL. U | x ~ Bern(sigma(BETA x)). BETA moves the experiment along the measured
  coupling axis (corr(x, S): -0.01 / 0.58 / 0.76 / 0.84 at BETA = 0 / 2.5 / 5 / 10) with the
  marginal P(U=1) = 1/2 preserved for every BETA by symmetry.

Outcome functional forms follow Kallus-Mao-Zhou (2019) (with X = 2x on [-2, 2]):
  Y(a) = (2a-1)X + (2a-1) - 2 sin(2(2a-1)X) - 2(2U-1)(1 + 0.5X) + N(0,1)
  CATE(X) = 2X + 2 - 4 sin(2X)   (U shifts levels only and cancels in the effect)
The confounder makes the treated arm look WORSE (U=1 lowers outcomes and raises treatment
probability), so naive methods under-treat.

BETA is overridden by the dgp_beta*.py wrappers. Quick-pilot results: coupled_beta*.json
(BETA = 10 ran with this file's default).
"""
import numpy as np

GSTAR = 4.95
BETA = 10.0
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid contract only)
K = 2
CAP = (1.0, 0.3)
NOISE = 1.0
_C = np.log(GSTAR)                                 # u-to-u log-odds


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

def e_nom(x): return _sig(1.5 * np.asarray(x, float) + 0.5)

def p_s1(x): return _sig(BETA * np.asarray(x, float))

def propensity(x, S):
    l = 1.5 * np.asarray(x, float) + 0.5
    return np.clip(_sig(l + 0.5 * _C * np.asarray(S, float)), 0.02, 0.98)

def _mu(x, a, S):
    X = 2.0 * np.asarray(x, float); s = 2.0 * a - 1.0
    return s * X + s - 2.0 * np.sin(2.0 * s * X) - 2.0 * np.asarray(S, float) * (1.0 + 0.5 * X)

def mu0(x, S): return _mu(x, 0, S)
def mu1(x, S): return _mu(x, 1, S)

def cate(x):
    X = 2.0 * np.asarray(x, float)
    return 2.0 * X + 2.0 - 4.0 * np.sin(2.0 * X)

def oracle_policy(x): return (cate(x) > 0).astype(float)


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    S = np.where(rng.uniform(size=n) < p_s1(x), 1.0, -1.0)
    e = propensity(x, S)
    T = (rng.uniform(size=n) < e).astype(int)
    Y0 = mu0(x, S) + rng.normal(0, NOISE, size=n)
    Y1 = mu1(x, S) + rng.normal(0, NOISE, size=n)
    Y = np.where(T == 1, Y1, Y0)
    observed = {"X": x.reshape(-1, 1), "T": T, "Y": Y}
    full = {**observed, "S": S, "Y1": Y1, "Y0": Y0, "e_true": e}
    return observed, full


def grid_truth():
    X = LEVELS
    return dict(X=X, cate=cate(X), oracle=oracle_policy(X))
