"""DGP engine for the Policy Lab app — pure logic, NO Streamlit imports.

Turns a structured ``DGPSpec`` (or a raw ``generate`` code string) into a ``generate(n, gamma, rng)``
callable compatible with ``run_experiment`` — i.e. one returning an object with ``.X .S .T .Y .Ypot .mu``
and ``.n_arms``.

The structured model (the "select a DGP" page):
  • X ~ a chosen distribution on R^d         (uniform / normal)
  • S ~ a chosen scalar distribution         (bernoulli / uniform / normal)   — the UNOBSERVED confounder
  • outcomes   P(Y=1 | X, S, T=k) = σ(a_k·X + b_k·S + c_k)         (per-arm logistic)
  • treatment  P(T=k | X, S)      = softmax_k( v_k·X + γ·d_k·(S − E[S]) )    (S confounds, strength = run γ)

The realised potential outcomes use each unit's own (unobserved) S; the deployable true mean μ_k(X)
marginalises S out — exactly the project's convention (Ypot realised-with-S, μ = S-marginal mean).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Dict, List

import numpy as np

# ----------------------------------------------------------------------------- distribution specs
_X_DISTS = ("uniform", "normal")
_S_DISTS = ("bernoulli", "uniform", "normal")


@dataclass
class DistSpec:
    """A covariate distribution. ``dim`` applies to X (S is always scalar, dim 1)."""
    name: str
    params: Dict[str, float]
    dim: int = 1

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` rows. X → (n, dim); S → (n,)."""
        p = self.params
        if self.name == "uniform":
            out = rng.uniform(p.get("low", -1.0), p.get("high", 1.0), size=(n, self.dim))
        elif self.name == "normal":
            out = rng.normal(p.get("mean", 0.0), p.get("std", 1.0), size=(n, self.dim))
        elif self.name == "bernoulli":
            out = (rng.uniform(size=(n, self.dim)) < p.get("p", 0.5)).astype(float)
        else:
            raise ValueError(f"unknown distribution {self.name!r}")
        return out if self.dim > 1 else out.ravel()

    def mean(self) -> float:
        """E[S] (scalar dists only) — the centring used so confounding is mean-zero in S."""
        p = self.params
        if self.name == "bernoulli":
            return p.get("p", 0.5)
        if self.name == "uniform":
            return 0.5 * (p.get("low", -1.0) + p.get("high", 1.0))
        if self.name == "normal":
            return p.get("mean", 0.0)
        raise ValueError(self.name)

    def support_points(self):
        """Discrete support + probabilities for exact S-marginalisation, or None (continuous → MC)."""
        if self.name == "bernoulli":
            p = self.params.get("p", 0.5)
            return np.array([0.0, 1.0]), np.array([1.0 - p, p])
        return None


# ----------------------------------------------------------------------------- structural models
@dataclass
class OutcomeModel:
    """Per-arm logistic success model  σ(a_k·X + b_k·S + c_k).  a:(K,d)  b:(K,)  c:(K,)."""
    a: List[List[float]]
    b: List[float]
    c: List[float]


@dataclass
class TreatmentModel:
    """Per-arm treatment logits  v_k·X + γ·d_k·(S − E[S]).  v:(K,d)  d_s:(K,).  (γ = run confounding.)"""
    v: List[List[float]]
    d_s: List[float]


@dataclass
class DGPSpec:
    """The full DGP specification edited on the DGP page."""
    n_arms: int
    x: DistSpec
    s: DistSpec
    outcome: OutcomeModel
    treatment: TreatmentModel
    mode: str = "structured"             # "structured" | "code"
    code: str = ""                       # raw generate() source when mode == "code"
    s_marginal_samples: int = 256        # MC draws for μ when S is continuous

    # ---- JSON round-trip (so a DGP can be saved with the run) ----
    def to_dict(self) -> dict:
        return dict(
            n_arms=self.n_arms, mode=self.mode, code=self.code,
            s_marginal_samples=self.s_marginal_samples,
            x=dict(name=self.x.name, params=self.x.params, dim=self.x.dim),
            s=dict(name=self.s.name, params=self.s.params, dim=1),
            outcome=dict(a=self.outcome.a, b=self.outcome.b, c=self.outcome.c),
            treatment=dict(v=self.treatment.v, d_s=self.treatment.d_s),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "DGPSpec":
        return cls(
            n_arms=int(d["n_arms"]), mode=d.get("mode", "structured"), code=d.get("code", ""),
            s_marginal_samples=int(d.get("s_marginal_samples", 256)),
            x=DistSpec(d["x"]["name"], dict(d["x"]["params"]), int(d["x"].get("dim", 1))),
            s=DistSpec(d["s"]["name"], dict(d["s"]["params"]), 1),
            outcome=OutcomeModel(d["outcome"]["a"], d["outcome"]["b"], d["outcome"]["c"]),
            treatment=TreatmentModel(d["treatment"]["v"], d["treatment"]["d_s"]),
        )


# ----------------------------------------------------------------------------- helpers
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise categorical sample from a (n, K) probability matrix → (n,) int arms."""
    c = np.cumsum(probs, axis=1)
    u = rng.uniform(size=(probs.shape[0], 1))
    return (u > c).sum(axis=1).astype(int)


# ----------------------------------------------------------------------------- structured builder
def build_generate(spec: DGPSpec) -> Callable:
    """Compile a ``DGPSpec`` into ``generate(n, gamma, rng)``. Routes to the code path if mode=='code'."""
    if spec.mode == "code":
        return build_generate_from_code(spec.code)

    K, d = int(spec.n_arms), int(spec.x.dim)
    a = np.asarray(spec.outcome.a, float).reshape(K, d)
    b = np.asarray(spec.outcome.b, float).reshape(K)
    c = np.asarray(spec.outcome.c, float).reshape(K)
    v = np.asarray(spec.treatment.v, float).reshape(K, d)
    d_s = np.asarray(spec.treatment.d_s, float).reshape(K)
    s_center = spec.s.mean()
    sx, sp = (spec.s.support_points() or (None, None))      # exact S-marginal if discrete

    def outcome_p(X, S):                                    # (n,K) success probs given realised S
        return _sigmoid(X @ a.T + S[:, None] * b[None, :] + c[None, :])

    def mu_marginal(X, rng):                                # (n,K) μ_k(X) = E_S[ σ(...) ]
        if sx is not None:                                  # discrete S → exact mixture
            acc = np.zeros((X.shape[0], K))
            for s_val, w in zip(sx, sp):
                acc += w * _sigmoid(X @ a.T + s_val * b[None, :] + c[None, :])
            return acc
        S_mc = spec.s.sample(spec.s_marginal_samples, rng)  # continuous S → Monte-Carlo average
        acc = np.zeros((X.shape[0], K))
        for s_val in S_mc:
            acc += _sigmoid(X @ a.T + s_val * b[None, :] + c[None, :])
        return acc / len(S_mc)

    def generate(n: int, gamma: float, rng: np.random.Generator):
        X = np.atleast_2d(spec.x.sample(n, rng))
        if X.shape[0] != n:                                 # dim==1 came back as (n,) → (n,1)
            X = X.reshape(n, -1)
        S = spec.s.sample(n, rng)
        Ypot = (rng.uniform(size=(n, K)) < outcome_p(X, S)).astype(float)
        t_logits = X @ v.T + gamma * (S[:, None] - s_center) * d_s[None, :]
        T = _categorical(_softmax(t_logits), rng)
        Y = Ypot[np.arange(n), T]
        mu = mu_marginal(X, np.random.default_rng(0))       # μ deterministic in X (fixed MC seed)
        return SimpleNamespace(X=X, S=S, T=T, Y=Y, Ypot=Ypot, mu=mu, n_arms=K)

    return generate


# ----------------------------------------------------------------------------- raw-code builder
_CODE_TEMPLATE = '''\
def generate(n, gamma, rng):
    """Return X (n,d), T (n,), Y (n,), Ypot (n,K), mu (n,K)  — or a SimpleNamespace with those."""
    import numpy as np
    K = 2
    X = rng.uniform(-1, 1, size=(n, 2))
    S = (rng.uniform(size=n) < 0.5).astype(float)
    sig = lambda z: 1 / (1 + np.exp(-z))
    p = np.column_stack([sig(0.3 + 0*X[:,0]), sig(0.5*X[:,0] + (S - 0.5))])
    Ypot = (rng.uniform(size=(n, K)) < p).astype(float)
    mu = np.column_stack([sig(0.3 + 0*X[:,0]), 0.5*sig(0.5*X[:,0]-0.5) + 0.5*sig(0.5*X[:,0]+0.5)])
    T = (rng.uniform(size=n) < sig(0.4*X[:,0] + gamma*(S - 0.5))).astype(int)
    Y = Ypot[np.arange(n), T]
    return X, T, Y, Ypot, mu
'''


def build_generate_from_code(code: str) -> Callable:
    """Exec user code that defines ``generate(n, gamma, rng)`` and wrap/normalise its output.

    The code runs with numpy + SimpleNamespace available. ``generate`` may return a SimpleNamespace
    (with .X .T .Y .Ypot .mu [.S]) or a tuple ``(X, T, Y, Ypot, mu)``; the wrapper validates shapes and
    always returns a normalised SimpleNamespace. (Local research tool — code is trusted, not sandboxed.)
    """
    ns: dict = {"np": np, "numpy": np, "SimpleNamespace": SimpleNamespace, "__builtins__": __builtins__}
    exec(compile(code or _CODE_TEMPLATE, "<dgp-code>", "exec"), ns)
    if "generate" not in ns or not callable(ns["generate"]):
        raise ValueError("your code must define a function  generate(n, gamma, rng)")
    user_generate = ns["generate"]

    def generate(n: int, gamma: float, rng: np.random.Generator):
        out = user_generate(n, gamma, rng)
        if isinstance(out, SimpleNamespace):
            X, T, Y, Ypot, mu = out.X, out.T, out.Y, out.Ypot, out.mu
            S = getattr(out, "S", None)
        else:
            X, T, Y, Ypot, mu = out
            S = None
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[0] != n:
            X = X.reshape(n, -1)
        T = np.asarray(T).astype(int).ravel()
        Y = np.asarray(Y, float).ravel()
        Ypot = np.asarray(Ypot, float)
        mu = np.asarray(mu, float)
        K = Ypot.shape[1]
        for nm, arr, shape in [("Ypot", Ypot, (n, K)), ("mu", mu, (n, K)), ("T", T, (n,)), ("Y", Y, (n,))]:
            if arr.shape != shape:
                raise ValueError(f"generate() returned {nm} with shape {arr.shape}, expected {shape}")
        return SimpleNamespace(X=X, S=S, T=T, Y=Y, Ypot=Ypot, mu=mu, n_arms=K)

    return generate


# ----------------------------------------------------------------------------- preview + defaults
def sample_preview(spec: DGPSpec, *, n: int = 400, gamma: float = 3.0, seed: int = 0) -> dict:
    """Draw one sample for the DGP page's live preview. Returns arrays + quick summaries."""
    gen = build_generate(spec)
    d = gen(n, gamma, np.random.default_rng(seed))
    K = d.n_arms
    arm_counts = np.bincount(d.T, minlength=K).tolist()
    return dict(
        X=d.X, S=d.S, T=d.T, Y=d.Y, Ypot=d.Ypot, mu=d.mu, n_arms=K,
        arm_counts=arm_counts,
        y_rate=float(d.Y.mean()),
        mu_mean=d.mu.mean(axis=0).tolist(),
        all_arms_present=all(c > 0 for c in arm_counts),
    )


def default_spec(n_arms: int = 2, x_dim: int = 2) -> DGPSpec:
    """A sensible starting DGP: clean flat control (arm 0) + risky S-confounded treatment arms."""
    K, d = n_arms, x_dim
    a = [[0.0] * d] + [[0.8] + [0.0] * (d - 1) for _ in range(K - 1)]       # arm0 flat; others slope in x0
    b = [0.0] + [1.0] * (K - 1)                                            # S boosts treatment outcomes
    c = [0.3] + [-0.4 * k for k in range(1, K)]                            # arm0 decent; others riskier
    v = [[0.0] * d] + [[1.0] + [0.0] * (d - 1) for _ in range(K - 1)]      # control unconfounded
    d_s = [0.0] + [2.0] * (K - 1)                                          # S drives non-control treatment
    return DGPSpec(
        n_arms=K,
        x=DistSpec("uniform", {"low": -1.0, "high": 1.0}, dim=d),
        s=DistSpec("bernoulli", {"p": 0.5}, dim=1),
        outcome=OutcomeModel(a=a, b=b, c=c),
        treatment=TreatmentModel(v=v, d_s=d_s),
    )
