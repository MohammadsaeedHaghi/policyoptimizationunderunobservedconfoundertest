"""End-to-end experiment runner — code 1.1.

Wires the whole pipeline for an experiment:  common (propensity + Hájek weights [+ outcome model])
  →  each METHOD (capped/uncapped, over a Γ-sweep × seeds)  →  EXTENSION (Shapley + KNN to the test set)
  →  SAVE per ``results/SAVING.md``.

It is driven by a single ``ExperimentConfig`` (common/config.py) that carries the three things defining the
run — the **geometry** (Wasserstein ground cost: z-score/metric), the **support** (discretisation: snap/mesh/
range/rounding), and the **methods** to compare — plus the experiment-level knobs (arms, capacity, Γ sweep,
seeds, sizes, the maximize convention).

DGP-agnostic: pass a ``generate(n, gamma, rng)`` returning an object with ``.X, .T, .Y, .Ypot, .mu``
(and optionally ``.S``). Method files live in hyphenated folders, so they are loaded by path
(importlib + sys.modules registration — required for their dataclasses to resolve).

Run the built-in synthetic demo with:  python run_experiment.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common
from common import persistence as P
from common.config import ExperimentConfig, WASSERSTEIN_METHODS, MAXIMIZE_METHODS, PARAMETRIC_METHODS

# extensions (loaded by path — hyphen-free here, but keep it uniform)
_EXT = {
    "shapley": _ROOT / "extensions" / "Shapley" / "shapley.py",
    "knn": _ROOT / "extensions" / "KNN" / "knn.py",
}


def _load(name: str, path: Path):
    """Load a module by file path and register it (so @dataclass + future-annotations resolve)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- method registry: name -> (file stem, fn stem, kind). kind drives the call signature. -----------
#   robust_w : (X,T,Y,w, n_arms, Gamma, [cap], support[, geom][, maximize]) — R-OW (no max), Hajek-OW (max+geom)
#   robust   : (X,T,Y,w, n_arms, Gamma, [cap], support[, maximize])         — R-O (no max),  Regret-O (max)
#   ipw      : (X,T,Y,w, n_arms,        [cap], support)                      — IPW (no Γ)
#   po       : (X,T,Y,   n_arms,        [cap], support)                      — Direct-X-X (no w, no Γ)
#   dr       : (X,T,Y,w, muhat, n_arms, Gamma, [cap], support[, geom])      — DoublyRobust, R-O/R-OW-DoublyRobust
# The geometry (zscore/metric) is passed ONLY to WASSERSTEIN_METHODS; maximize ONLY to MAXIMIZE_METHODS.
# (So R-OW-DoublyRobust = kind "dr" AND in WASSERSTEIN_METHODS ⇒ it receives muhat AND the geometry.)
REGISTRY = {
    "IPW-O-W":             ("ipw_o_w", "solve_ipw_o_w", "robust_w"),
    "IPW-O-X":             ("ipw_o_x", "solve_ipw_o_x", "robust"),
    "Hajek-O-W":          ("hajek_o_w", "solve_hajek_o_w", "robust_w"),
    "Hajek-O-X":          ("hajek_o_x", "solve_hajek_o_x", "robust"),
    "IPW-X-X":             ("ipw_x_x", "solve_ipw_x_x", "ipw"),
    "Direct-X-X": ("direct_x_x", "solve_direct_x_x", "po"),
    "DoublyRobust-X-X":    ("doublyrobust_x_x", "solve_doublyrobust_x_x", "dr"),   # plain AIPW — NO uncertainty set
    "DoublyRobust-O-W":    ("doublyrobust_o_w", "solve_doublyrobust_o_w", "dr"),   # DR objective + OW (box ∩ Wasserstein)
    "DoublyRobust-O-X":    ("doublyrobust_o_x", "solve_doublyrobust_o_x", "dr"),   # DR objective + O (box only)
}
_GAMMA_DEP = {"robust_w", "robust", "dr"}                     # methods whose policy depends on Γ

# Per-method objective semantics. The `objective_value` column is NOT comparable across these kinds
# (a worst-case value, a worst-case regret ≤0, a robust AIPW value and a clairvoyant value are different
# quantities) — `obj_kind` tags each row so analysis never mixes them. The sign/reward-vs-loss convention
# lives in config.json (`maximize`), separate from the quantity TYPE recorded here.
OBJ_KIND = {
    "IPW-O-W": "worst_case_value", "IPW-O-X": "worst_case_value",
    "Hajek-O-W": "worst_case_regret", "Hajek-O-X": "worst_case_regret",
    "IPW-X-X": "ipw_value", "Direct-X-X": "naive_value",
    "DoublyRobust-X-X": "aipw_value", "Kallus": "worst_case_regret",
    "DoublyRobust-O-W": "robust_aipw_value", "DoublyRobust-O-X": "robust_aipw_value",
}


def _realised(pi: np.ndarray, V: np.ndarray) -> float:
    """Average assigned value of a (K,n) policy on an (n,K) value matrix: (1/n) Σ_i Σ_k π_k(X_i) V_ik."""
    return float((pi * V.T).sum() / pi.shape[1])


def _support_kwargs(support) -> dict:
    """Map the SupportConfig onto the solver's discretisation arguments (its ``discretize`` bool = snap)."""
    return dict(discretize=support.snap, mesh=support.mesh,
                mesh_range=support.mesh_range, rounding_digits=support.rounding_digits)


def _call(fn, method, kind, *, X, T, Y, w, muhat, n_arms, Gamma, cap, support, geom, maximize):
    """Invoke a method's solver with the right arguments for its kind. cap=None ⇒ uncapped.

    Only the Wasserstein methods receive the geometry (zscore/metric); only the regret methods receive
    ``maximize`` — capability-gated so no solver is handed a kwarg it does not accept."""
    kw = dict(n_arms=n_arms, **_support_kwargs(support))
    if cap is not None:
        kw["cap"] = cap
    if method in MAXIMIZE_METHODS:
        kw["maximize"] = maximize
    if method in WASSERSTEIN_METHODS:
        kw.update(zscore=geom.zscore, metric=geom.metric)
    if kind in ("robust_w", "robust"):
        return fn(X, T, Y, w, Gamma=Gamma, **kw)
    if kind == "ipw":
        return fn(X, T, Y, w, **kw)
    if kind == "po":
        return fn(X, T, Y, **kw)
    if kind == "dr":
        return fn(X, T, Y, w, muhat, Gamma=Gamma, **kw)
    raise ValueError(f"unknown kind {kind}")


def run_experiment(config: ExperimentConfig, generate: Callable, *, results_root=None,
                   clean: bool = True) -> Path:
    """Run all (method × variant × Γ × seed), extend to test, and save everything under
    ``results/<experiment>/`` per SAVING.md. Returns the experiment directory.

    ``config`` is an ``ExperimentConfig``: ``config.geometry`` (Wasserstein ground cost), ``config.support``
    (discretisation) and ``config.methods`` (what to compare) are the three pieces that shape the run.

    ``clean`` (default True): prune STALE method outputs left by a previous run of the same experiment with
    a different method/variant/gamma_true set, so the results dir always reflects THIS config. Run-level
    artefacts (data/diagnostics/reference/config.json) are never touched. ``clean=False`` keeps orphans but
    warns about them."""
    cfg = config
    geom, sup = cfg.geometry, cfg.support
    n_arms, cap = cfg.n_arms, cfg.cap
    results_root = Path(results_root) if results_root else (_ROOT / "results")
    exp = P.experiment_dir(results_root, cfg.experiment)
    Gamma_grid = common.augmented_gamma_grid(cfg.gammas, cfg.gamma_true)   # sweep incl. matched Γ
    matched = common.matched_gamma(cfg.gamma_true)
    sh = _load("ext_shapley", _EXT["shapley"]); kn = _load("ext_knn", _EXT["knn"])

    # ---- 1. config / provenance (persist the EXACT config that produced these results) ----
    cfg.save_json(exp / "config.json")

    # ---- 1b. clean guard: drop (or warn about) stale method outputs from a prior, different config ----
    planned_leaves = [exp / spec.name / v / f"gtrue-{cfg.gamma_true:g}"
                      for spec in cfg.methods for v in spec.variants]   # Kallus's variant is ("Flat",)
    orphans = P.prune_experiment_dir(exp, planned_leaves, dry_run=not clean)
    if orphans:
        rel = ", ".join(str(o.relative_to(exp)) for o in orphans)
        print(f"  clean guard: {'removed' if clean else 'found (kept; clean=False)'} "
              f"{len(orphans)} stale dir(s): {rel}", flush=True)

    # ---- 2. per-seed raw draws + nuisances ----
    per_seed = {}
    for s in cfg.seeds:
        rng = np.random.default_rng(s)
        tr = generate(cfg.n_train, cfg.gamma_true, rng); te = generate(cfg.n_test, cfg.gamma_true, rng)
        w, _Pmat = common.ipw_weights_from_data(tr.X, tr.T, n_arms)        # estimated weights (never e_true)
        muhat = common.outcome_means(tr.X, tr.T, tr.Y, n_arms)            # outcome nuisance (for DR)
        per_seed[s] = dict(tr=tr, te=te, w=w, muhat=muhat)
        P.save_raw_draw(exp, s, train=dict(X=tr.X, T=tr.T, Y=tr.Y, Ypot=tr.Ypot, mu=tr.mu),
                        test=dict(X=te.X, T=te.T, Y=te.Y, Ypot=te.Ypot, mu=te.mu))
        # pre-algorithm diagnostics on the ESTIMATED weights (overlap / ESS) — catch a weight blow-up
        # before trusting any result. NEVER uses e_true (estimates ê from X,T). See common.diagnostics.
        diag = common.propensity_diagnostics(tr.X, tr.T, n_arms)
        P.save_json(exp / "diagnostics" / f"seed{s}.json", diag)
        if not diag["overlap_ok"]:
            print(f"  [warn] seed {s}: poor overlap — min propensity {diag['min_propensity']:.3g}, "
                  f"ESS {diag['ess']:.0f}/{cfg.n_train}; downstream results may be weight-driven.", flush=True)

    # ---- 3. reference ceilings (Full-info on Ypot, Best-true-means on μ) ----
    oc = _load("oracle_capped", _ROOT / "methods" / "Oracle" / "Capped" / "oracle_capped.py")
    ref = {}
    for s in cfg.seeds:
        d = per_seed[s]
        fi = oc.solve_oracle_capped(d["te"].X, d["te"].Ypot, n_arms=n_arms, cap=cap, **_support_kwargs(sup))
        bm = oc.solve_oracle_capped(d["te"].X, d["te"].mu, n_arms=n_arms, cap=cap, **_support_kwargs(sup))
        ref[f"full_info_realized_test__seed{s}"] = _realised(fi.pi, d["te"].Ypot)
        ref[f"best_means_exp_test__seed{s}"] = _realised(bm.pi, d["te"].mu)
    P.save_npz(exp / "reference" / "ceilings.npz", **{k: np.array(v) for k, v in ref.items()})

    # ---- 4. methods × variants (LP methods; Kallus is parametric and handled separately below) ----
    for spec in cfg.methods:
        method = spec.name
        if method in PARAMETRIC_METHODS:                              # Kallus: see step 5
            continue
        stem, fnstem, kind = REGISTRY[method]
        gdep = kind in _GAMMA_DEP
        for variant in spec.variants:
            csuf = variant.lower()                                        # 'capped'/'uncapped'
            cap_arg = cap if variant == "Capped" else None
            fpath = _ROOT / "methods" / method / variant / f"{stem}_{csuf}.py"
            fn = getattr(_load(f"{stem}_{csuf}", fpath), f"{fnstem}_{csuf}")
            mdir = P.method_dir(exp, method, variant, cfg.gamma_true)
            policy, extended, rows = {}, {}, []
            for s in cfg.seeds:
                d = per_seed[s]; tr, te, w, muhat = d["tr"], d["te"], d["w"], d["muhat"]
                grid = Gamma_grid if gdep else [matched]                  # Γ-indep ⇒ fit once (at matched)
                for G in grid:
                    res = _call(fn, method, kind, X=tr.X, T=tr.T, Y=tr.Y, w=w, muhat=muhat,
                                n_arms=n_arms, Gamma=G, cap=cap_arg, support=sup, geom=geom,
                                maximize=cfg.maximize)
                    sx = res.support_X
                    policy[P.gkey("pi", G, s)] = res.pi                    # on-support optimal policy
                    policy[P.gkey("usage", G, s)] = res.usage
                    for dn in ("beta", "mu", "nu"):                       # duals if the method has them
                        if hasattr(res, dn) and getattr(res, dn) is not None:
                            policy[P.gkey(dn, G, s)] = np.asarray(getattr(res, dn))
                    # deploy off-support: Shapley + KNN to train X and test X
                    for ext, mod in (("shapley", sh), ("knn", kn)):
                        fnm = getattr(mod, f"extend_with_{ext}_multiarm")
                        pi_tr = fnm(tr.X, sx, res.pi); pi_te = fnm(te.X, sx, res.pi)
                        extended[f"ext_{ext}__train__" + P.gkey("pi", G, s).split("__", 1)[1]] = pi_tr
                        extended[f"ext_{ext}__test__" + P.gkey("pi", G, s).split("__", 1)[1]] = pi_te
                        rows.append(dict(
                            seed=s, Gamma=round(float(G), 4), extension=ext,
                            realized_train=_realised(res.pi, tr.Ypot),           # raw on-support LP fit
                            realized_train_dep=_realised(pi_tr, tr.Ypot),        # deployed on train
                            realized_test=_realised(pi_te, te.Ypot),             # deployed on test
                            exp_train=_realised(pi_tr, tr.mu), exp_test=_realised(pi_te, te.mu),
                            objective_value=float(getattr(res, "objective_value", float("nan"))),
                            obj_kind=OBJ_KIND[method],

                            **{f"train_use_{k}": float(res.usage[k]) for k in range(n_arms)},
                            **{f"cap_{k}": (float(cap[k]) if cap_arg is not None else float("nan")) for k in range(n_arms)},
                        ))
                policy[f"support_X__seed{s}"] = sx
            P.save_npz(mdir / "policy.npz", **policy)
            P.save_npz(mdir / "extended.npz", **extended)
            P.write_csv(mdir / "outcomes.csv", rows)
            print(f"  saved {method}/{variant}  ({len(rows)} outcome rows)", flush=True)

    # ---- 5. Kallus (PARAMETRIC): fit θ, deploy via predict_kallus — no support/extensions, no capacity ----
    if any(m.name in PARAMETRIC_METHODS for m in cfg.methods):
        kal = _load("kallus_method", _ROOT / "methods" / "Kallus" / "kallus.py")
        mdir = P.method_dir(exp, "Kallus", "Flat", cfg.gamma_true)
        policy, extended, rows = {}, {}, []
        for s in cfg.seeds:
            d = per_seed[s]; tr, te, w = d["tr"], d["te"], d["w"]
            for G in Gamma_grid:                                      # Γ-dependent; odds set = faithful paper baseline
                res = kal.fit_kallus(tr.X, tr.T, tr.Y, w, n_arms=n_arms, Gamma=G, maximize=cfg.maximize)
                pi_tr = kal.predict_kallus(res.theta, tr.X, res.basis).T   # (K, n_tr) — the softmax IS the deploy
                pi_te = kal.predict_kallus(res.theta, te.X, res.basis).T   # (K, n_te)
                usage = pi_tr.mean(axis=1)
                policy[P.gkey("theta", G, s)] = res.theta             # save θ (parametric — not an on-support π)
                policy[P.gkey("usage", G, s)] = usage
                extended[P.gkey("pi_train", G, s)] = pi_tr            # predict_kallus deployment (no Shapley/KNN)
                extended[P.gkey("pi_test", G, s)] = pi_te
                rows.append(dict(
                    seed=s, Gamma=round(float(G), 4), extension="parametric",
                    realized_train=_realised(pi_tr, tr.Ypot), realized_train_dep=_realised(pi_tr, tr.Ypot),
                    realized_test=_realised(pi_te, te.Ypot),
                    exp_train=_realised(pi_tr, tr.mu), exp_test=_realised(pi_te, te.mu),
                    objective_value=float(res.objective_value), obj_kind=OBJ_KIND["Kallus"],
                    **{f"train_use_{k}": float(usage[k]) for k in range(n_arms)},
                    **{f"cap_{k}": float("nan") for k in range(n_arms)},
                ))
        P.save_npz(mdir / "policy.npz", **policy)
        P.save_npz(mdir / "extended.npz", **extended)
        P.write_csv(mdir / "outcomes.csv", rows)
        print(f"  saved Kallus/Flat  ({len(rows)} outcome rows)", flush=True)

    print(f"DONE -> {exp}", flush=True)
    return exp


# --------------------------------------------------------------------------- built-in synthetic demo
def _synthetic_dgp(n: int, gamma: float, rng: np.random.Generator):
    """Tiny self-contained binary-treatment DGP (2-D X, unobserved S) — for the runner smoke/demo only."""
    from types import SimpleNamespace
    X = rng.uniform(-1, 1, size=(n, 2))
    S = (rng.uniform(size=n) < 0.5).astype(float)
    proj = X[:, 0] + 0.6 * X[:, 1]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    p0 = sig(0.3 + 0.0 * proj)                                # control: flat, clean
    p1 = sig(-0.5 + 1.5 * proj + 2.0 * (S - 0.5))            # treatment: risky + S-confounded
    Ypot = np.column_stack([(rng.uniform(size=n) < p0).astype(float), (rng.uniform(size=n) < p1).astype(float)])
    # S-marginal true means μ_k(X) (no S): average the two S branches
    mu0 = sig(0.3 + 0.0 * proj)
    mu1 = 0.5 * sig(-0.5 + 1.5 * proj - 1.0) + 0.5 * sig(-0.5 + 1.5 * proj + 1.0)
    mu = np.column_stack([mu0, mu1])
    T = (rng.uniform(size=n) < sig(0.5 * proj + gamma * (S - 0.5))).astype(int)   # mis-targeted + S channel
    Y = Ypot[np.arange(n), T]
    return SimpleNamespace(X=X, S=S, T=T, Ypot=Ypot, Y=Y, mu=mu, n_arms=2)


if __name__ == "__main__":
    from common.config import GeometryConfig, SupportConfig, MethodSpec

    cfg = ExperimentConfig(
        experiment="demo-synthetic-K2",
        n_arms=2, cap=(1.0, 0.40), gamma_true=3.0, gammas=(1.0, 3.0, 7.0), seeds=(9000, 9001),
        n_train=200, n_test=400, maximize=True,
        geometry=GeometryConfig(zscore=True, metric="euclidean"),     # scale-fair Wasserstein ground cost
        support=SupportConfig(snap=True, mesh=5, mesh_range=(-1.0, 1.0)),
        methods=["IPW-O-W", "IPW-O-X", "IPW-X-X", "Kallus",                       # what to compare (Kallus = parametric baseline)
                 MethodSpec("DoublyRobust-X-X", variants=("Capped",))],
    )
    run_experiment(cfg, _synthetic_dgp)
