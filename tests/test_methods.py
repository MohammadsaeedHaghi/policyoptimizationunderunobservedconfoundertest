"""Method solvers (needs Gurobi): valid simplex / finite obj / capacity respected, for K=2 AND K=3,
plus the maximize=False↔negate-Y identity and the discretize=False (no-grid) path."""
import numpy as np
import pytest

pytest.importorskip("gurobipy")

LP_METHODS = ["IPW-O-W", "IPW-O-X", "Hajek-O-W", "Hajek-O-X", "IPW-X-X", "Direct-X-X", "DoublyRobust-X-X"]
WASS = {"IPW-O-W", "Hajek-O-W"}
GAMMA = {"IPW-O-W", "IPW-O-X", "Hajek-O-W", "Hajek-O-X", "DoublyRobust-X-X"}
MAXIMIZE = {"Hajek-O-W", "Hajek-O-X"}


def _call(loader, method, variant, d, *, discretize=True, maximize=False):
    fn = loader(method, variant)
    kw = dict(n_arms=d["K"], discretize=discretize, mesh=5)
    if variant == "Capped":
        kw["cap"] = d["cap"]
    if method in GAMMA:
        kw["Gamma"] = d["Gamma"]
    if method in MAXIMIZE:
        kw["maximize"] = maximize
    if method == "Direct-X-X":
        return fn(d["X"], d["T"], d["Y"], **kw)
    if method == "DoublyRobust-X-X":
        return fn(d["X"], d["T"], d["Y"], d["w"], d["muhat"], **kw)
    return fn(d["X"], d["T"], d["Y"], d["w"], **kw)


def _assert_valid(res, K, n, cap=None):
    pi = np.asarray(res.pi)
    assert pi.shape == (K, n)
    assert np.all(pi >= -1e-7) and np.all(pi <= 1 + 1e-7)
    assert np.allclose(pi.sum(axis=0), 1.0, atol=1e-6)
    assert np.isfinite(res.objective_value)
    if cap is not None:
        assert np.all(np.asarray(res.usage) <= np.array(cap) + 1e-6)


@pytest.mark.parametrize("method", LP_METHODS)
@pytest.mark.parametrize("variant", ["Capped", "Uncapped"])
def test_lp_methods_k2(loader, k2, method, variant):
    res = _call(loader, method, variant, k2)
    _assert_valid(res, k2["K"], len(k2["T"]), k2["cap"] if variant == "Capped" else None)


@pytest.mark.parametrize("method", LP_METHODS)
def test_lp_methods_k3(loader, k3, method):
    res = _call(loader, method, "Capped", k3)
    _assert_valid(res, k3["K"], len(k3["T"]), k3["cap"])


@pytest.mark.parametrize("method", sorted(MAXIMIZE))
def test_maximize_false_equals_negate_y(loader, k2, method):
    fn = loader(method, "Capped")
    kw = dict(n_arms=k2["K"], Gamma=k2["Gamma"], cap=k2["cap"], discretize=True, mesh=5)
    a = fn(k2["X"], k2["T"], k2["Y"], k2["w"], maximize=False, **kw)
    b = fn(k2["X"], k2["T"], -k2["Y"], k2["w"], maximize=True, **kw)
    assert np.max(np.abs(np.asarray(a.pi) - np.asarray(b.pi))) < 1e-6
    assert abs(a.objective_value - b.objective_value) < 1e-6


def test_discretize_false_runs(loader, k2):
    res = _call(loader, "IPW-O-W", "Capped", k2, discretize=False)
    _assert_valid(res, k2["K"], len(k2["T"]), k2["cap"])


def test_oracle(loader, k2):
    fn = loader("Oracle", "Capped")
    res = fn(k2["X"], k2["Ypot"], n_arms=k2["K"], cap=k2["cap"], discretize=True, mesh=5)
    _assert_valid(res, k2["K"], len(k2["T"]), k2["cap"])


def test_kallus_parametric(mod_loader, k2):
    kal = mod_loader("kallus_t", "methods/Kallus/kallus.py")
    res = kal.fit_kallus(k2["X"], k2["T"], k2["Y"], k2["w"], n_arms=k2["K"], Gamma=k2["Gamma"],
                         maximize=False, n_iters=6, n_restarts=1)
    pol = kal.predict_kallus(res.theta, k2["X"], res.basis)
    assert pol.shape == (len(k2["T"]), k2["K"])
    assert np.allclose(pol.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(res.objective_value)
