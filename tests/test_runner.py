"""run_experiment end-to-end (needs Gurobi): saved tree + obj_kind + Kallus + diagnostics, and the
clean guard (orphans pruned when clean=True, kept+warned when clean=False)."""
import csv
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gurobipy")

import run_experiment as R
from common.config import ExperimentConfig


def _dgp(n, gamma, rng):
    X = rng.uniform(-1, 1, size=(n, 2))
    S = (rng.uniform(size=n) < 0.5).astype(float)
    proj = X[:, 0]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    Ypot = np.column_stack([(rng.uniform(size=n) < sig(0.3 + 0 * proj)).astype(float),
                            (rng.uniform(size=n) < sig(0.5 * proj + (S - 0.5))).astype(float)])
    mu = np.column_stack([sig(0.3 + 0 * proj), sig(0.5 * proj)])
    T = (rng.uniform(size=n) < sig(0.4 * proj + gamma * (S - 0.5))).astype(int)
    Y = Ypot[np.arange(n), T]
    return SimpleNamespace(X=X, T=T, Y=Y, Ypot=Ypot, mu=mu, n_arms=2)


def _cfg(methods, exp="pytest-run"):
    return ExperimentConfig(experiment=exp, n_arms=2, cap=(1.0, 0.5), gamma_true=3.0,
                            gammas=(3.0,), seeds=(0,), n_train=60, n_test=80, methods=methods)


def test_end_to_end_saves_tree(tmp_path):
    exp = R.run_experiment(_cfg(["IPW-X-X", "Kallus"]), _dgp, results_root=tmp_path)
    assert (exp / "config.json").exists()
    assert list((exp / "diagnostics").glob("seed*.json"))
    assert (exp / "IPW-X-X" / "Capped" / "gtrue-3" / "outcomes.csv").exists()
    assert (exp / "Kallus" / "Flat" / "gtrue-3" / "outcomes.csv").exists()
    row = next(csv.DictReader(open(exp / "IPW-X-X" / "Capped" / "gtrue-3" / "outcomes.csv")))
    assert "objective_value" in row and "obj_kind" in row and "obj_ipw" not in row


def test_clean_guard_prunes_orphans(tmp_path):
    R.run_experiment(_cfg(["IPW-X-X", "Direct-X-X"]), _dgp, results_root=tmp_path)
    exp = tmp_path / "pytest-run"
    assert (exp / "Direct-X-X").exists()
    R.run_experiment(_cfg(["IPW-X-X"]), _dgp, results_root=tmp_path, clean=True)
    assert not (exp / "Direct-X-X").exists()         # orphan pruned
    assert (exp / "IPW-X-X").exists()                             # planned kept


def test_clean_false_keeps_orphans(tmp_path):
    R.run_experiment(_cfg(["IPW-X-X", "Direct-X-X"]), _dgp, results_root=tmp_path)
    exp = tmp_path / "pytest-run"
    R.run_experiment(_cfg(["IPW-X-X"]), _dgp, results_root=tmp_path, clean=False)
    assert (exp / "Direct-X-X").exists()             # kept (only warned)
