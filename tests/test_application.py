"""Tests for the Policy Lab app's pure core (no Streamlit needed): DGP builder + experiment adapter."""
import sys
from pathlib import Path

import numpy as np
import pytest

_APP = Path(__file__).resolve().parents[1] / "application"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from core.dgp import default_spec, build_generate, build_generate_from_code, sample_preview, DGPSpec


def test_structured_generate_shapes():
    gen = build_generate(default_spec(3, 2))
    d = gen(60, 3.0, np.random.default_rng(0))
    assert d.X.shape == (60, 2)
    assert d.Ypot.shape == (60, 3) and d.mu.shape == (60, 3)
    assert d.T.shape == (60,) and set(np.unique(d.T)) <= {0, 1, 2}
    assert np.all((d.mu >= 0) & (d.mu <= 1))               # μ are probabilities
    assert set(np.unique(d.Y)) <= {0.0, 1.0}               # binary outcome


def test_spec_json_roundtrip():
    spec = default_spec(3, 2)
    assert DGPSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_code_mode_template_runs():
    gen = build_generate_from_code("")                     # empty → built-in template
    d = gen(40, 2.0, np.random.default_rng(1))
    assert d.Ypot.shape[0] == 40 and d.n_arms == 2


def test_code_mode_bad_shape_raises():
    bad = "def generate(n, gamma, rng):\n    return rng.uniform(size=(n,2)), [0]*n, [0]*n, [[0,0]], [[0,0]]\n"
    gen = build_generate_from_code(bad)
    with pytest.raises(ValueError):
        gen(10, 1.0, np.random.default_rng(0))


def test_preview_summary():
    pv = sample_preview(default_spec(2, 2), n=300, seed=1)
    assert pv["n_arms"] == 2 and len(pv["arm_counts"]) == 2
    assert 0.0 <= pv["y_rate"] <= 1.0


def test_build_config_and_run(tmp_path):
    pytest.importorskip("gurobipy")
    from core.experiment import build_experiment_config, run
    state = dict(experiment="apptest", n_arms=2, cap=[1.0, 0.5], gamma_true=3.0, gammas=[3.0], seeds=[0],
                 methods=[{"name": "IPW-X-X", "variants": ["Capped"]}, {"name": "Kallus", "variants": ["Flat"]}],
                 maximize=True, n_train=60, n_test=80, zscore=True, metric="euclidean",
                 snap=True, mesh=5, mesh_range=(-1.0, 1.0), rounding_digits=6)
    cfg = build_experiment_config(state)
    assert cfg.n_arms == 2 and [m.name for m in cfg.methods] == ["IPW-X-X", "Kallus"]
    exp, _log = run(cfg, build_generate(default_spec(2, 2)), results_root=tmp_path)
    assert (exp / "IPW-X-X" / "Capped" / "gtrue-3" / "outcomes.csv").exists()
    assert (exp / "Kallus" / "Flat" / "gtrue-3" / "outcomes.csv").exists()


def test_preset_exp_c_generates_grid_4arm():
    from core.presets import PRESETS
    spec = PRESETS["exp_c · discrete 4-arm (R-OW wins)"]
    assert spec.mode == "code" and spec.n_arms == 4
    d = build_generate(spec)(200, 3.0, np.random.default_rng(0))
    assert d.X.shape == (200, 1) and d.Ypot.shape == (200, 4) and d.mu.shape == (200, 4)
    assert set(np.unique(d.T)) <= {0, 1, 2, 3}
    assert len(np.unique(d.X)) <= 21                           # X lives on the 21-point grid
    assert np.all((d.mu >= 0) & (d.mu <= 1))


def test_all_presets_generate_valid():
    from core.presets import PRESETS, CUSTOM
    for name, spec in PRESETS.items():
        if name == CUSTOM:
            continue
        K = spec.n_arms
        d = build_generate(spec)(150, 3.0, np.random.default_rng(1))
        assert d.Ypot.shape == (150, K) and d.mu.shape == (150, K), name
        assert d.T.shape == (150,) and set(np.unique(d.T)) <= set(range(K)), name
        assert np.all((d.mu >= 0) & (d.mu <= 1)), name
