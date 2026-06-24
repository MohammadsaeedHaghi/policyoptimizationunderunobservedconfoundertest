"""common.diagnostics: ESS + propensity_diagnostics shape/semantics."""
import numpy as np

import common


def test_ess_uniform_equals_n():
    assert np.isclose(common.effective_sample_size(np.ones(50)), 50.0)
    assert common.effective_sample_size(np.array([10.0, 0.0, 0.0])) < 1.5   # concentrated ⇒ tiny ESS


def test_propensity_diagnostics(k2):
    d = common.propensity_diagnostics(k2["X"], k2["T"], k2["K"])
    for key in ("min_propensity", "max_weight", "ess", "ess_frac", "overlap_ok", "per_arm"):
        assert key in d
    assert 0.0 < d["ess"] <= len(k2["T"]) + 1e-6
    assert isinstance(d["overlap_ok"], bool)
    assert set(d["per_arm"].keys()) == set(range(k2["K"]))
    for k in range(k2["K"]):
        assert d["per_arm"][k]["n"] >= 1
