"""exp_owgap_v2 cap-robustness variant: identical DGP, treatment budget cap = 50% (base v2: 30%).

Same generative process as assets/exp_owgap_v2/dgp.py in every respect; only the capped-regime
budget differs. Sweep cap in {0.3, 0.4, 0.5} shows the capped O-W margin is not an artifact of
one budget choice. Loaded by runners via --dgp; delegates everything to the v2 module.
"""
import importlib.util, sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent / "dgp.py"
_spec = importlib.util.spec_from_file_location("owgap_v2_base_c50", str(_BASE))
_m = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _m; _spec.loader.exec_module(_m)

CAP = (1.0, 0.5)
ALPHA, LEVELS, K, NOISE, THETA = _m.ALPHA, _m.LEVELS, _m.K, _m.NOISE, _m.THETA
p_s1, propensity, mu0, mu1 = _m.p_s1, _m.propensity, _m.mu0, _m.mu1
grid_truth, oracle_policy, generate, exact_value = _m.grid_truth, _m.oracle_policy, _m.generate, _m.exact_value
