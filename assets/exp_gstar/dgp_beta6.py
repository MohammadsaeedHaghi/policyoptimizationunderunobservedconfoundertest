"""exp_gstar coupling-sweep variant: P(S=+1|X) = sigma(ALPHA X) with ALPHA = 6 (base: 10).
Varies how predictable the hidden S is from X (the W-term's leverage) while the selection
odds ratio stays EXACTLY Gamma* = 5 (propensity untouched). Mutation pattern: the base module's
ALPHA is overwritten BEFORE re-export, so generate()/p_s1 see the new value."""
import importlib.util, sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent / "dgp.py"
_spec = importlib.util.spec_from_file_location("gstar_base_b6", str(_BASE))
_m = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _m; _spec.loader.exec_module(_m)
_m.ALPHA = float(6)

ALPHA, LEVELS, CAP, K, NOISE, THETA, LAM = _m.ALPHA, _m.LEVELS, _m.CAP, _m.K, _m.NOISE, _m.THETA, _m.LAM
p_s1, propensity, mu0, mu1 = _m.p_s1, _m.propensity, _m.mu0, _m.mu1
grid_truth, oracle_policy, generate, exact_value = _m.grid_truth, _m.oracle_policy, _m.generate, _m.exact_value
