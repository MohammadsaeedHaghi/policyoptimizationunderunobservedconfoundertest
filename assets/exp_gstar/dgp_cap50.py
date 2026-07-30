"""exp_gstar cap-robustness variant: identical DGP, treatment budget cap = 50% (base: 30%).
Delegates everything to dgp.py; only CAP differs (the runner reads d.CAP directly)."""
import importlib.util, sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent / "dgp.py"
_spec = importlib.util.spec_from_file_location("gstar_base_c50", str(_BASE))
_m = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _m; _spec.loader.exec_module(_m)

CAP = (1.0, 0.50)
ALPHA, LEVELS, K, NOISE, THETA, LAM = _m.ALPHA, _m.LEVELS, _m.K, _m.NOISE, _m.THETA, _m.LAM
p_s1, propensity, mu0, mu1 = _m.p_s1, _m.propensity, _m.mu0, _m.mu1
grid_truth, oracle_policy, generate, exact_value = _m.grid_truth, _m.oracle_policy, _m.generate, _m.exact_value
