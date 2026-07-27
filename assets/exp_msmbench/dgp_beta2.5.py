"""msmbench-coupled wrapper: BETA = 2.5 (see dgp_coupled.py)."""
import importlib.util as _ilu
from pathlib import Path as _P
_sp = _ilu.spec_from_file_location("kmz_coupled_b2.5", str(_P(__file__).resolve().parent / "dgp_coupled.py"))
_m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)
_m.BETA = 2.5
LEVELS, K, CAP, NOISE, GSTAR, BETA = _m.LEVELS, _m.K, _m.CAP, _m.NOISE, _m.GSTAR, 2.5
p_s1, propensity, mu0, mu1, cate = _m.p_s1, _m.propensity, _m.mu0, _m.mu1, _m.cate
oracle_policy, generate, grid_truth = _m.oracle_policy, _m.generate, _m.grid_truth
