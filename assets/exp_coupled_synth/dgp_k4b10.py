"""exp_coupled_synth wrapper: KAPPA = 4, BETA = 10 (see dgp.py)."""
import importlib.util as _ilu
from pathlib import Path as _P
_sp = _ilu.spec_from_file_location("coupled_k4b10", str(_P(__file__).resolve().parent / "dgp.py"))
_m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)
_m.BETA = 10; _m.KAPPA = 4
LEVELS, K, CAP, NOISE, GSTAR, BETA, KAPPA = _m.LEVELS, _m.K, _m.CAP, _m.NOISE, _m.GSTAR, 10, 4
e_nom, p_s1, propensity, mu0, mu1, cate = _m.e_nom, _m.p_s1, _m.propensity, _m.mu0, _m.mu1, _m.cate
oracle_policy, generate, grid_truth = _m.oracle_policy, _m.generate, _m.grid_truth
