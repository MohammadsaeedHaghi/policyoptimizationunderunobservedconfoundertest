"""KMZ'19 DGP at confounding strength log(Gamma*) = 2.0 (Gamma* = e^2.0), per the paper's own
sweep (Kallus-Mao-Zhou 2019, arXiv 1810.02894, Sec. experiments: log Gamma* in {0.5, 2.0, 1.5}).
Mutation-pattern wrapper over dgp.py: the base module's GSTAR is overwritten BEFORE re-export,
so propensity()/generate() see the new value. The MSM-extremal construction keeps the
sensitivity model exactly specified at the known Gamma*."""
import importlib.util as _ilu, sys as _sys
import numpy as _np
from pathlib import Path as _P
_sp = _ilu.spec_from_file_location("kmz_base_g20", str(_P(__file__).resolve().parent / "dgp.py"))
_m = _ilu.module_from_spec(_sp); _sys.modules[_sp.name] = _m; _sp.loader.exec_module(_m)
_m.GSTAR = float(_np.exp(2.0))

GSTAR, LEVELS, K, CAP, NOISE = _m.GSTAR, _m.LEVELS, _m.K, _m.CAP, _m.NOISE
e_nom, p_s1, propensity, mu0, mu1, cate = _m.e_nom, _m.p_s1, _m.propensity, _m.mu0, _m.mu1, _m.cate
oracle_policy, generate, grid_truth = _m.oracle_policy, _m.generate, _m.grid_truth
