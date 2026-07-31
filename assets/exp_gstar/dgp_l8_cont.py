"""exp_gstar strength variant: Lambda* = 8.0. Mutation wrapper -- LAM feeds CS at module
level, so BOTH must be set before re-export or the propensity would keep the old odds ratio."""
import importlib.util, sys
import numpy as np
from pathlib import Path

_sp = importlib.util.spec_from_file_location("gstar_core_l8_cont", Path(__file__).resolve().parent / "dgp_cont.py")
_m = importlib.util.module_from_spec(_sp); sys.modules["gstar_core_l8_cont"] = _m; _sp.loader.exec_module(_m)
_m.LAM = 8.0
_m.CS = 0.5 * np.log(8.0)
for _k in dir(_m):
    if not _k.startswith("__"):
        globals()[_k] = getattr(_m, _k)
