"""exp_kz18 strength variant: Lambda* = e^{0.5} = 1.6487. GSTAR is read at call time
inside generate(), so setting the module attribute before re-export is sufficient."""
import importlib.util, sys
import numpy as np
from pathlib import Path

_sp = importlib.util.spec_from_file_location("kz18_core_g05", Path(__file__).resolve().parent / "dgp.py")
_m = importlib.util.module_from_spec(_sp); sys.modules["kz18_core_g05"] = _m; _sp.loader.exec_module(_m)
_m.LOGG = 0.5
_m.GSTAR = float(np.exp(0.5))
for _k in dir(_m):
    if not _k.startswith("__"):
        globals()[_k] = getattr(_m, _k)
