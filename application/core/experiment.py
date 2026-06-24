"""Experiment adapter for the Policy Lab app — pure logic, NO Streamlit imports.

Bridges the UI state to the code-1.1 engine: builds an ``ExperimentConfig`` from the config-page state,
runs ``run_experiment`` with the DGP-page ``generate`` callable, and loads the saved results back for the
results page. Adds the code-1.1 root to sys.path so ``common`` / ``run_experiment`` resolve.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

# application/core/experiment.py -> parents: [0]=core [1]=application [2]=code 1.1
_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from common.config import (ExperimentConfig, GeometryConfig, SupportConfig, MethodSpec,  # noqa: E402
                           KNOWN_METHODS, WASSERSTEIN_METHODS, MAXIMIZE_METHODS, PARAMETRIC_METHODS,
                           VALID_VARIANTS)
import run_experiment as _runner  # noqa: E402

# What the UI offers, with the capability flags it needs to render the right controls.
METHOD_CATALOG = [
    {"name": m,
     "parametric": m in PARAMETRIC_METHODS,
     "wasserstein": m in WASSERSTEIN_METHODS,
     "maximize": m in MAXIMIZE_METHODS}
    for m in KNOWN_METHODS
]
DEFAULT_RESULTS_ROOT = _CODE_ROOT / "results"


def build_experiment_config(state: Dict) -> ExperimentConfig:
    """Construct an ``ExperimentConfig`` from the config-page state dict.

    ``state['methods']`` is a list of ``{"name", "variants"}``; ``n_arms`` must equal the DGP's K (the
    config page sources it from the DGP spec so they cannot disagree)."""
    geom = GeometryConfig(zscore=bool(state["zscore"]), metric=str(state.get("metric", "euclidean")))
    sup = SupportConfig(snap=bool(state["snap"]), mesh=int(state["mesh"]),
                        mesh_range=tuple(state["mesh_range"]), rounding_digits=int(state["rounding_digits"]))
    methods = [MethodSpec(m["name"], tuple(m["variants"])) for m in state["methods"]]
    return ExperimentConfig(
        experiment=str(state["experiment"]),
        n_arms=int(state["n_arms"]),
        cap=tuple(float(c) for c in state["cap"]),
        gamma_true=float(state["gamma_true"]),
        gammas=tuple(float(g) for g in state["gammas"]),
        seeds=tuple(int(s) for s in state["seeds"]),
        methods=methods,
        geometry=geom, support=sup,
        maximize=bool(state["maximize"]),
        n_train=int(state["n_train"]), n_test=int(state["n_test"]),
    )


def run(config: ExperimentConfig, generate: Callable, *, results_root=None, clean: bool = True):
    """Run the experiment, capturing the runner's stdout. Returns ``(exp_dir, log_text)``."""
    results_root = Path(results_root) if results_root else DEFAULT_RESULTS_ROOT
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exp_dir = _runner.run_experiment(config, generate, results_root=results_root, clean=clean)
    return exp_dir, buf.getvalue()


# ----------------------------------------------------------------------------- load results back
def _read_csv(path: Path) -> List[dict]:
    import csv
    with open(path) as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_results(exp_dir) -> Dict:
    """Load everything the results page needs: per-(method,variant) outcome rows, reference ceilings,
    per-seed diagnostics, and the saved config."""
    exp = Path(exp_dir)
    methods: Dict[str, List[dict]] = {}
    for csv_path in sorted(exp.glob("*/*/*/outcomes.csv")):          # <Method>/<Variant>/gtrue-γ/outcomes.csv
        method, variant = csv_path.parts[-4], csv_path.parts[-3]
        rows = _read_csv(csv_path)
        for r in rows:                                              # numeric coercion
            for k, v in list(r.items()):
                if k not in ("extension", "obj_kind"):
                    try:
                        r[k] = float(v)
                    except (ValueError, TypeError):
                        pass
        methods[f"{method}/{variant}"] = rows

    ceilings = {}
    ceil_path = exp / "reference" / "ceilings.npz"
    if ceil_path.exists():
        with np.load(ceil_path) as z:
            full = [float(z[k]) for k in z.files if k.startswith("full_info")]
            best = [float(z[k]) for k in z.files if k.startswith("best_means")]
        if full:
            ceilings["Full info"] = float(np.mean(full))
            ceilings["Best (true means)"] = float(np.mean(best)) if best else None

    diagnostics = []
    for dpath in sorted((exp / "diagnostics").glob("seed*.json")) if (exp / "diagnostics").exists() else []:
        diagnostics.append(json.loads(dpath.read_text()))

    config = json.loads((exp / "config.json").read_text()) if (exp / "config.json").exists() else {}
    return dict(exp_dir=str(exp), methods=methods, ceilings=ceilings,
                diagnostics=diagnostics, config=config)
