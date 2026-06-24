# Policy Optimization under Unobserved Confounding

Code, experiments, and results for robust policy optimization when treatment
assignment depends on an unobserved confounder. Methods are organized by the
`estimator-uncertaintyset-W` convention:

- **X-X** (no uncertainty set): `IPW-X-X`, `DoublyRobust-X-X`, `Direct-X-X`
- **O-X** (MSM odds-box): `IPW-O-X`, `DoublyRobust-O-X`, `Hajek-O-X`
- **O-W** (box ∩ Wasserstein covariate balance): `IPW-O-W`, `DoublyRobust-O-W`
- `Oracle`

## Layout

| Path | Contents |
|------|----------|
| `methods/` | Solvers, split into `Capped/` and `Uncapped/` per method |
| `common/` | Shared pre-algorithm modules (propensity, weights, outcome means, Wasserstein ε, Gurobi setup) |
| `assets/` | Per-experiment DGPs, runners, build scripts, and saved numeric results (NPZ/JSON) |
| `extensions/` | Shapley / KNN deployment extensions |
| `results/` | Saved experiment outputs (see `results/SAVING.md`) |
| `application/` | Streamlit "Policy Lab" GUI app |
| `tests/` | Test suite |
| `index.html` | Interactive research report (all experiment tabs) |
| `memory/` | Project notes / running summary of the work |
| `run_experiment.py` | Single-process config-driven runner |
| `assets/run_experiment_parallel.py` | Parallel runner (multiprocessing over (regime, seed) jobs) |

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `scikit-learn`, `gurobipy` (Gurobi license required for the
  optimization solvers), `matplotlib`
- Streamlit (for the `application/` GUI)

## Running

```bash
# single experiment (sequential)
python3 assets/run_experiment_parallel.py \
  --dgp assets/exp_owwin3/dgp.py --out assets/exp_owwin3/owwin3_results.json \
  --n 700 --seeds 5 --gammas 1,1.5,2,3,4,6,8 --regimes uncap,cap --ceps 1.0 \
  --workers 8 --threads 1
```

Open `index.html` in a browser to view the full interactive report.
