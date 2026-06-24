---
name: code_1_1_overhaul
description: "The clean \"code 1.1\" restructure — folder layout, method naming, common/ pre-algorithm modules, results spec."
metadata: 
  node_type: memory
  type: project
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

**`code 1.1/`** (started 2026-06-09) is a from-scratch clean restructure of the project, a **sibling of `code/`**
at the workspace root (`…/paper - policy optimization under unobsorved confounder - code/code 1.1/`). Built
incrementally on the user's direction.

**Layout:**
```
code 1.1/
├── methods/   <Method>/<Capped|Uncapped>/   (8 capacity-capable) + Kallus/ (flat, parametric)
├── extensions/  Shapley/  KNN/
├── results/   SAVING.md (the save-everything spec) — experiment outputs go here
└── common/    pre-algorithm shared .py modules
```

**Method naming (user-chosen, hyphenated; encode Odds-box / +Wasserstein):**
- `R-OW` = robust, odds-box ∩ Wasserstein (was RW / `solve_rw_multiarm`)
- `R-O` = robust, odds-box only (was RW-noW / `solve_rw_no_wasserstein_multiarm`)
- `Regret-OW` = Kallus regret, box ∩ Wasserstein (was Kallus-W / `solve_kallus_w_multiarm`)
- `Regret-O` = Kallus regret, box only (was Kallus-odds / `solve_kallus_multiarm`)
- `Kallus` = the **parametric** Kallus (softmax θ) — **flat, NO Capped/Uncapped** (capacity not enforceable on a
  parametric policy)
- `IPW` (was Direct-IPW), `Policy-Optimization` (was Direct-Optimization), `Oracle`, `DoublyRobust` (new, TBD)
- Each capacity-capable method has `Capped/` and `Uncapped/` subfolders ("Cap" must be in the name — user req).

**`common/` (pre-algorithm primitives, implemented + smoke-tested 2026-06-09):**
- `propensity.py` — `fit_propensity_model`, `propensity_matrix(X,T,K)`, `factual_propensity`, `propensity_binary`.
  Multinomial logistic on (X,T); clip 1e-3; rows renormalised. **Dropped the deprecated `multi_class` arg** (no
  sklearn FutureWarning). Takes ONLY (X,T) — structurally cannot use e_true (see [[no_true_propensities]]).
- `hajek.py` — `inverse_propensity_weights`, `hajek_normalize(target="n"|"size"|"global")`, `ipw_weights_from_data`.
  Per-arm Σw=n is the project default/convention.
- `geometry.py` — `pairwise_distance_matrix`, `standardize` (z-score, fit-train/apply-test), `distance_matrix(X, zscore=True)`, `euclidean_distance`. The Wasserstein ground cost.
- `support.py` — `snap_to_grid(X, n_levels=6)`, `grid_levels`, `tie_groups`, `build_lp_support(...)→SupportInfo`.
  The continuous-X binning that makes the free-π LP non-degenerate (see the no-grid degeneracy in [[conti_binary_study]]).
- `sensitivity.py` — `matched_gamma(γ)=e^{γ/2}`, `marginal_sensitivity_box(ŵ,Γ)→(a,b)` (min/max handles
  sub-1 Hájek weights), `augmented_gamma_grid`.
- `gurobi_env.py` — `configure_gurobi_license()` (ported; shared by all LP methods).
- `wasserstein_radius.py` — `tight_epsilon_arm`, `tight_epsilon` (the good_epsilon transport LP; needs Gurobi).
All faithful ports of `code/common/lp_utils.py` + `code/rw_implementation/srpo` conventions, so numerics match.
`__init__.py` re-exports the full API (`import common; common.ipw_weights_from_data(...)`). PENDING common modules
proposed but not yet built: `data.py`, `diagnostics.py` (overlap/ESS), `persistence.py` (implements SAVING.md), `seeding.py`.

**Methods implemented so far:**
- **R-OW** (2026-06-09) — robust Odds-box ∩ Wasserstein value maximiser (= old `solve_rw_multiarm`). TWO
  self-contained files: `methods/R-OW/Capped/r_ow_capped.py` (`solve_r_ow_capped`, has the capacity constraint)
  and `methods/R-OW/Uncapped/r_ow_uncapped.py` (`solve_r_ow_uncapped`, no cap). Each returns `ROWResult` (pi,
  support_X, objective, usage, + duals β/μ/ν/γ/θ). "Just the method": weights passed in (no propensity est).
  Args: `geometry` (bool: snap-to-grid via support.py vs raw X) + `mesh` (grid levels/axis, default 6) +
  `mesh_range`, `c_eps`, `epsilon` (override; else internal tight-ε), `rounding_digits`. **VERIFIED bit-identical
  to old `solve_rw_multiarm`** (max|Δπ|=0, |Δobj|=0) on matched inputs. Folder HTML `methods/R-OW/R-OW.html`
  writes out the in-sample problem (primal worst-case value max–min over the MSM box ∩ Wasserstein balls + the
  dual LP actually solved, MathJax).
- **R-O** (2026-06-09) — robust Odds-box-ONLY value maximiser = R-OW minus the Wasserstein term (= old
  `solve_rw_no_wasserstein_multiarm`). `methods/R-O/{Capped/r_o_capped.py (solve_r_o_capped), Uncapped/r_o_uncapped.py
  (solve_r_o_uncapped)}` + `R-O.html`. Returns `ROResult` (pi, support_X, objective, usage, β/μ/ν — NO Wasserstein
  duals). β here is the FREE-SIGN per-arm Hájek-calibration multiplier (NOT a radius). Same `geometry`/`mesh`
  discretisation args but NO `c_eps`/`epsilon` (no Wasserstein). **VERIFIED bit-identical to old**
  (max|Δπ|=|Δobj|=|Δβ|=|Δμ|=0).
- **Regret-OW, Regret-O, IPW, Policy-Optimization** (2026-06-10, via a 4-agent `port-four-methods` workflow,
  then parent-verified). Each: `methods/<M>/{Capped,Uncapped}/<file>.py` + `<M>.html`, mirroring the R-OW/R-O
  template, same geometry/mesh args.
  • **Regret-OW** = old `solve_kallus_w_multiarm` (Kallus regret over box ∩ Wasserstein; MIN LP; obj=worst-case
    regret ≤0). `solve_regret_ow_capped/uncapped`. mu/nu hold Kallus box duals p/q; beta≥0 = Wasserstein radius;
    has gamma_dual/theta. Needs c_eps/epsilon.
  • **Regret-O** = old `solve_kallus_multiarm` (box only; MIN LP). `solve_regret_o_capped/uncapped`. beta=free-sign
    calibration; mu/nu=p/q; no Wasserstein.
  • **IPW** = old `solve_direct_ipw_multiarm` (MAX Hájek/IPW value; no Gamma/box/duals). `solve_ipw_capped/uncapped`.
  • **Policy-Optimization** = old `solve_direct_opt_multiarm` (naive MAX Σ Yπ, unit weights, NO ips_weights arg).
    `solve_policy_optimization_capped/uncapped`.
  **Verification (capped, bit-identical to old):** Regret-OW, Regret-O, IPW ALL bit-identical (max|Δπ|=|Δobj|=0,
  duals match). **Policy-Optimization: |Δobj|=2e-16 but max|Δπ|=1.0** — EXPECTED alternate-optima (naive unit-weight
  objective Σ Yπ has values in {0,1} → massive ties → non-unique optimum; the old code is equally non-unique; IPW
  with real weights breaks the ties and IS bit-identical). Uncapped variants all run, valid simplex, no cap
  constraint/field. All 4 HTMLs validate. **NB regret LPs (Regret-OW/-O) require per-arm Hájek-calibrated weights
  (Σ_{T=k} w=n) or the free-sign β makes them unbounded — that's the `common.ipw_weights_from_data` convention.**

- **Oracle** (2026-06-10) — reference ceiling = old `solve_oracle_capacity`. `methods/Oracle/{Capped/oracle_capped.py
  (solve_oracle_capped), Uncapped/oracle_uncapped.py (solve_oracle_uncapped)}` + `Oracle.html`. Takes a value matrix
  `values` (n,K) — pass Ypot (Full-info) or μ (Best-means); geometry/mesh; OracleResult(pi,support_X,objective,
  cap,usage). **VERIFIED bit-identical** (max|Δπ|=0). NB: Oracle legitimately uses ground-truth values (it's a
  benchmark, not a learned/deployed method) — does NOT violate [[no_true_propensities]].
- **Kallus** (2026-06-10) — PARAMETRIC Kallus&Zhou regret-minimiser, **flat** (no cap, no discretisation — it's a
  smooth softmax). `methods/Kallus/kallus.py` (`fit_kallus`, `predict_kallus`, `design_matrix`) + `Kallus.html`.
  **NEW DERIVATION** (no reference): the repo only had parametric *value* (`parametric_multiarm`); I adapted it
  value→regret — inner Gurobi LP = worst-case REGRET (MAX over box[∩Wasserstein] of (1/n)Σ(1[T=0]−π_θ(T|X))Yw),
  outer subgradient DESCENT (θ += η·score-grad, keep min-regret θ). `wasserstein=False` (box+calibration, classic
  odds) or True (box∩Wasserstein, needs epsilon). CAVEAT (in HTML): worst-case regret can be slightly >0 because
  the softmax can't represent π0 exactly (free-π Regret-O hits ≤0; here +0.07 on the test data). Verified by
  sanity (runs, valid softmax, finite regret; slack-ε Wasserstein correctly reduces to box-only calibration).

- **DoublyRobust** (2026-06-10) — robust AIPW value-max, **box-only** (user chose this formulation + box-only via
  AskUserQuestion). `methods/DoublyRobust/{Capped/doublyrobust_capped.py (solve_doublyrobust_capped), Uncapped/
  doublyrobust_uncapped.py (solve_doublyrobust_uncapped)}` + `DoublyRobust.html`. Objective = direct outcome term
  (1/n)Σ_iΣ_k π_k μ̂_k(X_i) + worst-case-box on the RESIDUAL r_i=Y_i−μ̂_{T_i}(X_i) IPW correction = R-O's box-only
  dual with Y→resid PLUS the direct term. Takes ips_weights AND `outcome_means` (n,K) — both estimated upstream.
  NEW nuisance `common/outcome_model.py` → `outcome_means(X,T,Y,K, cross_fit=False, kind='auto')` (per-arm
  logistic if Y binary else linear; optional K-fold cross-fit). **VERIFIED: DoublyRobust(μ̂=0) == R-O bit-identical**
  (capped+uncapped, all duals 0) ⇒ transitively grounded to the old solver; real & cross-fit μ̂ run with valid
  simplex, usage≤cap. Γ=1 ⇒ ordinary AIPW. (box∩Wasserstein DR-OW deferred — "box-only first".)

**Methods status: 9/9 DONE** — R-OW, R-O, Regret-OW, Regret-O, IPW, Policy-Optimization, Oracle, DoublyRobust (all
bit-verified or μ̂=0-reduction-verified), Kallus (parametric, new derivation, sanity-verified). Every method folder
has its solver(s) + a folder HTML stating the in-sample problem.

**EXTENSIONS done (2026-06-10):** `extensions/Shapley/shapley.py` (extract_support, shapley_pdf closed-form,
extend_with_shapley, extend_with_shapley_multiarm) + `extensions/KNN/knn.py` (extend_with_knn, extend_with_knn_multiarm).
Self-contained (numpy/pandas/sklearn; KNN imports extract_support from the Shapley dir). **VERIFIED bit-identical**
to old `srpo.extension.*_multiarm` (max|Δ|=0, both extensions). Deploy a method's on-support (K,n) policy to new X.

**RUNNER + PERSISTENCE done (2026-06-10):** `common/persistence.py` (gkey `q__G<Γ>__seed<s>`, experiment_dir,
method_dir, save_json/npz, write_csv, save_raw_draw — implements SAVING.md). `code 1.1/run_experiment.py` —
DGP-agnostic driver: `run_experiment(slug, generate_fn, n_arms, cap, gamma_true, gammas, seeds, methods, variants,
geometry, mesh)`. Has a METHOD REGISTRY (name→stem,fn,kind ∈ {robust_w, robust, ipw, po, dr}) handling the
heterogeneous signatures; loads method files by path (importlib+sys.modules). Per seed: estimate w
(common.ipw_weights_from_data) + μ̂ (common.outcome_means), save raw draw; per method×variant×Γ: fit on train,
extend (Shapley+KNN) to train & test, compute outcomes, save policy.npz (pi+duals+support_X, keyed by Γ,seed) +
extended.npz + outcomes.csv; reference ceilings (Full-info on Ypot, Best-means on μ) in reference/. Γ-indep methods
(IPW/PO) fit once at matched Γ. **DEMOّD end-to-end** (`python run_experiment.py`, synthetic 2-arm DGP) →
`results/demo-synthetic-K2/` with the full SAVING.md tree, validated. NOTE: runner registry covers the LP methods +
Oracle ceilings; **Kallus (parametric)** isn't in it (saves θ not on-support π — plugs in separately), and the
box+Wasserstein **DR-OW** is deferred ("box-only first"). Still pending `common`: data.py, diagnostics.py, seeding.py.

**HTML DERIVATIONS added (2026-06-10, via an 11-agent workflow; parent authored the math, agents formatted+inserted,
parent reviewed):** every method HTML now has a numbered "How it is solved — derivation" section showing how the
in-sample problem becomes the LP/algorithm actually run: R-O & R-OW dualise the inner min_w (box / box∩Wasserstein
transport) via LP strong duality → the single LP; Regret-O/-OW dualise the inner MAX → the MIN LP; DoublyRobust =
R-O's dual on the residual + direct term; IPW/Policy-Optimization/Oracle = "already an LP, solved directly"; Kallus
= inner LP (per θ) + outer envelope subgradient (not one LP). **Extensions now have HTMLs** (`extensions/Shapley/
Shapley.html`, `extensions/KNN/KNN.html`) — the closed-form π^S=min_k max_j A_jk operator / inverse-distance kNN,
each: what it is, the operator, the compute steps, and how it's used (extend_with_*_multiarm(X_test, support_X, π)).
All 11 HTMLs validate (div balance, MathJax, display-delimiter balance); R-OW/Regret-OW Wasserstein duals spot-checked.

**CODE↔HTML ALIGNMENT AUDIT (2026-06-10, 11-agent report-only workflow + parent verification):** 9/11 fully
aligned. TWO real misalignments found, both = HTML overclaim vs code (parent-verified, documented under a
"⚠ Misalignment — code vs. this page" section in those two HTMLs; code NOT changed — surfaced for the user to
resolve): (1) **Kallus.html** said θ is selected "across all iterations (not merely the last iterate)" but
`kallus.py` discards per-iteration regret and only compares each restart's FINAL iterate (best-across-restarts).
(2) **Shapley.html** prose called the 2nd arg `support_X` but the signature is `extend_with_shapley_multiarm(X_test,
X_train, π)` — raw X_train, collapsed internally (behaviourally harmless; API table was already correct). The other
7 methods + KNN + R-OW/R-O/Regret/IPW/PO/Oracle/DoublyRobust: clean. **RESOLVED 2026-06-10 (user: "fix both —
reword"):** reworded both docs to match the code (Kallus §3/§4 → "smallest worst-case regret across the restarts,
each evaluated at its final iterate"; Shapley §4 → 2nd arg is `X_train`, receives the returned support_X, collapsed
internally) and REMOVED the two Misalignment sections (now aligned). All 11 HTMLs aligned + valid. Code unchanged.

**KALLUS-PAPER VERIFICATION + FIXES (2026-06-11, user: "read kallus paper … is your kallus okay"):** Read
`paper/kallus/kallus.pdf` (Kallus & Zhou 2021, "Minimax-Optimal Policy Learning Under Unobserved Confounding"),
pp.1–14 incl. Algorithm 1 (p.2883). **Core faithful**: MSM/Tan odds box eq(3) `aᵢ=1+Γ⁻¹(W̃ᵢ−1)`,
`bᵢ=1+Γ(W̃ᵢ−1)` is EXACTLY `common/sensitivity.marginal_sensitivity_box`; regret vs π₀=all-control; Hájek
self-normalisation E[W·1[T=t]]=1; softmax policy class. Flagged **4 deviations**; user confirmed fixing 1–3 and
clarified 4:
- **(1) exact policy-score gradient** — `Kallus/kallus.py` outer loop now uses the paper's
  `∇π_θ(Tᵢ|Xᵢ)=π_{Tᵢ}(onehot−π)φ` (added the missing `π_t` factor): `g=(Yc·w*/n·π_t)[:,None]*(onehot−π); grad=gᵀΦ`.
- **(2) Polyak iterate-average return** — returns `θ̄=(1/n_iters)Σ_k θ_k` per restart (Algorithm 1's average of
  iterates), not the last iterate; keeps the min-regret θ̄ across restarts.
- **(3) `maximize` convention** — added `maximize: bool = True` arg to **kallus.py AND all 4 Regret LP solvers**
  (Regret-O/-OW × Capped/Uncapped). `Yc = Y if maximize else -Y`; reward(default)→regret V(π₀)−V(π) (studies'
  convention), loss(maximize=False)→regret V(π)−V(π₀) (the paper's exact convention, reward=−loss). Replaced every
  `float(Y[i])`→`float(Yc[i])` in the regret/feas constraints.
- **(4) Regret-OW is an INTENTIONAL HYBRID, not pure Kallus** (user: "regret_OW … should use the OW as the
  uncertainty set for the w and also regret as the objective … but … should not have the policy class restricted
  since we want to add capacity"): Kallus regret objective + R-OW's odds-box∩Wasserstein "OW" uncertainty set +
  FREE (non-parametric) π so capacity is addable — NOT Kallus's own extension (total-variation budget) and NOT the
  parametric softmax. Documented as a `.card` ("An intentional hybrid — not pure Kallus") in Regret-OW.html.
- **VERIFIED (separate processes):** maximize=True (default) bit-identical to OLD Regret-O/Regret-OW solvers
  (max|Δπ|=|Δobj|=0 — no regression); maximize=False runs (loss obj differs as expected); Kallus exact-grad+Polyak
  regret IMPROVED to +0.0665 (from old +0.0714), still slightly >0 (softmax-can't-be-π₀ caveat holds), valid softmax.
- **HTML synced:** Kallus.html (exact ∇π in §3/§4, Polyak return, maximize `.muted` note + Arguments row),
  Regret-O.html & Regret-OW.html (added `maximize` Arguments row), Regret-OW.html (the intentional-hybrid card). All
  validate (div + `\(\)` + `\[\]` balanced).

**EXPERIMENT CONFIG + ACTIVE GEOMETRY (2026-06-11, user: "in config you should get the geometry, support, and the
methods that we want to compare"):** New `common/config.py` with typed dataclasses (JSON round-trip via
`save_json`/`load_json`/`to_dict`/`from_dict`), re-exported from `common/__init__`:
- `GeometryConfig(zscore=True, metric="euclidean")` — the Wasserstein ground-cost geometry (only euclidean
  implemented; bad metric raises).
- `SupportConfig(snap=True, mesh=6, mesh_range=(-1,1), rounding_digits=6)` — the discretisation. NB `snap` maps to
  the solver's confusingly-named `geometry` (bool) arg.
- `MethodSpec(name, variants=("Capped","Uncapped"))` — one comparison method + its variants (validates name ∈
  KNOWN_METHODS, variants ∈ {Capped,Uncapped}).
- `ExperimentConfig(experiment,n_arms,cap,gamma_true,gammas,seeds,methods, geometry,support, maximize=True,
  n_train,n_test)` — accepts method names/dicts/MethodSpecs; validates cap length == n_arms.
- Capability sets (must mirror the solver signatures): `WASSERSTEIN_METHODS={R-OW,Regret-OW}` (accept geometry),
  `MAXIMIZE_METHODS={Regret-O,Regret-OW}` (accept maximize). Verified by `inspect.signature`: ALL solvers accept
  geometry/mesh/mesh_range/rounding_digits; only R-OW/Regret-OW have c_eps/epsilon (and now zscore/metric); only
  Regret-O/Regret-OW have maximize; DR has positional `outcome_means`; Oracle takes `values` not weights.

**ACTIVE geometry threaded into the 4 Wasserstein solvers** (user chose "Active — thread into the W solvers"):
R-OW & Regret-OW (Capped+Uncapped) gained `zscore: bool = False, metric: str = "euclidean"` kwargs; the D build
changed from `pairwise_distance_matrix(support_X)` → `distance_matrix(support_X, zscore=zscore)` (+ metric guard).
**Solver-param default `zscore=False` = raw distances = BIT-IDENTICAL to the original solver** (verified R-OW
obj=0.53701 unchanged, Δπ=0); the `GeometryConfig` default is `zscore=True` (scale-fair), so the config path
standardises. Proof zscore is live: it changes the tight ε radii (0.071/0.086 → 0.117/0.142); the optimum is
unchanged on the 2-arm fixture only because the W term is non-binding there (β=[0,0], the documented 2-arm collapse).

**`run_experiment.py` is now CONFIG-DRIVEN:** signature is `run_experiment(config: ExperimentConfig, generate, *,
results_root=None)`. `_call` is capability-gated (passes geometry only to WASSERSTEIN_METHODS, maximize only to
MAXIMIZE_METHODS) and threads the full SupportConfig (`_support_kwargs`) to every solver + Oracle. Per-method
variants come from each `MethodSpec.variants`. The persisted `results/<exp>/config.json` is now the full config
(geometry+support+methods+knobs), replacing the old hand-rolled provenance dict. `__main__` demo rebuilt around a
config; runs end-to-end (R-OW/R-O/IPW Capped+Uncapped + DoublyRobust Capped-only honoured). HTML synced: R-OW.html
& Regret-OW.html got a `zscore`/`metric` Arguments row (both validate). Smoke still 9/9 (no regression). See
[[keep_html_in_sync]].

**SMOKE HARNESS:** `_smoke/` (gitignore-able scratch) has `make_fixture.py` (DGP→estimated w+μ̂→`fixture.npz`),
`smoke_all.py` (rerunnable end-to-end maximize=False check over all 9 algos + extensions; writes `smoke_results.json`)
— all 9 PASS; independently adversarially verified (no refutations). Reuse for quick post-change regression checks.

**FOUR FOLLOW-UP EDITS (2026-06-11, user picked all 4 from my recommendation):**
1. **Kallus wired into config/runner.** `KNOWN_METHODS` now includes `"Kallus"`; `PARAMETRIC_METHODS=("Kallus",)`;
   `MethodSpec` forces Kallus's variant to `("Flat",)` (parametric ⇒ no Capped/Uncapped, no capacity). The runner
   has a dedicated **step-5 Kallus branch** (it's NOT in REGISTRY/_call): `fit_kallus(...maximize=cfg.maximize)`
   per Γ×seed (odds set, wasserstein=False = faithful paper baseline), deploys via `predict_kallus(θ, X)` (NO
   Shapley/KNN — the softmax IS the deploy), saves θ keyed by (Γ,seed) to `Kallus/Flat/`. obj_kind=worst_case_regret.
2. **`obj_ipw` → `objective_value` + `obj_kind`.** The per-row objective column was misleadingly named `obj_ipw`
   for ALL methods. Now `objective_value` + an `OBJ_KIND[method]` semantics tag (worst_case_value | worst_case_regret
   | ipw_value | naive_value | robust_aipw_value). NOT cross-method comparable — the tag prevents mixing; the
   reward/loss convention stays separate in config.json (`maximize`).
3. **`common/diagnostics.py`** (re-exported from `common`): `propensity_diagnostics(X,T,K, min_overlap=0.05)` +
   `effective_sample_size` (Kish). Estimates ê(X) (never e_true), reports per-arm + global min propensity, max
   inverse weight, ESS/ESS-frac, `overlap_ok`. Runner runs it per seed → `results/<exp>/diagnostics/seed<s>.json`
   and prints a `[warn]` if overlap_ok=False. The safety net for the no-true-propensities regime ([[no_true_propensities]]).
4. **Renamed the solver bool `geometry` → `discretize`** (resolves the naming collision: config `geometry`=metric/
   zscore, `support.snap`=the toggle, solver `discretize`=the toggle). Done via an 8-agent fan-out workflow + audit
   across all **16 LP solver files + 9 method HTMLs**; I did the callers (`run_experiment._support_kwargs` →
   `discretize=support.snap`, `_smoke/smoke_all.py`, Kallus docstring/HTML "no geometry/mesh"→"no discretize/mesh").
   `py_compile` can't catch a signature/body mismatch (un-renamed `if geometry else` compiles, fails at runtime) so
   I verified at RUNTIME: 0 toggle leftovers (`grep "if geometry\b|geometry: bool|geometry=True"` empty), 16/16
   solvers have `discretize: bool`, **smoke 9/9**, demo runs, all 9 HTMLs balanced (0 `<code>geometry</code>`),
   `discretize=False` no-grid path works.
**Verified end-to-end:** demo config (R-OW,R-O,IPW,Kallus,DoublyRobust-Capped) saves the full tree incl. Kallus/Flat
+ per-seed diagnostics + objective_value/obj_kind CSVs.

**CLEAN GUARD + TESTS (2026-06-11, user: "add the experiment dir clean guard and a tests/ dir"):**
- **Clean guard** — `common.persistence.prune_experiment_dir(exp, planned_leaf_dirs, *, dry_run=False)`: removes
  stale `<Method>/<Variant>/gtrue-<γ>` leaves NOT in the current run's plan (dropped method, dropped Capped/Uncapped
  variant, or superseded gamma_true); NEVER touches run-level `data/`/`diagnostics/`/`reference/`/`config.json`;
  cleans now-empty method/variant dirs. `run_experiment(..., clean=True)` (DEFAULT) prunes + logs `clean guard:
  removed N stale dir(s): …`; `clean=False` warns (dry_run) but keeps. Fixes the orphan-`obj_ipw` confusion.
- **`tests/` dir (pytest, 46 tests, ~2.7s, all pass):** `conftest.py` (path-load helpers `load_solver`/`load_module`
  + tiny confounded K-arm DGP fixtures `k2`/`k3` with estimated w+μ̂). `test_config.py` (round-trip, guards,
  Kallus→Flat, coercion), `test_common.py` (per-arm Hájek Σw=n, no-true-propensity signature, MSM box=eq3, support,
  zscore distance), `test_diagnostics.py`, `test_persistence.py` (clean guard), `test_methods.py`
  (`pytest.importorskip("gurobipy")`; every LP method Capped+Uncapped valid simplex/finite/usage≤cap for **K=2 AND
  K=3**, maximize=False↔−Y identity, **discretize=False** no-grid path, Oracle, Kallus parametric), `test_runner.py`
  (end-to-end saved tree + obj_kind + Kallus + diagnostics; clean-guard prune vs keep). Run: `python3 -m pytest tests/ -q`.

**TOP-LEVEL `index.html` (2026-06-11):** tabbed overview page at `code 1.1/index.html` — **11 tabs**: an
**Overview** tab (the 3 inline-SVG figures — pre-algorithm **pipeline** data→common→solver→extension→results with S
marked UNOBSERVED, **method taxonomy** 4 families/canonical colours, **uncertainty set** U(Γ,ε)=(O)box∩(W)balls — +
common/ + extensions/ + conventions) followed by **one tab per method** (R-OW, R-O, Regret-OW, Regret-O, Kallus,
IPW, Policy-Optimization, DoublyRobust, Oracle) and an **Experiment** tab (placeholder). User wanted per-method tabs,
NOT all on one scroll. (2026-06-14: now **4 top tabs** — Methods [Overview + per-method subtabs], **Configuration**
[ExperimentConfig field tables + config-structure SVG + rules + example; field names have CSS hover tooltips with
example values], **Sanity Check** [renamed from the old Experiment tab — the exp_c old-code vs code-1.1 PARITY at
n=200/600: config card, pre-step + result parity tables, 4 Γ-sweep figures in assets/parity_exp_c/], **Experiment**
[a real code-1.1 method comparison; subtab `exp_c · n=1000` = exp_c at n_train=1000/n_test=2000, 4 figures in
assets/exp_c_n1000/]. `showSub` is now SCOPED to `btn.closest('.tab-panel')` so Methods & Experiment subtabs don't
clobber each other. Modern CSS refresh (Inter + JetBrains Mono, sticky frosted tabs, soft-shadow surfaces, rounded
hover tables, dark code block). exp_c@n1000 run: /tmp/exp_n1000.py — n=1000 Wasserstein LP slow (~260s/Γ, ~28min full sweep; matches the study's
"N=1000 dropped" note), tight-ε precomputed once. DONE: 4 figures in assets/exp_c_n1000/ (+csv/npz) embedded.
**Finding @ n=1000:** at the operating Γ, R-OW ≈ R-O ≈ 0.72 > IPW 0.688; regret methods conservative (~0.54);
ceilings Full-info 0.966 / Best-means 0.738. The Wasserstein edge R-OW had over R-O at n=200 (Sanity Check) CLOSES
at n=1000 — more data ⇒ better overlap ⇒ box-only R-O catches R-OW. Broken-img guard: n=1000 imgs have
onerror=hide + a removable #genph "generating" banner (removed once plots exist).
**2-ARM companion (2026-06-14):** added a 2nd Experiment subtab `2-arm · discrete · n=1000` — same discrete-X/
unobserved-S family reduced to K=2 (outcome σ(a+bX+cS), a=(0,0) b=(-0.7,0.7) c=(0,1.4); treatment logit κX+γ(S-½),
κ=0.6). Script /tmp/exp_2arm.py, plots in assets/exp_2arm_n1000/ (~31min n=1000 Wasserstein). **Finding: the K=2
collapse** — R-OW = R-O = IPW ≈ 0.64 across the whole Γ-sweep (neither Wasserstein nor box robustness changes the
realised outcome; the R-OW edge needs ≥3 arms, cf exp_c K=4). Regret methods conservative (~0.51); ceilings Full-info
0.827 / Best-means 0.668. Confirms the documented 2-arm Wasserstein collapse.
**2-ARM where R-OW WINS (2026-06-14, user: "change DGP, keep 2 arms, until R-OW best with meaningful spread"):** the
1-D 2-arm collapses (R-OW=R-O); searched DGPs (/tmp/eval_dgp.py 1-D, /tmp/eval_dgp2d.py + /tmp/grid2d.py 2-D) →
**the R-OW edge in 2 arms needs 2-D covariates + a loose capacity cap** (covariate balance must matter AND bind).
Winner DGP (final /tmp/exp_2arm2d.py, assets/exp_2arm2d/, n=800): discrete 2-D X=(X1,X2) 11×11 grid, S~Bern(.5);
control σ(0.5), treatment σ(-1.2+4·proj+5·S) proj=X1+0.6X2; treatment logit -proj+γ(S-½); γ_true=5 (matched Γ=12.18),
cap=(1.0,0.5), zscore ON. **Result: R-OW 0.665 > IPW 0.645 > R-O 0.623 > Regret 0.60** at every Γ (W-edge R-OW−R-O=
+0.042); ceilings Full 0.848/Best-means 0.728. Policy plots are 2-D HEATMAPS (chosen-arm + R-OW P(treat) over the
11×11 grid) — R-OW learns a clean contiguous treat-region, IPW/R-O noisier, regret over-conservative. The exp_2arm
subtab now shows THIS (replaced the 1-D collapse; button "2-arm · 2-D · R-OW wins").) Each method tab = pill+colour, one-liner, in-sample problem (MathJax), how-solved, an
Arguments table, returns/duals, obj_kind, link to methods/<M>/<M>.html. JS `showTab` (scrolls top + re-typesets
MathJax); method tab buttons carry a colour dot. Spot-verified aligned to code (zscore only R-OW/Regret-OW; PO has
no ips_weights param; DR takes outcome_means; obj_kind strings match run_experiment.OBJ_KIND). Validates (div 47/47,
\(\) 85/85, \[\] 9/9, 11 buttons==11 panels, all 11 links resolve). NOTE for [[keep_html_in_sync]]: index.html is a
doc-of-record — update it when method behaviour/args change. NB MathJax does NOT typeset inside SVG <text>; use Unicode.

**HTML↔CODE RE-AUDIT (2026-06-11, 11-agent read-only workflow):** all 11 HTMLs (9 method + Shapley/KNN) audited
post-rename/post-config; **8 fully aligned, 0 material misalignments**, 3 COSMETIC stale labels fixed inline (no
misalignment sections needed): R-OW.html & Regret-OW.html §1 `common.pairwise_distance_matrix`→`common.distance_matrix`
(D now z-scoreable); KNN.html §3 3rd arg `pi`→`pi_train`. All pages still balance. See [[keep_html_in_sync]].

**END-TO-END PARITY vs code/ (2026-06-13, user: "run a code/ experiment with code 1.1, same config; report; if
different, explain in index.html"):** Ran **exp_c** (discrete 4-arm) through BOTH implementations in separate
processes with every pre-step held identical (old code generated data + estimated w + D + tight-ε + box; fed to both
solver sets; pre-steps also independently re-derived by `common`). Config: seed0, Γ=e^1.5=4.4817, Uncapped,
n_tr=200/n_te=600. **RESULT: bit-identical, no difference.** Pre-steps: max|Δŵ|=0 (same multinomial-logistic +
per-arm Hájek), max|Δε|=0 (same transport LP), D/box identical. Solvers (R-OW, R-O, IPW, Regret-O, Regret-OW):
|Δobj|=|Δrealised_test|=max|Δπ|=0 for all 5. R-OW realised 0.729 > R-O 0.685 > IPW 0.675 in both (R-OW-wins holds).
Only cosmetic diff: old `code/` emits a sklearn `multi_class='multinomial'` FutureWarning (code 1.1 dropped the arg;
sklearn default already multinomial ⇒ Δŵ=0, no effect). Documented in **index.html → Experiment tab** (was the empty
placeholder — now the reproducibility report with pre-step + result parity tables). Scripts: /tmp/parity_old.py
(old process) + /tmp/parity_new.py (new process).
**Γ-SWEEP + PLOTS (2026-06-13):** extended to a full sweep Γ∈{1,2,3,4.4817,5,7}, all 5 methods, both codes — still
bit-identical (max|Δobj|=0, max|Δrealised|=1.1e-16, max|Δπ|=0). Generated 4 comparison figures (old lines vs code-1.1
○): realised-outcome-vs-Γ, worst-case-objective-vs-Γ, policy-strips (chosen-arm-vs-X at matched Γ, old≡1.1 rows),
R-OW π_k(x) detail — saved in `code 1.1/assets/parity_exp_c/` (+ parity_sweep.csv + parity_sweep_data.npz) and
EMBEDDED in the Experiment tab. Scripts: /tmp/parity_old_sweep.py + /tmp/parity_new_plots.py. NB Γ=1 ⇒ box collapses
⇒ all methods = IPW (0.675); R-OW tops for Γ>1; regret methods hug control as Γ grows.

**Import caveats (learned):** (1) `code/common` and `code 1.1/common` are BOTH top-level `common` packages — they
COLLIDE in one process; never import old `srpo` and new `common` together (cross-tests must run in separate
processes). (2) Method folders have hyphens (`R-OW`) so they're NOT importable as packages — load method files by
path via `importlib.util` AND register in `sys.modules[name]=mod` before `exec_module` (else the `@dataclass` +
`from __future__ annotations` fails to resolve). The eventual runner must do this.

**`results/SAVING.md`** — authoritative "save everything" spec: layout `results/<experiment>/<Method>/<Cap>/gtrue-<γ>/`
with Γ & seed as **keys inside files** (`…__G<Γ>__seed<s>`); sections A–H (config, raw draws, optimal policy +
**dual-semantics-labelled** duals, extended Shapley/KNN policy, train/test realised+expected outcomes, reference
ceilings, Rosenbaum, figures+data) + per-experiment checklist. The dual fields (mu/nu/beta/gamma_dual/theta) are
field-overloaded across methods → must store an `obj_kind`/semantics label. See [[save_all_artifacts]].
