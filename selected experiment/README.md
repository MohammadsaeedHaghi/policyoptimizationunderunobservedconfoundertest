# Selected experiment — exp_owgap

The experiment chosen for the paper: **confounding-robust policy optimization**, where the
**O-W methods (odds-box ∩ Wasserstein)** beat every other method on a DGP with hidden confounding.
(This folder is **owgap-only** — `exp_nmcap` was removed.)

| method | exp_owgap (hidden-S confounding) |
|---|---|
| naive IPW-X-X | **0.33 — fails** ❌ |
| naive AIPW (DoublyRobust-X-X) | 0.64–0.65 |
| **IPW-O-W** | **0.78** ✅ |
| **DoublyRobust-O-W** | **0.78** ✅ |

(realised E[Y] at the operating point, uncapped, c_ε = 1.0; oracle = 0.847, never-treat = 0)

---

## exp_owgap — hidden vitality (the experiment)

Clinical DGP: aggressive therapy under an **unobserved vitality** S that dominates outcomes.
Both O-W methods beat **every** other method by a large gap, peaking at small Γ (2–3), in both regimes.

- **X**: discrete, 7 levels, Uniform on linspace(−1, 1, 7)
- **S ∈ {−1,+1}**: unobserved, P(S=+1|X) = σ(10X)  (strong S–X coupling)
- **T**: e(X,S) = clip(σ(0.8S − 2X), .02, .98)  (moderate S-selection → matched Γ ≈ 5, small)
- **Y**: μ₀ = 8S, μ₁ = 9S + 1.5X, noise N(0, 0.6²)
- **CATE(X)** = (d₁−d₀)·E[S|X] + 1.5X = (2σ(10X)−1) + 1.5X  ⇒ **treat iff X > 0**

**Strongest claim (uncapped, c_ε=1.0):** both O-W beat the best naive competitor (AIPW, 0.64) by
+0.11 to +0.18 at Γ=2–3. Locked at **20 seeds with paired bootstrap CIs:
IPW-O-W − AIPW = +0.12 [0.09, 0.15]** at Γ=3. Box-only AIPW (DR-O-X) fails → the **Wasserstein term**
is the differentiator.

**Honest caveat:** under a capacity cap the cap rescues AIPW to 0.71, so IPW-O-W carries the clean win
while DoublyRobust-O-W only ties at small Γ. The cleanest, CI-backed win is **uncapped**.

**Key design insight:** because mean_X E[Y(0)|X] = 0, realised value V(π) = mean_X[π·CATE],
**independent of d₀** — so confounding magnitude (→ gap size) and CATE scale (→ collapse speed) decouple.

---

## Folder contents

- **`dgp.py`** — the data-generating process.
- **`build_owgap.py`, `run_owgap.sh`** — original build/run pipeline (N=600).
- **`run_owgap_n1000_2w.sh`, `run_owgap_n1000.sh`, `merge_regimes.py`** — the N=1000 reruns (2-worker, license-safe).
- **Results:** `owgap_results_ce{1.0,1.5,2.0}.json` (N=600, 5-seed), `owgap_uncap_20seed_ce1.0.json` +
  `owgap_uncap_paired_ci.json` (20-seed CI), `owgap_extreme_sweep.json` (extreme Γ×ε). N=1000 results
  (`owgap_results_n1000_ce*.json`) are added here as each ε completes.
- **Charts/plots:** `owgap_charts.json`, `owgap.html`, truth PNGs (`cate/sx/prop/outcome/obs`),
  value PNGs (`val_{uncap,cap}_ce{1,1.5,2}.png`, mean±SD), extreme-Γ×ε PNGs (`ex_ipw`, `ex_family`).

## The report
`../report.html` (built by `../build_report.py`) is a self-contained, owgap-only report with 3 tabs:
**Overview**, **exp_owgap** (DGP truth + Uncapped/Capped sub-tabs), **Limitations** (extreme Γ×ε).
Rebuild it from this folder's data with: `cd .. && python3 build_report.py`.

## What to put in the paper
1. **Main result:** lead with the uncapped value-vs-Γ figure + the 20-seed paired-CI claim.
2. **The result table** (above) as the headline.
3. **Over-conservatism (extreme Γ×ε)** → Limitations/Discussion.

**Caveat to state, not hide:** uniform dominance over every naive baseline *under a capacity cap* is
structurally hard — the cap rescues whichever naive correction is well-specified. Anchor strongest
claims on the **uncapped** regime.
