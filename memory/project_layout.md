---
name: project_layout
description: Workspace layout for the policy-optimization-under-unobserved-confounder research project.
metadata:
  node_type: memory
  type: project
  originSessionId: 85317083-7452-4dfa-ba5f-16d73019ebb5
---

Working directory: `/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/`

Top-level folders:
- `paper/` — three reference papers, each in its own subfolder: `bill/`, `kallus/`, `saeed/`. `paper/kallus/` has `kallus.pdf` + `github link.rtf`; `paper/kallus/repo/` is our local clone of CausalML/confounding-robust-policy-improvement.
- `DGP/` — the user's data-generating-process specs: `DGP - case 1.rtf` … `DGP - case 9.rtf` plus `DGP - discrete exp_a/b/c.rtf` (prose/math definitions of each experiment's DGP), and a few reference implementations (`case8.py`, `case9.py`, `discrete_dgps.py`). These case/exp specs are the source the experiments are built from.
- `code/` — the original implementation (no longer empty).
- `code 1.1/` — the clean restructured sibling of `code/`; the current main codebase (assets/, srpo/, experiments/, application/, index.html). See [[code_1_1_overhaul]].
- `claude.txt` — just a note with the `claude --resume <session-id>` command; not project content.

**Why:** User is doing a research paper on policy optimization under unobserved confounders; reference papers studied (Kallus first), implementation lives in `code/` and `code 1.1/`, all driven by the DGP specs under `DGP/`.

**How to apply:** New implementation goes under `code 1.1/` (the active codebase). Reference material is under `paper/<author>/`. When an experiment needs a DGP definition, the canonical spec is in `DGP/DGP - <case|discrete exp_*>.rtf`. Don't put implementation files inside `paper/`.

See also [[kallus_paper]], [[kallus_repo]], [[code_1_1_overhaul]], and [[firstexp_case4]] (the "Case 4" DGP from `DGP/`).
