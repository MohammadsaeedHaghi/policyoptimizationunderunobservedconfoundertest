---
name: html_design_system
description: Unified visual theme + canonical concept-colour map shared by all four README.html files.
metadata: 
  node_type: memory
  type: project
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

All four HTML reports (`code/README.html` + the three per-folder READMEs) share one design system, applied 2026-06-01. Each file's `<style>` block starts with the same neutral base (bg `#f7f6f2`, system/Inter font, white cards with 1px border + soft shadow, dark code blocks `--pre-bg:#1e2330`, formula boxes, hover-able rounded tables) and the same canonical concept palette as `--c-*` vars.

**One colour per concept, matching the gamma_sweep.py figure legend** (this is the core rule the user asked for — spot a colour, know the method):
- Sample Oracle `#111`/`#000`, Direct IPW `#9467bd` (purple), Kallus `#1f77b4` (blue), Kallus+W `#17becf` (cyan), RW-Shapley/RW family `#d62728` (red), RW-KNN `#ff7f0e` (orange), RW (no W) `#bcbd22` (olive), RW-parametric `#8c564b` (brown), `common/` infra `#2e8b3d` (green). Caution/snapshot = amber `--warn:#c0840f`.

**Each page is "owned" by its method colour** via `--theme`: Direct IPW = purple, Kallus = blue (was indigo `#4f46e5` before — fixed), RW = red, top-level hub = neutral slate `#3c4a72`. Header gradient, h2/h3, tabs, table-header tint all derive from `--theme`.

Legacy per-file var names (`--accent`, `--accent2`, `--saeed`, `--kallus`, `--accent4`, etc.) are preserved and repointed onto the palette so inline `style="color:var(--…)"` in the bodies keeps working — bodies were left untouched. A "Colour key" legend card sits in the top-level Methods tab.

When editing any HTML: pull method colours from this map (don't invent new ones), keep `--theme` = the page's own method, and re-run a check that every inline `var(--x)` is defined in that file's `:root`. See [[keep_html_in_sync]].
