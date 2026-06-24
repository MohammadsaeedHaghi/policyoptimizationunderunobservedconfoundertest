---
name: colorful_method_boxes
description: "User preference: explain methods (objective, uncertainty set, how-solved, args) inside colourful theme-respecting boxes, not plain paragraphs."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 58aeb24f-db15-4091-9dfb-a7f743c17425
---

When explaining a method (its objective, uncertainty set, how it is solved, arguments, notes…) in the HTML
report, present each section inside a **colourful, theme-respecting box** rather than plain `<p>`/`<h3>` text. The
user explicitly likes this — "I think it will organize everything."

**Why:** boxes visually segment objective vs uncertainty-set vs algorithm vs args, making dense method docs scannable.

**How to apply:** use the `.mbox` system in index.html (CSS near the top `<style>`): a tinted rounded box with a
coloured left border + a small uppercase coloured label. Kinds (theme-consistent accents):
`mbox-obj` (objective, indigo) · `mbox-set` (uncertainty set, amber) · `mbox-solve` (algorithm/LP, teal) ·
`mbox-args` (arguments/returns, slate) · `mbox-note` (notes, rose). Colour stays in the slate family theme.
Pair with the colour-by-family / shape-by-uncertainty plot scheme — see [[method_naming_plot_style]]. Also the
top-of-Overview `.lbx` legend box is the model for "pretty modern themed box".
