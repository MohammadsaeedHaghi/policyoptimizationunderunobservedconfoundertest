---
name: html-raw-lessthan-bug
description: "A raw '<' in MathJax/text (e.g. Y(1)<Y(0)) silently breaks HTML parsing and collapses the panel; validate by headless render, not just string counts."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b9b1bd8f-6c5e-40f9-8b50-6e13ed6d978f
---

A single **raw `<` in math/text** inside index.html — `\mathbb 1[\,Y(1)<Y(0)\,]` (kallus_repro DGP block) —
silently broke the page: `<Y` opened a **bogus HTML tag** that swallowed content up to the next `>` (incl. a
`</div>`), cascading to collapse BOTH new Kallus Experiment subtabs to `offsetHeight=0` (rendered totally
**empty** in the real browser). exp_c and earlier panels were fine (they precede the bad char).

**Why:** Why this is insidious — the standard validation (`count('<div')==count('</div>')`, inline `\(`/`\)`
balance) **PASSES** because `<Y` isn't `<div`. So a green validation does NOT mean the page renders. The user
saw "empty subtab / no plots" for several turns while I wrongly blamed browser cache.

**How to apply:**
1. In MathJax `\( \)` / `\[ \]` and any HTML text, NEVER use a literal `<` (or `>` before a letter). Use LaTeX
   `\lt` / `\gt`, or HTML `&lt;` / `&gt;`. Grep new HTML: `grep -nP '[A-Za-z0-9)]\s*<\s*[A-Za-z\\(]'` and
   `grep -n '<[A-Z]\|<\\'` (exclude `<script>` JS like `i<5`).
2. After adding/editing any panel, **verify it actually renders** with headless Chrome, not just string counts:
   `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-sandbox
   --virtual-time-budget=6000 --screenshot=out.png "file://.../index.html"`. To check a hidden subpanel, copy the
   file, set its tab+subpanel classes to `active`, and probe `getComputedStyle(el).display` + `el.offsetHeight`
   (offsetHeight=0 with children>0 ⇒ broken layout / float collapse / bogus-tag). Panel-only render (head+that
   panel) isolates content-vs-context. See [[keep_html_in_sync]], [[application_app]].
