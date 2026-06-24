---
name: control_arm_philosophy
description: Design philosophy — control (arm 0) is the uncapped, never-optimal fallback, but must stay a SOLID (non-weak) arm.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

Control (arm 0) is always the **uncapped** arm and should be **essentially never the optimal arm** (it need
not win any meaningful region) — BUT its outcome VALUE must stay **solid, not weak**. "Never-optimal" and
"weak" are different things, and conflating them breaks the experiment (see the correction below).

**Why:** control is the do-no-harm baseline and the capacity-overflow absorber — when the capped specialist
arms are rationed, the displaced mass goes to control. So it being rarely/never-best is by design. HOWEVER,
the robust methods (RW, Kallus do-no-harm) HEDGE TOWARD CONTROL under high Γ; if control is genuinely weak,
that hedging is penalised and a committing non-robust method (Direct IPW) can beat RW. So "RW clearly best"
REQUIRES control to be a solid arm. The user made this explicit (2026-06-08): first "I dont want control to
be the STRONGEST arm — it has no cap constraint", then, once the tension was surfaced, chose option C
**"Clear RW best only (control is a solid fallback)"**.

**How to apply:** keep control **uncapped** (cap=1.0) and tune it to be a CLEAN, SOLID baseline that wins
only a tiny region — e.g. [[conti2d_study]] final: θ0=1.10 ⇒ control μ≈0.75, optimal in only ~2.2% of X
(small centre), with c_0=0 so control is NOT itself confounded. Do NOT make control the strongest arm, and
do NOT make it weak/worst either — both break the intended ordering. (The earlier "control may be the WORST
treatment, don't worry about its value" stance was WRONG for robust-method comparisons and has been
superseded.) Frame control in figures/text as the clean, uncapped, never-optimal-but-solid fallback.
