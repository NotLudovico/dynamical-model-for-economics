---
name: project-nmin5-switch
description: "Pending (2026-10-28) switch of the paper's facts from n_min=20 to n_min=5; user will run it on a bigger machine"
metadata:
  node_type: memory
  type: project
  originSessionId: 30c62eca-8804-4ebd-9fd1-ff89cf112b5e
  modified: 2026-10-28T17:40:16.675Z
---

MSB growth rates are annual diffs rolled over quarters (g = ln S_{t+4} - ln S_t), so their "at least 20" cut = 6 years = 5 non-overlapping annual growth rates; their mean 46 ~ 12 years. Our n_min=20 is stricter than MSB, n_min=5 is the like-for-like cut.

**Why:** the paper's justification compared overlapping (MSB) with non-overlapping (model) counts.
**How to apply:** the full plan is the `% TODO (n_min = 5 switch ...)` comment above `\paragraph{Numerics and protocol.}` in paper/paper.tex. Facts get recomputed at 5; theory tests stay at 20. Don't re-run it locally unless asked — the user runs it on a more powerful machine.

Related: [[project-msb-protocol-spells]], [[reference-msb2024-facts]].
