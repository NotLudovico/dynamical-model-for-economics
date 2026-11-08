---
name: project-nmin5-switch
description: "Done 2026-11-08: paper compares with data at n_min=5 (MSB like-for-like), theory tests stay at n_min=20"
metadata:
  node_type: memory
  type: project
  originSessionId: 30c62eca-8804-4ebd-9fd1-ff89cf112b5e
  modified: 2026-11-08T15:00:32.234Z
---

MSB growth rates are annual diffs rolled over quarters (g = ln S_{t+4} - ln S_t), so their "at least 20" cut = 6 years = 5 non-overlapping annual growth rates; their mean 46 ~ 12 years. n_min=5 is the like-for-like cut.

Switched on 2026-11-08: the production run (8 economies, N=32000, ~25 min on the 12-core Mac) was re-simulated locally into data/demographic/robustness/prod_seed*.npz; facts table, Fig. facts and appendix numbers now at n_min=5 (zeta_1 0.209, ratios 1.92/2.69/3.28, b 5.2). Theory tests stay at 20. The other robustness jobs in that folder were not re-run here, and notebook 06's stored outputs are from the old machine.

**Why:** the old justification compared overlapping (MSB) with non-overlapping (model) counts.
**How to apply:** don't revert to n_min=20 for data comparisons; quote n_min=20 only for theory tests.

Related: [[project-msb-protocol-spells]], [[reference-msb2024-facts]].
