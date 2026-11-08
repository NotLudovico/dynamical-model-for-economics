---
name: project-noise-vs-interactions
description: "Ablation (2026-10-13) — sqrt(DS) noise alone gives the decline (granular, ratios = q); interactions alone give no decline but the bend and large-firm fat tails"
metadata:
  node_type: memory
  type: project
  originSessionId: a2564485-6e81-41e0-a6d4-d93f2a921ebd
  modified: 2026-10-13T15:47:50.840Z
---

Ablation on reference economy seed 0 (N=16000, n_min=20), one-firm runs from the measured field:
- full: zeta_1 0.28, ratios 1.80/2.41/2.85, b 6.6, kurtosis 4->20 by fifth
- noise only (each node's time-mean field frozen, sqrt(DS) noise, turnover): zeta_1 0.62, ratios 2.01/3.04/4.08 (fixed shape), b 10.5
- interactions only (measured moving field, no noise, exit below 1e-3): zeta_1 ~0 (hump), zeta_q<0 for q>1 (spread grows with size), kurtosis 0.2->7.4

User loved the 3x4 figure (rows = full / noise / interactions; cols = decline, zeta_q vs q, v/E[v|S] CCDF, rescaled-growth CCDF). Script was in session scratchpad (msb_rows.py), not yet in repo.

**Why:** reframes the paper: sqrt(x) demographic noise = granular term (slope ~1/2, ratios q); interaction floor sets level ~0.2 and its heterogeneity gives the bend. Differs from Bouchaud–Mézard (noise ∝ x, no size dependence).
**How to apply:** frame the abstract/intro as "granular decline + heterogeneous interaction floor"; don't claim interactions cause the decline. Related: [[project-demographic-noise]], [[reference-msb2024-facts]].
