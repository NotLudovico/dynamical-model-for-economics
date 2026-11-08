---
name: project-bouchaud-mezard-test
description: "2026-11-08 mean-field Bouchaud-Mezard (x' = x sigma xi + J(m-x)) fails every MSB fact except Zipf; zeta_q slightly negative"
metadata:
  node_type: memory
  type: project
  originSessionId: 80ddcf9f-199e-4663-90be-de6e7135b404
  modified: 2026-11-08T19:55:50.758Z
---

Tested 2026-11-08 (scratchpad script, not in repo; uses notebooks/economy.py lives/zeta_bins/hill). sigma=0.3/yr, J set by Zipf target 1+2J/sigma^2 in {1.05,1.5,3}, N=20000, windows 12 and 60 yr as lives (BM has no entry/exit).
- Zipf OK (Hill 0.92/1.45/2.71).
- zeta_q = -0.004..-0.15: volatility flat or RISING with size. Reason: multiplicative noise is size-blind; the only size term J m/x is a deterministic inflow that pins small firms (var g ~ sigma^2 x/J when J/x >> 1/yr).
- sigma/sigmabar: no heterogeneity, b 12-27 (sampling noise only) vs 4.6. Rescaled growth Gaussian (excess kurt ~0 vs ~5). No persistence.
Sparse-network BM argued (not run) not to help: small firms track the neighbour sum, vol ~ sigma/sqrt(k), still beta <= 0.
Related: [[reference-msb2024-facts]], [[project-zero-sum-model]].
