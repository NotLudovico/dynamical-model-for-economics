---
name: project-slot-model
description: Oct 2026 candidate GLV replacement — firm = k slots of killed-GBM revenue links; solvable, results and flaws
metadata:
  type: project
---
Proposed 2026-10-05 after user said GLV "not cutting it" and asked for something solvable. Scratchpad only (session 3a795932): slot_model.py, slot_moments.py.
Model: firm i has k_i slots (degree, tail alpha); each slot holds link w, d ln w = m dt + s dW, dies at rate delta, reset to w0=1; S_i = sum w.
- Link law exact: ln w asymmetric Laplace, tail mu = (sqrt(m^2+2 delta s^2) - m)/s^2 (sim 1.243 vs 1.242).
- E[H|k] = k int t E[w^2 e^{-tw}] phi(t)^{k-1} dt matches sim; sd(g) ~ sqrt((s^2+delta) H) within ~10%.
- Mean-H asymptote is beta=(mu-1)/2 (0.12 at mu=1.24, quadrature to k=1e8), NOT Wyart-Bouchaud 1-1/mu (that is the typical firm). Over k=1e3..1e5 at mu=1.25: zeta_q .20 .35 .44 .50, ratios 1.7 2.2 2.5 — a slow crossover, so effective exponents.
- Current-size protocol: alpha>mu -> big firms are single-link dominated, vol RISES with size (alpha=3.5: zeta_1=-0.27). alpha<mu: flat at small S (k=1 firms have H=1), local beta 0.15-0.3 above — same wrong curvature vs Fontanelli as GLV.
- Core flaw: a lucky link makes a firm both big and concentrated. Candidate fix: cap links by customer size (real network role). See [[project-size-volatility-exploration]], [[reference-fontanelli2025]].

v2 customer-size cap tried 2026-10-05 (scratchpad slot_v2.py): S = h + c Pi S, Pi = input shares from killed-GBM u. FAILED under current-size protocol:
- h (household link) killed GBM: H ~0.5 flat in S, vol rises at top (zeta_1 -0.08..-0.2). Uncapped final-demand link dominates again.
- h constant: H falls with S (0.76->0.29) but vol RISES (0.06->0.11): big firms are big via noisy b2b sales, small via calm household demand.
- Pattern across GLV, v1, v2: when size comes from multiplicative luck, size conditioning puts the luckiest/most concentrated/noisiest-composition firms on top.
- Also v1 dynamic growth moments (alpha=1.1) gave zeta_1 ~0.05, weaker than static E[H|S] (~0.15): dominant-link deaths fatten |g|.

Niche-GLV test 2026-10-05 (scratchpad niche_glv.py): GLV niches on a 2D random geometric graph (torus, C~30), firms = blobs vs random sets.
BLOCKED on dynamics, no beta result yet: relative GLV fluctuates only for sigma~1.3-1.5 at N=2000 (seed-fragile), condenses at N=8000 (a region wins under the global normalization); standard GLV (lam=1e-4) diverges (NaN) at sigma>sqrt2 with mixed-sign couplings.
