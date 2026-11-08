---
name: project-capacity-model
description: 2026-10-11 "option 2" test — second firm variable K_i (Gibrat carrying capacity + Reed exit) in the thesis GLV; both natural versions fail
metadata:
  type: project
---
Context: thesis GLV gives data-like ratios only on <2 decades above the hump (all-bin beta ~0) because the x/m brake that makes beta also caps sizes. User chose to explore a second variable per firm (option 2) over a no-go paper (option 1).
Code: scripts/screen/variants5.py (lvK, lvK_rel, lvK_sat); d ln K = s dW, exit delta=.05 resets x=K=K0<K>, x and K divided by <K> each step. Smoke tests N=2000, T=200 (scratchpad smoke5*.py), not yet in run.py.
- lvK, absolute LV competition x'=x[1-(x+sum a x_j)/K]: giants dominate every firm's field -> yearly vol ~3, 1/3 of nodes active and flickering, Hill 3.5 (no Zipf); mu 0 or .5 overflows.
- lvK_rel (field /m): condenses (top share .53), beta -0.19.
- lvK_sat, saturating pressure sum a x_j/(K_i+K_j): fills decades 1e-3..1 (28/35/16%), Hill .8-.96, but top share .3-.48 and beta ~0 (-.09..+.01), current-size slope .05-.17, vol .15-.4 flat.
Reading: a firm-specific scale makes volatility scale-free (beta ~0); absolute sizes give beta but let giants dominate. Getting beta ~0.2 needs absolute size to enter weakly = an exponent (knob). Related: [[project-size-volatility-exploration]], [[feedback-minimal-parameters]], [[project-sales-network-proposal]].

Also 2026-10-11: cooperative mean (mu -0.5, -1) in the thesis base model, N=2000 smoke (scratchpad coop.py): size ~ degree^1.2-1.9 and Hill ~1 at alpha 2.2, sigma 1.75, but top share .37, yearly vol ~3-5 rising with size, >90% of firm-time below 1e-3, beta < 0; sigma 1.5 freezes. Dead end.
