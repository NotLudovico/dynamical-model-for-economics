---
name: project-depth-collapse
description: 2026-11-06 — zeta_1 and the bend collapse on the depth sigma - sigma_c(mu); zero-growth line crosses the working band for mu -1.25..-2.25
metadata:
  node_type: memory
  type: project
  originSessionId: c0b46eb6-618c-4c62-9837-7b784b0f74eb
  modified: 2026-11-06T16:13:26.828Z
---

User's idea: facts stable along the zero-growth line (small mean-firm growth) -> a region, not a point. Tested at N=16000, D=0.1, alpha 2.5, 2 seeds, full window. Now in notebooks/02_phase_diagram.ipynb section "Where the facts hold" (caches data/demographic/strip*.npz, zero_growth_scan.npz), figure paper/figures/fig_strip.pdf, paper Sec. facts paragraph "Where the facts hold" + abstract sentence. User loved it.
- Simulated zero-growth line sigma_0(mu): -0.5:1.06, -1:1.37, -1.25:~1.49, -1.5:1.60, -1.76:1.68, -2:1.77, -2.5:1.97. Static renewal closure gives it within 0.04 (1.062, 1.372, 1.498, 1.611, 1.719, 1.812, 1.986). fbar rises steeply across it (~3.4 per unit sigma).
- Along the whole line zeta_1 (n>=5) falls 0.39 -> 0.11: NOT stable. Controlled by depth sigma - sigma_c(mu): zeta_1(n>=5) = 0.49 - 1.34*depth, residual sd 0.014 over 27 points (mu -1..-2.5, zeta_1 0.05-0.35); mu adds ~nothing. zeta_1 0.15-0.25 <-> depth 0.18-0.25. Ratios also roughly collapse.
- Zero-growth line sits in that band for mu ~ -1.25..-2.25 with fbar ~ -0.08..+0.16. Told the user: idea partly right (region exists, small growth lands in it) but zero growth isn't the cause and there's no self-organisation.
- At like-for-like n>=5, zeta_1~0.2 gives ratios ~1.95/2.80/3.5 vs data 1.95/2.55/2.90 (q=4 too high); n>=20 gives ~1.80/2.40/2.85.
Related: [[project-vf-floor-theory]], [[feedback-no-sensitivity-scans]].
