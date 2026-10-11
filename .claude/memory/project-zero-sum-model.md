---
name: project-zero-sum-model
description: "2026-10-11 radical-model test — deterministic zero-sum exchange x_i' = sum_j W_ij (x_i x_j)^a / m^(2a-1), W antisymmetric, with entry/exit"
metadata:
  node_type: memory
  type: project
  originSessionId: 46b70a26-a69c-47ac-b2e6-f869c9251fc0
  modified: 2026-10-11T04:36:33.870Z
---

User asked for a radical new model and wants NO external noise (deterministic chaos only). Scripts: scripts/zerosum/simulate.py (args: kind N T floor x0 out a), scripts/zerosum/analyse.py; output in data/zerosum/ (gitignored).

Why: GLV self-limitation -x/m both makes volatility fall with size AND caps sizes (wrong curvature, bimodal). General point: with x_i' = x_i F_i(x) and F_i independent of x_i (mass action, no self term), volatility is size-blind (beta = 0). Options: self-limitation (cap), flux exponent a (beta = 1-a, one knob — user found it "interesting"), demographic noise (user dislikes noise).

Results (entry at 1e-3, exit at 1e-4 of mean, entrant gets fresh couplings):
- a=1 (zero-sum GLV, gamma=-1, mu=0, no self term): half of species die without entry. With entry: sizes span 4.3 dec, flat log density, no bimodality; growth symmetric, kurtosis 5.4 (alpha 2.5) / 7.6 (2.2); NO size-volatility (flat current-size, life-panel beta -0.1); Hill tail 3.1, no Zipf.
- a=0.8 / 0.6 (alpha 2.5, N=2000): current-size vol falls with local slope ~1-a in the upper half (0.17 vs 0.2; 0.42 vs 0.4), flatter below. But life-panel (MSB) beta ~0, ratios inverse (2.4 4.3 6.5 at a=.8). Churn 0.12-0.16 births/node/time, median life ~16-18: every life is an entry->exit transit, which swamps the plateau. No Zipf (Hill ~4), top firms mean-revert (mean g -0.46).
- a<1 numerics: x' ~ -x^a hits 0 in finite time; freeze firms below the floor inside the RHS or the solver stalls.
Open next: entry size X0 calibrated to data (IPO ~1e-2..1e-1 of mean), check life panel. See [[project-size-volatility-exploration]], [[project-demographic-noise]], [[feedback-minimal-parameters]].
