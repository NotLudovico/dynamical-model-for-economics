---
name: project-region-mu-sigma-d
description: "2026-11-08 — facts region as a slab above sigma_c(mu, D); D enters via sigma_c, slope c(mu)=0.30..0.49; slab width empirical"
metadata:
  node_type: memory
  type: project
  originSessionId: 02a29c65-d455-4670-9d95-e276e9fa27c3
  modified: 2026-11-08T20:33:44.191Z
---

User asked whether mu, sigma, D can be linked into a region where the model works. Answer given: region = slab
d_lo <= sigma - sigma_c(mu, D) <= d_hi, d in [0.18, 0.25] (zeta_1 0.15-0.25 at n>=5), mu <= -1.
- Renewal closure, reference graph (alpha 2.5, N 32000, C 200), D 0.05/0.1/0.2: dsigma_c/dD = 0.296, 0.333, 0.368,
  0.403, 0.434, 0.494 at mu = -1, -1.25, -1.5, -1.76, -2, -2.5 (~0.30 + 0.13(|mu|-1)). Zero-growth line moves half as
  fast (0.14-0.23), so larger D makes zero-growth economies shallower. Script: scratchpad sigc_grid.py (not in repo), ~1 min on 12 cores.
- Evidence D acts only via sigma_c: two equal-depth pairs in Table tab:robust (mu -1.76) match to 0.006 in zeta_1.
- Equal-depth test run 2026-11-08 (strip protocol, depth 0.21, D 0.05/0.1/0.2, 2 seeds; scratchpad equal_depth.py/.npz):
  zeta_1(n>=5) 0.180/0.192/0.187 at mu -1.25, 0.213/0.227/0.226 at mu -2.5 (fit says 0.209; seed spread 0.09 at -2.5).
  Residual D effect <= 0.015 vs 0.07-0.10 at fixed sigma -> zeta_1 collapses on depth. Bend does NOT fully: zeta_4/zeta_1
  rises with D at fixed depth, +0.2 (n>=5), +0.35-0.42 (n>=20) over D 0.05->0.2 (larger D -> larger demographic share w).
  fbar at fixed depth rises with D as the slower sigma_0 shift predicts (+0.08, +0.12).
- Missing link for a fully derived region: depth -> field-motion amplitude (two-time closure, open); floor -> zeta_q chain is done.
Related: [[project-depth-collapse]], [[project-vf-floor-theory]], [[feedback-no-sensitivity-scans]].
