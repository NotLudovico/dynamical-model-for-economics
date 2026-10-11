---
name: reference-msb2024-facts
description: Full MSB 2024 target list read from the paper (arXiv:2404.15226 v3) — zeta_q values, sigma/sigmabar inverse-gamma b=4.6, rescaled-growth tails, persistence, correlation lead
metadata:
  type: reference
---
Read in full 2026-10-11 (pdftotext). Compustat 1961-2020, 25 equal-count bins by lifetime mean normalised size Sbar (~970 firms each), vol = sqrt(pi/2) MAD.
- E[sigma^q|S] ~ S^-zeta_q over 1e-3..1e1: zeta = 0.20(.01), 0.39(.02), 0.51(.03), 0.58(.04). Largest bin sits ABOVE the line. Mean vol 1.218 (smallest bin) -> 0.219 (largest).
- sigma/sigmabar(S) collapses across all 25 bins, unimodal (bimodality test rejects a hump), Modified Inverse Gamma with tail exponent -(1+b), b = 4.6 (robust: SD instead of MAD, >=20 growth rates, December fiscal year).
- Rescaled growth (g - gbar_i)/sigma_i: fat, stretched-exponential tails (GSE fit z ~ 0.38), universal across size, NO Gaussianisation with size; if anything large firms less Gaussian.
- Multi-year (2-4y) rescaled growth: tails fatter than iid convolution of 1y -> persistence.
- Zipf is cited (Axtell) but not re-measured in the paper.
- Their lead (not a model): correlation of a firm's growth with the aggregate, and between similar-size firms, grows with size; they surmise this is the missing ingredient (Herskovic et al. 2020).
Related: [[project-msb-protocol-spells]], [[project-scope-model]].
