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
- Granular model prediction (Prop. 5 / Prediction 2, two granularity sources, subunit tail mu, P(K) tail alpha, 1<alpha<mu): E[sigma^q|S] = C1 S^(1-mu) + C2 S^(q(1-mu)/mu) + C3 S^(alpha-mu). For q>=2 the poorly diversified firms (C3) dominate, so zeta_q = mu-alpha ~ 0.2 for ALL q>=2 (ratios ~1, far too bent). The data rejects it. Data instead look like the well-diversified term, zeta_q = q*zeta_1 -> 0.4/0.6/0.8; MSB blame the shortfall (0.51, 0.58) on subdominant corrections from the incipient tail of sigma/sigmabar. So granular is NOT "ratios = q": the asymptotic, poorly diversified regime gives ratios ~1, and only the well-diversified term gives q.
- Their GSE parameters (Table 1, eq. 15) do NOT reproduce their own plotted fit (they give ~1e-2 at |z|=8, the plot ~3e-4). Use the plotted curves instead: scripts/msb_curves.py reads Fig. 5 (whole sample) and Fig. 7 (25 size bins) from the PDF's vector paths into data/msb_rescaled_growth.npz. LOO-rescaled excess kurtosis on |z|<=8 by fifth: 4.2, 5.6, 5.1, 5.4, 5.7 (model at n>=5: 3.0 -> 5.7, Gaussian null 2.5 -> 0.1).
Related: [[project-msb-protocol-spells]], [[project-scope-model]].
