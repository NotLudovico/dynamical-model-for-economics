---
name: project-field-spread-shape
description: "2026-11-08 — the 'excess kurtosis 2.1' of the mean field at equal degree is a top-class artifact (true +0.33); a Gaussian spread gives the same panel"
metadata:
  node_type: memory
  type: project
  originSessionId: 169defa4-0f36-48e2-ac2d-eda214af742f
  modified: 2026-11-08T21:58:30.901Z
---

Checked on the Linux copies of the reference economies (which differ from the Mac's: 197,636 vs 198,840 lives n>=20).
- Notebook 04's +2.1 pooled 20 quantile classes of both seeds; the top class (kappa 2.6..98) alone has kurtosis +33
  because the mean field falls with degree inside it. Other 19 classes +0.42. Proper: residual about a within-class
  own-degree trend (scripts/two_time_dmft.degree_classes, hubs resolved) / sqrt(r): +0.33 / +0.34 per seed (boot sd .05);
  global cavity residual (m - a - b kappa)/sqrt(kappa): +0.32 / +0.36.
- Surrogate test (f = c sqrt(kappa), same noise): own m 1.84/2.52/3.04 (n>=20); residual reshuffled 1.84/2.51/3.03;
  residual Gaussian 1.84/2.51/3.04. Shape is irrelevant. Notebook 07's "m shuffled in 40 classes" (1.82/2.45/2.92)
  gets extra bend by mixing hubs in its top class — compensates the bend lost to f = c sqrt(kappa).
- What the closure misses is the AMPLITUDE spread at equal degree (CV of f/sqrt(kappa) in class 0.28): own (m, f)
  0.238, 1.80/2.41/2.85 (n>=20; full model here 0.254, 1.80/2.42/2.88); f reshuffled in class 0.232, 1.79/2.41/2.87;
  f at class value 0.283, 1.84/2.51/3.03 (= c sqrt(kappa) case = the closure with three measured numbers, 1.83/2.49/2.98).
  Paper rewritten accordingly (intro, Sec. one-firm, discussion, closure appendix), Linux numbers in \todo{} for the Mac run.
- Field constants here: c 2.204 (Mac 2.155; paper had rounded it to 2.16, now 2.15), frozen f 2.671 (Mac 2.663).

**Why:** the paper said "fat tail (excess kurtosis 2.1)" and "the bend depends on the shape of the spread, whose fat tail a
Gaussian closure cannot produce" (intro, Sec. one-firm, discussion, closure appendix).
**How to apply:** never pool raw degree classes that span decades; standardise residuals about the own-degree trend.
Related: [[project-vf-floor-theory]], [[project-demographic-paper]].
