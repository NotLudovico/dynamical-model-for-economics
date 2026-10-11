---
name: project-size-volatility-exploration
description: Oct 2026 findings on why the model has no 4-decade -0.2 volatility law (hump, bimodality, trade-off)
metadata:
  type: project
---
Findings 2026-10-04/05 (scratchpad experiments, not in repo; only change: optional K arg in relative_glv/model.py integrate):
- Spell-panel E[v|Sbar] is a hump; decline spans < 1 decade; MSB's -0.2 spans 1e-3..1e1. Larger-half fit is cherry-picking.
- Hump = Sbar conditioning: small-Sbar spells are calm brief floor crossings; by CURRENT size small firms are most volatile (monotone, no hump).
- Sizes bimodal (floor ~3e-4 vs plateau ~3x mean, ~5% time between): logistic plateau + quenched drift; small firms are transits.
- lambda=0 freezes the economy; Pareto K condenses onto top firm; products->firms portfolio gives beta=1/2 (CLT).
- sigma sweep alpha=3.5: sigma 2.6 fills sizes (flat log density, 5 decades) but v RISES with size; sigma>=3 condenses/freezes. Trade-off: self-regulation gives both the volatility decline and the bimodality.
- Overnight scan (103 runs, gamma=0, alpha/sigma/mu/lambda/C/N): NO regime gives everything; beta = 1.12*(size-density slope)+0.02, corr 0.92 — steeper decline always means more bimodal sizes. Report: scratchpad overnight_report.md (session e90db744). See [[feedback-minimal-parameters]], [[project-msb-protocol-spells]].
- 2026-10-05, MSB-style fit over ALL spells (25 equal-count bins, data/headline_spells): beta = -0.07/-0.05/-0.05/-0.02 at alpha 2.5/3.5/3.5(lam 1e-2)/6 over ~2.7 decades. Local slope by fifths: -0.7..-1.1 (rising), ~-0.1, then +0.1..+0.75 at the top. The paper's beta 0.19-0.55 is the upper-half slope only; the user noted "in this model beta = 0" and was right.
