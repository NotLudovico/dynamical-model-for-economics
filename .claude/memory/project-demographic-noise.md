---
name: project-demographic-noise
description: 2026-10-05 test replacing the lambda floor by demographic noise sqrt(D S) dW + newborn replacement at size D (scratchpad demo_noise.py)
metadata:
  type: project
---
Model: relative GLV in S units, dS = S(f - fbar)dt + sqrt(D S)dW (mass-conserving), dead node -> newborn S = D, no lambda. Euler dt=0.01, N=4000, alpha=6, sigma=1.75, window 200 (one economy each).
- Economy does NOT freeze. But churn is huge: 0.72 births/firm/time, median life 2, 31% of nodes die >1x per unit time; 9-13% of nodes never die and hold 40-55% of total size.
- Life panel (birth to death): hump GONE, monotone decline over 2-4 decades, all-bin beta 0.10/0.16/0.22 at D=1e-3/1e-2/1e-1; but still flat-ish then plunging above Sbar~0.5 (top steepening persists).
- Size density still bimodal: newborn mode at S~D plus plateau mode ~3x mean.
- Current size: flat high vol (~1.4) below a few D, step down, ~flat 0.7->0.45 above. No -0.5 regime: for S<1 growth = Phi with |Phi|~O(1), so fitness swamps D/S except below ~D. At D=0.1 the drop from S=0.1 to 1 has slope ~-0.4 then flat (Fontanelli-like), but firms below D are all dying newborns.
Structural point: d lnS/dt = Phi - S makes sub-plateau firms Gibrat (size-independent) and only S>~1 damped -> flat-then-steep curvature by construction.
See [[project-size-volatility-exploration]], [[reference-fontanelli2025]].

Batch 2026-10-05 (scratchpad demo_batch.py, N=8000, 8 econ, window 480; summary via `demo_batch.py summary`). Lives >=20 growth rates:
alpha=6: D 1e-3/1e-2/1e-1 -> beta .15/.25/.40, ratios data-like, but shape flat then plunge (local beta 0 -> 1.0 across fifths). User called it "horrible".
alpha=3.5: D 1e-3 beta .08 (hump returns), D 1e-2 beta .13, D 1e-1 beta .20+-.02, ratios 1.86 2.57 3.15, near-straight (fifths .09 .17 .25 .31 .29) — best result so far, BUT only 1.2 decades (Sbar 0.35-6) and lives>=2 gives beta .07. Churn 0.91 births/firm/time; 0.7% nodes never die.
Extended batch (same script, min growth 2/5/10/20):
- beta rises with min life at every point (alpha=3.5 D=.1: .07/.10/.14/.20) -> sample-selection dependence; ratios data-like for >=10/20 everywhere, inverse (>q) for >=2.
- Range of lives = log10(plateau/D): larger D -> straighter but shorter (D=.3: 1.0 dec). Small D -> longer but flat/hump.
- Best: alpha=2.5, D=0.1: near-straight for BOTH >=2 (pooled beta .16, 2.2 dec) and >=20 (.22); per-econ beta .17/.20/.22/.27 for 2/5/10/20; ratios >=10 1.79 2.36 2.75. Less churn (0.68 births, 6% nodes never die, 26% of size). Caveat: alpha=2.5 graph saturated (40% stubs lost at N=8000, C=200).
- Current-size: beta .23-.32 but ratios 2.2/3.5/5 (inverse multiscaling) at all points.
- N=32000, C=200 (alpha=2.5, D=.1): survives; beta .13/.16/.16/.20 (min g 2/5/10/20), fifths at >=20 .15 .19 .23 .24 .16 (flattens at top), ratios ~1.77 2.3 2.6-2.7; 3% nodes never die.
- Neighbourhood (N=8000): D .05/.1/.15 smooth (beta>=10 .16/.22/.25, same shape). sigma is knife-edge: 1.6 -> beta .26-.46 with top plunge (24% survivors, 65% of size); 1.9 -> beta ~0-.07 flat, churn 1.16. Straight shape only near sigma 1.75.
- Why no small firms: long lives need Sbar >~ few D; plateau caps at few x mean -> range ~log10(plateau/D); between, d lnS/dt = Phi (Gibrat). D=.1 straightness = overlap of noise crossover (near D) and plateau damping. Fix candidates: Taylor-law noise S^b (b=1.6 <-> beta .2, extra param), rowavg (rejected), subunits (failed).
Log-space variant (scratchpad strat_batch.py / strat_plot.py): d lnS = (f - fbar - D/2)dt + sqrt(D/S)dW == lambda model with lambda=D/2 plus sqrt(DS) noise; zero unreachable, no deaths, spells above 4D.
- Spell panel: hump back (beta ~0), as in the lambda model.
- Current size, all firms: decline over ~5 decades, beta .21/.28/.35 (alpha 2.5, D .002/.02/.2); ratios data-like at alpha=2.5 (1.85 2.54 3.08 at D=.002) but NONE at alpha=6 (1.98 2.95 3.91). Shape is a staircase: steep at tiniest S (noise), flat ~1 over the floor population (noise sqrt(D/S)=sqrt(2|Phi|) independent of D), steep above ~0.1 (plateau approach).
- Size density still bimodal, dip shallower (factor 2-4).
Other facts at alpha=2.5 sigma=1.75 D=0.1 replacement model (scratchpad facts.py / facts_plot.py / facts.png, 2 econ):
holds: tent kurtosis 2.2; Gibrat flat above S~0.5; exit hazard falls with age (0.62 -> 0.004) and size; |g| falls with age (1.4 -> 0.19).
fails: growth distribution skewed (-0.45, fat left tail only); established-firm sizes left-skewed, cut off ~10x mean (no Pareto tail); all-firm sizes bimodal; median life 0.5 (one step, extreme infant mortality); lag-1 growth autocorr +0.24 at dt=0.5; small established firms shrink on average (data: grow faster).
Theory step 0/1 (2026-10-05): integrator in repo as relative_glv/model.integrate_demographic (+ test in tests/test_model.py, uncommitted).
Step 1 PASSED (scratchpad step1_reduction.py): one firm driven by its measured field Phi_i = 1 - (aS)_i - fbar with fresh noise reproduces the full model's life panel (beta .198 vs .203 at >=2, .376 vs .388 at >=20; ratios within .05; death rates within 2%). So the theory is a one-firm problem: dS = S(Phi - S)dt + sqrt(DS)dW, death at 0, rebirth at D.
Note: economy-to-economy spread at N=8000 is large (deaths/node/time 1.20 vs 0.61 for seeds 0, 1).
Step 2 PASSED (scratchpad step2_constant.py, step2b_exact.py): constant Phi one-firm = 1D diffusion, scale fn s'(y)=exp(-(2/D)(Phi y - y^2/2)).
Exact quadratures match sim <1.5%: P(reach plateau before 0 | D) = int_0^D s'/int_0^Phi s'; mean extinction time T(x0) = 2 int_0^x0 s'(y) int_y^inf dz/(D z s'(z)) dy.
Plateau: var S = D/2, Var g(dt) = (D/Phi^2)(1-e^{-Phi dt}) good for Phi>=1. Linear-Feller 1-e^{-2Phi} only rough; bare e^{-Phi^2/D} not asymptotic at D=0.1.
Step 3 first result (scratchpad step3_adiabatic.py, full model seeds 0+1, 56k lives >=20): adiabatic law var g ~ A + B,
A = var(Delta ln Phi_+) (plateau tracks field), B = D dt <1/S>, gets the SHAPE (slope .255 vs measured .235 over Sbar .19-3.6)
but overestimates sd g by a near-constant factor 1.42-1.51. sqrt B slope ~.40 (noise, ~1/2), sqrt A slope ~.18 (fitness; f grows with degree).
Not adiabatic: 37% (small Sbar) to 5% (large) of life steps have Phi < 0. Next: linear-response filtering of the logistic firm
(OU in ln S with rate Phi) for the amplitude, and a transient term for Phi<0 stretches.
Step 3b KEY RESULT (scratchpad step3_filtered.py): per-life var g = V_det + B_filt predicts sd g within 0.94-1.09 across Sbar .19-3.6.
V_det = growth variance of the noise-free one-firm ODE driven by the life's measured Phi (sqrt ~0.25-0.32, nearly FLAT in Sbar);
B_filt = mean (D/S^2)(1 - e^{-S dt_g}) (OU in lnS with local rate S) -> DdtS^-1 at small S, D S^-2 at large S.
=> life-panel decline is almost all demographic noise; fitness gives a size-independent floor. beta~0.2 = crossover of
v^2 = v_F^2 + D*filter(S)/S around S* ~ D/v_F^2 ~ 1 (Fontanelli-type: -1/2 small, flat large). Predicted slope .268 vs measured .223.
Step 3c (scratchpad step3c_vf.py): field ACF keeps (1+L/tau)e^{-L/tau} shape but tau=1.42 (vs ~3.5 with lambda). Linear-response
V_F = f^2 K(S0) FAILS (falls with size; sim V_F flat ~.25-.32): large firms ride ln Phi nonlinearly (f/m large), small ones spend 37% of steps at Phi<0.
STATUS at wrap-up 2026-10-05: next = (1) write crossover law v^2 = v_F^2 + B(S), S* ~ D/v_F^2, predict beta(D) vs batch D=.05/.1/.15 with v_F measured;
(2) later v_F from nonlinear one-firm problem per degree class. Section planned in thesis/standalone/theory/theory.tex.
Committed 21608d4 on theory-phase-diagram: model.integrate_demographic + test; scripts/demographic/{batch,facts,facts_plot,reduction,constant_fitness,volatility_law}.py; outputs in data/demographic/ (gitignored). Not committed: step3c_vf (failed linear response), strat_* (log-space variant).
Paper draft (2026-10-05): thesis/standalone/demographic/demographic.tex (revtex, 6 pp, compiles clean; refs.bib adds altieri2021, gabaix1999, karlintaylor1981).
Figures from scripts/demographic/paper_figures.py into thesis/standalone/demographic/figures/. Draft, paper_figures.py and the volatility_law.py npz-save edit are uncommitted.
Phase-diagram theory (2026-10-05, scratchpad renewal.py, noisy_fp.py, sc_line.py, sc_check.py):
- Renewal law p(x|h) = s(min(x,D)) m(x)/T(D) exact vs single-node sim (<S>, <S^2>, turnover to ~0.5%).
- Static HDMFT with renewal law (gamma=0, same classes/kernel as hdmft.py; damped fixed-point iteration, a from normalisation):
  stability sigma^2 E[chi^2] Perron = 1 with chi = d<S>_h/dh. sigma_c(D): alpha 2.5: 1.460(.003) 1.462 1.470 1.497(.1) 1.537 1.577(.3);
  alpha 6: 1.399 1.402 1.413 1.450 1.501 1.551. D->0 matches deterministic (alpha 6: 1.400). Quenched vs frozen closure differ <.015.
- Turnover predicted even above sigma_c: alpha 6 sigma 1.75: .716 vs sim .72-.73; alpha 2.5: .63 vs .66-.68. Turnover ~independent of D.
- Growth at sigma 1.75 alpha 2.5 crosses zero between D .2 and .3.
Phase theory now in repo (uncommitted): relative_glv/renewal.py (renewal_moments, solve_renewal_fixed_point, renewal_sigma_c, renewal_zero_growth) + tests/test_renewal.py; scripts/demographic/phase_lines.py -> data/demographic/phase_lines.npz + figures/phase_lines.png.
FC sigma_c(D=.003)=1.4125 (sqrt2 1.4142). Zero-growth sigma_0(D) 1.70 -> 1.75 (D .003 -> .3), above sigma_c (not controlled). Temporal field share: static closure underestimates 2-3x below sigma_c (no collective amplification).
Draft has new "Phase diagram" section (7 pp, compiles clean). patil2026 bib entry breaks revtex bst (arXiv-only) -> citation dropped in this draft.
Committed a073db4: renewal.py + test, phase_lines.py, paper_figures.py, draft tex + refs.bib. Draft figures live in thesis/assets/figures/demographic_*.png (figures/ dirs elsewhere are gitignored); scripts write there.
