---
name: project-lifecycle-theory
description: Life-cycle theory of beta/multiscaling under the unbalanced (spell) panel — single-firm surrogate, lifetime identity, note at thesis/standalone/lifecycle_theory (2026-10-02)
metadata:
  type: project
---

Written 2026-10-02 as an uncommitted standalone note, `thesis/standalone/lifecycle_theory/lifecycle_theory.tex`. Scripts are in `scripts/lifecycle/`, data in `data/lifecycle/`. Key results:
- Single-firm reduction: dlnS/dt = Phi - S + lam(1/S-1), with Phi_i = m_i + f_i*eta (Gaussian, ACF (1+L/tau)e^{-L/tau}) and the measured (m_i, f_i). With no network it reproduces beta and zeta ratios under BOTH panels, plus the slow window convergence (alpha=6, 3.5, 2.5).
- Exact lifetime identity for complete spells: sigma = sqrt(pi/2)(2 ln(Smax/Sthr) + V_p)/n. The rise and fall carry 74–100% of sigma on the decline branch. Model-free bound: sigma >= sqrt(pi/2)*2 ln(Smax/max(Sa,Sb))/n.
- zeta_q ≈ zeta_q[1/n] - q/ell, verified to 0.02. The bend comes from Var(ln n|S) growing with size, a general criterion with random factor X (sqrt(kappa) or ell/n).
- Rice lifetime law with merging factor ~2. Lifetime ~ exp(S^2/2f^2), so beta is an effective exponent; the crossover moves as sqrt(ln T).
- Heavy-tail flattening is caused by the spread of the amplitude f (ablation: homogeneous f restores 1.90, 2.67, 3.29).
- Caveat: model firms enter at ~1e-2 of the mean size, so the life-cycle share is probably overstated versus Compustat. Empirical predictions: survivorship raises beta; lifetime bound; age/entry-exit trimming moves the ratios toward q.
- (m,f) law (Prop. "Fitness law" in the note; `scripts/lifecycle/mf_law.py`): m|kappa ~ N(1 - gbar - mu kappa Bbar, sigma^2 kappa Q_T), f = sigma sqrt(kappa (Q0 - Q_T)), rho from Q(L), all degree-weighted averages. The cavity check is near exact (standardised field sd 1.00; f within ±13%; reproduces skew(m) = -4.5 at alpha=2.5). Closed chain with HDMFT order parameters: alpha=6 beta 0.403 vs 0.422 (ratios within 0.03), alpha=3.5 0.285 vs 0.214, alpha=2.5 0.216 vs 0.198. The remaining gap is finite-C amplitude spread at fixed degree (not derived).
- The user wants a mechanism paper rather than an exact fit, and found the theory direction worthwhile.

Related: [[project-msb-protocol-spells]], [[project-article-goal]], [[feedback-no-window-dependent-beta]].

Review of 2026-10-02 (not yet fixed in the note):
- Eq. lifetime-rep holds for the transient proxy 2√(π/2)ℓ/n, not for σ. At λ=1e-3 σ has ζ_q = .316 .593 .824 1.01 against .427 .718 .931 1.10 for the transient. The lifetime bend overstates σ's bend 2.5–3×, because the plateau part dilutes Var(ln n).
- Spell β still drifts with T at α=6 (√ln T crossover).
- MSB growth rates are overlapping annual differences sampled quarterly, so the bound must be adapted for the Compustat test.
- The HDMFT τ deficit (2.41 vs 3.51) is not a cancellation: the law with τ=3.51 gives β 0.432.
- β precision is limited by the peak-bin jump; see [[reference-decline-beta-peak-instability]].

IPO-entry test (2026-10-02, scratch scripts only, nothing in the repo). A spell is listed from its first reach of S_ipo until it falls below 8λ. Fit as MSB do: 25 bins, mean vol, all bins, no peak bin.
- Full model, α=6, σ=1.75, S_ipo=0.1: β 0.192 (sd 0.019), ratios 1.85 2.49 2.89, close to the data (0.20; 2.0 2.6 2.9). β rises to 0.26 at S_ipo=0.3 and 0.32 at 1.0. α=3.5 fails (β 0.04–0.08).
- The shape is wrong, though: the lower half of the range is flat (slope −0.08) and the upper half steep (+0.64) over 1.8 decades, a hockey stick rather than a power law. The data are near-straight and flatten at the top.
- Without IPO entry the all-bin β is about 0. The hump comes from the listing rule (spells near the threshold) and moves with the threshold.
- S_ipo should be calibrated from Compustat IPO size over mean size; treat it as data, not a free knob.
- Shape test (surrogate, α=6, IPO 0.1): the β-note adiabatic law f·A(S̄) does NOT explain the IPO curve. It predicts σ 0.74 at S̄=0.07, where the measured value is 0.25 (the low end rises, then is flat to S̄≈1, then falls). Measured/plateau ratio grows with spell life (0.34, 0.76, 1.10 at life 5, 9, 14 against τ=3.5). Lead: short spells subtract their own mean growth and the fitness hardly changes within a spell shorter than about 2τ. Next is a finite-life correction plus the transient term; plateau+transient as a sum double counts.

Spell-theory push (2026-10-04; theory now unified in thesis/standalone/theory/theory.tex, which supersedes the five notes):
- Step 1 done (lifetime law, in the doc; `scripts/lifecycle/kill_rule.py`). At lam=0 the one-firm eq is exact: 1/S = int exp(-[A(t)-A(t')]) dt', A = int Phi. Near a downcrossing S = inverse Mills ratio, S_dc = sqrt(2s/pi). Kill rule: a dip ends the spell iff its area > ln(sqrt(s/2pi)/S_thr); it is 96–98% accurate. Lbar = Phi_N(u)/(nu(u) P_kill(u)), with P_kill from the Slepian model; within 3% for u>=1 with no simulation input. Var ln L -> pi^2/6 at u=1.5. -ln P_kill ~ u^{4/3} asymptotically (only for u>~4).
- Step 2 half done (`scripts/lifecycle/spell_anatomy_merges.py`). The adiabatic sqrt(D)/S plateau term is ABSENT for spells <~8 tau: a spell is a parabolic hump in ln S, with ln(Smax/Sbar) = 0.5 ln(4 ell/pi) ~ 1. Each non-killing dip adds ~0.35–0.4 * 2 ell to TV. The hump+merge law gets per-spell v to 4–10%, but the zeta_q are all ~0.08 too high, a q-independent offset that wrecks the ratios because zeta_1 is small. The missing variation sits only in the top size quartile (+0.16 * 2 ell): wiggles of large firms whose fitness never reaches zero. That is the open sub-problem.
- Decided: the theory targets the decline-branch curves E[v^q|Sbar] and local exponents; drop the fit-range question; keep the model threshold 8 lam (no IPO modelling).
