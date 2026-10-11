# Research status (2026-10-11)

Goal: a journal article on a disordered-interaction model of firms that explains the size-volatility
facts of Moran, Secchi & Bouchaud (2024, "MSB"). The thesis is submitted; this is follow-up work.

## Main model

The relative GLV with demographic noise and turnover (`scripts/demographic/`), at alpha = 2.5,
sigma = 1.70, D = 0.1, C = 200. The deterministic relative GLV (thesis model, lambda floor) is kept for
the exact one-firm theory (kill rule, lifetime law) as the noise-free limit.

## What it reproduces (N >= 16000, lives with >= 5 growth rates, all 25 bins)

- beta = 0.20 and zeta_q = 0.20, 0.38, 0.54, 0.66 (MSB 0.20, 0.39, 0.51, 0.58), with no hump: an
  all-bins fit as MSB do it.
- Correlation of firm growth with the aggregate and between similar-size firms rises with size.
- The multiscaling has a derived, tested mechanism: with v^2 = v_F^2 + B(Sbar),
  zeta_q = (q/2)<w>_q, w = B/(v_F^2 + B), so the ratio curve bends below q iff the fitness floor
  differs across firms; v_F^2 is proportional to the degree (fitted exponent 1.00), the two-term law
  predicts ratios 1.86, 2.56, 3.06 against 1.80, 2.42, 2.88 measured, and a regular graph gives ratios
  near q (`multiscaling_theory.py`).
- Theory: renewal phase line sigma_c(D) and turnover, one-firm reduction, volatility law.

## What it does not

- The decline spans about 2 decades, not 4; no Zipf law (Hill 4-5), sizes split into a newborn and
  a plateau mode. The paper argues this is structural (self-limitation ties the volatility decline
  to a size cap).
- Large firms are about twice as volatile as in Compustat (0.41 against 0.22).
- MSB's baseline sample (>= 2 growth rates) gives ratios above q.
- b of sigma/sigmabar: 6.9 at >= 20 growth rates against 4.6; the 4.4 at >= 2 is an estimator effect
  (`b_null.py`).
- Rescaled-growth tails fatten with size; multi-year persistence only for the largest firms.
- Aggregate volatility does not fall as N^-1/2 (about 0.07 per step at N = 64000).

## Paper

`thesis/standalone/paper/paper.tex` was restructured on 2026-10-11 around the deterministic model,
before the switch to the demographic model; it needs a rewrite with the demographic model in the
data section and the multiscaling mechanism as the central result. Red `\todo{}` marks numbers that
need a reproducible script. No TeX is installed on the workstation, so it has not been compiled.

## Next steps

1. Production run at the reference point (N = 32000, 6-8 economies) for the paper's data section.
2. Rewrite the paper around the demographic model.
3. Theory still missing: field statistics above sigma_c (two-time theory with noise), beta(sigma, D),
   co-movement rising with size, the 2-decade range.

Alternatives that were tried and dropped are summarised in `scripts/screen/README.md`.
