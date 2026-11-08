---
name: project-vf-floor-theory
description: "V_F (fitness floor) decomposition 2026-11-06 — floor = plateau core + entry/exit transients; bend comes mostly from life-length heterogeneity, not plateau/degree"
metadata:
  node_type: memory
  type: project
  originSessionId: c0b46eb6-618c-4c62-9837-7b784b0f74eb
  modified: 2026-11-06T11:50:52.295Z
---

Demographic model, reference point, two N=16000 economies, 198,840 lives with n>=20 (scratch scripts were in session scratchpad vf/step1-5.py, not in repo).
- Linearised filter dy' = dPhi - S_F dy reproduces per-life V_F (log-corr 0.91-0.99, mean within 6-25%); adiabatic 2(1-rho)VarPhi/h^2 fails (fast field components).
- V_F = core (plateau, between first/last time noise-free S >= half its median) + entry climb + exit decline. Core falls with size (0.084 -> 0.052 by fifth), ~kappa^0.48; transient share 26% -> 62%. Flat floor = two opposite trends.
- Transient law: V_t ~ 0.38 ln^2(S_med/D)/n (coef stable 0.37-0.42). Life length n explains 26-61% of Var ln V_F at fixed size, degree 14-37%; Var(ln n|Sbar) 0.08 -> 0.94; hubs live shorter (corr -0.4).
- Ratio tests: only-core heterogeneous 1.98/2.94/3.85; only-transient 1.90/2.69/3.33; both 1.85/2.55/3.07 (paper's).
- Measured panel with plateau only (entry+exit cut): zeta_1 0.25 -> 0.38, ratios 1.79/2.37/2.80 -> 1.92/2.74/3.44. Transients both flatten the decline and make the bend.
- Same conclusion as the lambda-era life-cycle theory ([[project-lifecycle-theory]]), which survives in notebooks/03_life_cycles.ipynb (theory.tex and scripts/lifecycle were removed in 7e29ef0). User asked why recover deleted stuff: don't resurrect old files, build on the notebooks.
- Paper currently says "degree heterogeneity is what bends the curve" — incomplete; degree acts partly through life length. Compustat entry at IPO means birth climb from S=D is not observed there.
- Regular graph (sims take 2.5 min, not 1 h; now in data/demographic/multiscaling/regular_seed*.npz): same transient share (25->56%) but floor falls with size (0.073->0.030), w stays >=0.64, s2 0.007->0.052. Top-fifth s2 ratio PL/regular 5.1 = (1-w)^2 factor 2.9 x Var ln V_F factor 2.2. Degree keeps the floor level up for large firms (amplitude + shorter lives); life length is the largest single dispersion source on both graphs.
- Lifetime law: Rice downcrossing rate of 0 from node (m, f, rho at 0.5) matches observed within ~10%; established lives memoryless (CV(n-20) 1.0-1.18); mean n 22 -> 232 yrs with u=m/f; implied P_kill ~0.95 (u~0) -> 0.5 (u~1.5). Logistic transients: entry SS ~ dt h (ln(h/D)-1) (corr 0.5), exit ~ dt[phi ln(h/S_end)+h] (corr 0.9, 15-35% high).
- Window: 480 time units = 960 "years". Compustat-length 60-yr windows give zeta_1 0.43 (n>=20) / 0.28 (n>=5); ratios stable (1.77/2.30/2.63). Paper doesn't disclose this.
- Kill law per dip: Feller branching P_kill = exp(-2/(D J)), J = int dt/S_det over the dip; calibrated on 446k dips (0.725 pred vs 0.722 obs), no fit.
- Gaussian-field one-firm surrogate: node (m, f) + one network-mean ACF reproduces the panel (n>=20 ratios 1.78/2.37/2.80). m spread within class essential. CORRECTED 2026-11-08: f spread at fixed kappa DOES matter (own m + class-value f: 1.84/2.51/3.03 vs own f 1.80/2.41/2.85); the '40-class shuffle' only matched by compensation (see [[project-field-spread-shape]]). Cavity 4-constant law m ~ N(a - b kappa, s^2 kappa), f = c sqrt(kappa): 1.83/2.49/3.01, zeta1 0.29; n>=5 -> ~q (from the missing f spread, not m tails: within-class m is ~Gaussian, kurtosis +0.33). Identities s^2 = sigma^2 Q_T, c^2 = sigma^2 (Q0-Q_T) hold to ~4%; b vs mu*Bbar off 10-15%.
- Two-time HDMFT by iterative single-site sampling (soft-config kernel, 24 classes, neutral start; scratchpad vf/dmft*.py) converges: Bbar 0.90, Q_T 2.8, exits 0.61, rho(1) 0.86 all in seed range; Q0-Q_T 1.13 vs 1.52-1.58 (full model has 10-30% collective field mode). Panel n>=20: zeta1 0.30, ratios 1.81/2.46/2.96 (full 0.253, 1.79/2.37/2.80); n>=5 zeta1 0.22 but ratios ~q. Extending field ACF cut 60->200 tu did not raise Q0-Q_T.
- Regular-graph control: full model Q_T 3.33-3.36, Q0-Q_T 0.59-0.63, exits 0.68-0.70 at N=16000 and 64000 (no N trend); collective share 2.6-3.2% -> 1.8-2.0%. The sampler does NOT converge on Q0-Q_T: rises to 0.76 then drifts to 0.45 by iteration 26 (the 0.581 at it 14 was a passing value; retracted to the user). Suspected sampler bias: Q_T from finite windows counts slow life-to-life motion as static, leaking dynamic variance each iteration. Fix = windows >> lifetimes (N=4000 nodes enough on regular). Gap-1 cause is open.
- notebooks/07_floor_theory.ipynb (generator in session scratchpad nb/make_07.py) holds all of this; caches in data/demographic/floor/. Notebook 03 already had the deterministic-limit (lambda) version: fitness law, lifetime identity, kill rule, Rice x Slepian lifetime law.
- data/demographic/volatility_law.npz is a stale cache (56293x4) that life_rows/decomposition (5 cols) would not produce.
