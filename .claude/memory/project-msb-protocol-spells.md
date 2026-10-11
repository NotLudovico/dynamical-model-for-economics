---
name: project-msb-protocol-spells
description: "MSB 2024 use an unbalanced panel; repo's balanced survivor filter causes beta(T). The spell (floor-crossing) estimator failed as a fix. Findings of 2026-10-02"
metadata:
  node_type: memory
  type: project
  originSessionId: 129eb78b-f87d-407b-bebc-95a6ac144e36
  modified: 2026-10-02T14:01:05.340Z
---

Verified from Moran–Secchi–Bouchaud, arXiv:2404.15226, Sec. 3.1: Compustat 1961–2020, an unbalanced panel. sigma_i is the sqrt(pi/2) MAD of firm i's annual growth around its own mean over its observed life. S̄_i is the lifetime-average normalised size. There are 25 equal-count bins by S̄_i. A firm needs ≥2 growth rates (robustness check: ≥20). zeta_q comes from E[sigma_i^q | S̄]. The thesis slides' <|g|^q|S> notation doesn't match this.

Repo `msb.size_volatility` keeps only firms above the 1e-6 share floor for the whole window. That balanced filter causes beta(T): at alpha=2.5, beta = 0.41/0.39/0.38/0.35 at T = 60/120/240/480. The ratios stay stable (~1.9, 2.7, 3.5).

`msb.spell_volatility` (one record per floor-crossing spell; driver `scripts/beta_spells.py` → `data/beta_spells.npz`, added 2026-10-02). Long runs out to T=1920 (scratchpad only, not in repo) show it converges:
- spells≥2 beta: alpha=2.2 ≈ 0.08 (flat from T=480), alpha=2.5 ≈ 0.20, alpha=3.5 ≈ 0.32, alpha=6 → about 0.36 (0.46/0.42/0.39/0.37).
- Ratios under spells: alpha=6 is data-like and stable (1.87, 2.59, 3.13). alpha=3.5 is close (1.78, 2.31, 2.60). alpha≤2.5 is broken (flat or negative).
- So under the MSB protocol the multiscaling bend does NOT need degree heterogeneity, which reverses the thesis attribution. The user pushed back on "doesn't converge" and was right.

All-firm estimator (no filter): beta converges at 2.2/2.5/6 (0.28/0.47/0.72), but ratios are far above q on heavy tails.

Open: a fully connected graph under spells (does entry/exit alone bend the ratios?), and a sigma/lambda scan for the beta level (0.37 at alpha=6 vs the empirical 0.15–0.20). The user felt "the model is wrong"; the answer given was that the protocol choice is untested, not that the model is refuted.

Scan launched 2026-10-02 ~17:20: `scripts/scan_spells.py` covers alpha {2.5,3,3.5,6} x sigma {1.6,1.75,1.9} x lam {1e-4,1e-3,1e-2}, mu=1.76, N=8000, 8 economies, listing threshold share 8*lam/N. It writes one npz per point to `data/scan_spells/` (resumable; log in scan.log). `uv run python scripts/scan_spells.py summary` prints the table and marks joint matches with *. Earlier diagnosis of the flat ratios at alpha=2.5: the within-degree (life-cycle) bend is data-like (1.82, 2.38, 2.68), the degree channel lowers zeta_1 (0.29→0.18), and the total bend is too strong (4ζ1−ζ4 = 0.52 vs 0.22 in the data).

Scan result (finished 2026-10-02, 103 min, all 36 points): beta needs heavy tails or high sigma, while a data-like ratio shape needs light tails (alpha≥3.5). The two meet near alpha≈3.5, sigma≈1.9. Best: alpha=3.5, sigma=1.9, lam=1e-2, spells≥20 gives beta 0.19 and ratios 1.80, 2.41, 2.83. With spells≥2, alpha=3.5, sigma=1.9, lam=1e-3 gives beta 0.214 and ratios 1.80, 2.31, 2.54. alpha=6 ratios are data-like everywhere, but beta ≥ 0.28. lam matters only on heavy tails (1e-4 best). Proposed next: alpha {3.5,4.5,6} x sigma {1.95,2.0,2.1} x lam {1e-3,1e-2}, then an N=16000 check and the spells≥2 vs ≥20 sensitivity.

Theory lead (2026-10-02): under spells the multiscaling is a life-cycle transient plus the degree channel. sigma_spell ≈ sigma_ad + sqrt(pi/2)*e*ln(Sbar/8lam)/n (climb and fall between the listing threshold and the plateau, spread over n growth steps, e = spell ends inside the window). Within size bins it explains 60–75% of the variance, with a fitted coefficient of 1.45–1.73 vs 1.25 predicted. Subtracting it gives near-fixed-shape ratios at alpha=6 (1.97, 2.87, 3.68); at alpha=3.5 the remainder is the degree bend. Plan: reuse the beta-note extreme-value law as the spell-length law P(L>T|u)=Phi(u)^{T/2tau} and build the spell ensemble (kappa, Sbar, L), giving zeta_q analytically. Empirical prediction: trimming entry/exit years or conditioning on age weakens multiscaling. The user wants a mechanism paper, not an exact fit; theory is the focus.

Related: [[feedback-no-window-dependent-beta]], [[project-article-goal]].
