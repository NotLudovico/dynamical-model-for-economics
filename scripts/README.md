# Scripts

Four folders, one per part of the story ([`docs/STORY.md`](../docs/STORY.md)). Run every script from
the repository root, `uv run python scripts/<folder>/<script>.py`. Outputs go to `data/` (large ones
gitignored) and figures to `thesis/assets/figures/`; printed results are kept in `data/logs/`.
Shared estimators live in the package (`relative_glv/msb.py`: `lives`, `zeta_bins`, `hill`,
`tail_exponent`; `relative_glv/model.py`: `integrate_demographic`, `configuration_coupling`).

## `demographic/`: the main model and its data comparison

Reference point: α = 2.5, σ = 1.70, D = 0.1, C = 200 (code sign `mu=1.76`), one year = 0.5.

| Script | Purpose | Log |
|---|---|---|
| `robustness.py` | one economy per job, every MSB fact; N = 8k–64k, seeds, D, σ, α, C; `robustness.py retune` scans σ × D at N = 16k | `demographic_robustness.log`, `demographic_retune.log` |
| `msb_facts.py` | every MSB fact by size fifth at the reference point (simulates its own economies) | `demographic_msb_facts.log` (old point) |
| `multiscaling_theory.py` | test of ζ_q = (q/2)⟨w⟩_q: decomposition by degree, two-term law, regular-graph ablation | `demographic_multiscaling.log` |
| `b_null.py` | null test for the tail index b | printed |
| `phase_lines.py` | renewal theory: σ_c(D), zero-growth line, turnover | |
| `reduction.py` | one-firm reduction (a firm in its measured field plus its own noise) | |
| `constant_fitness.py` | exact one-firm results at constant field (quadratures) | |
| `facts.py`, `volatility_law.py`, `facts_plot.py` | stylised facts and the volatility law v² = V_F + B (old point σ = 1.75) | |
| `plot_*.py` | figures `demographic_{msb_facts,robustness,retune,multiscaling}.png` | |

## `phase/`: phase diagram of the deterministic model

`phase_theory_lines.py` (HDMFT lines for a family of graphs), `phase_sigma_c_bilinear.py` and
`phase_zero_growth_bilinear.py` (closed forms), `phase_coexistence.py` (chaos against trapping above
σ_c), `plot_phase_theory.py` (figure `phase_theory.png`), `plot_hdmft_zero_growth.py`,
`hdmft_twotime_validation.py` (two-time HDMFT against simulation).

## `lifecycle/`: exact one-firm theory (deterministic limit)

`beta_spells.py` (spell-panel runs of the full model; constants used by the others), `fitness_stats.py`
and `mf_law.py` (the per-firm fitness and its mean-field law), `surrogate.py` (one firm in a Gaussian
field), `kill_rule.py` and `rice.py` (which dip ends a life; lifetime law), `plot_fitness_law.py`,
`plot_lifetime_law.py` (figures `fitness_law.png`, `lifetime_law.png`).

## `thesis/`: generators of the submitted thesis figures

One script or notebook per figure; [`thesis/FIGURES.md`](../thesis/FIGURES.md) maps figures to
generators. `notebooks/story.ipynb` is the thesis-era explainer and no longer runs (its data files
were removed with the own-degree normalisation).

Code removed in the 2026-10-11 cleanup (the cemetery models, exploratory β studies, diagnostics) is
listed with its results in [`docs/CEMETERY.md`](../docs/CEMETERY.md) and can be restored with
`git checkout archive/pre-cleanup -- <path>`.
