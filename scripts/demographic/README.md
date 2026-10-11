# Demographic-noise model

The relative GLV with birth-death noise `sqrt(D m x_i) dW_i` and firm turnover (a firm that reaches
zero is replaced by a newborn of size `D m`), integrated by `relative_glv.model.integrate_demographic`
(`compact=True` stores float32 trajectories for large N).

Reference point for the paper (after retuning at N = 16000): alpha = 2.5, sigma = 1.70, D = 0.1,
C = 200, mu = 1.76 (code convention, competition > 0), one year = 0.5 time units.

Outputs go to `data/demographic/` (gitignored, regenerable); printed results are in `data/logs/`.

| Script | What it does | Log |
|---|---|---|
| `batch.py`, `facts.py`, `reduction.py`, `constant_fitness.py`, `volatility_law.py`, `phase_lines.py`, `paper_figures.py` | first study (Oct 5): parameter batch, stylised facts, one-firm reduction, constant-field quadratures, volatility law, renewal phase lines, draft figures | |
| `factsheet.py` | fact sheet at the old reference point (sigma 1.75), 2 points x 4 economies, with a calibrated year | `demographic_factsheet.log` |
| `msb_facts.py` | every MSB 2024 fact on the old reference point | `demographic_msb_facts.log` |
| `robustness.py` | N = 8k-64k, seeds, D, sigma, alpha, C; `robustness.py retune` scans sigma x D at N = 16k | `demographic_robustness.log`, `demographic_retune.log` |
| `multiscaling_theory.py` | test of zeta_q = (q/2)<w>_q: variance decomposition by degree, two-term law v^2 = A kappa + B(Sbar), regular-graph ablation | `demographic_multiscaling.log` |
| `b_null.py` | null test for the tail index b of sigma/sigmabar (iid growth, no firm differences) | printed |
| `plot_factsheet.py`, `plot_msb_facts.py`, `plot_robustness.py`, `plot_retune.py`, `plot_multiscaling.py` | figures into `thesis/assets/figures/demographic_*.png` | |

Run any script with `uv run python scripts/demographic/<script>.py`; most take `summary` or `analyse`
to reprint results from cached output. See `docs/STORY.md` for what the results mean and `docs/CEMETERY.md` for what was dropped.
