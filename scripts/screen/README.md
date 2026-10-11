# Model screen

A shared harness for testing alternative models against one fixed scorecard (`score.py`), run in
parallel with joblib. Results are cached per run in `data/screen/` (gitignored); the printed tables
are in `data/logs/screen_run*.log`.

```
uv run python scripts/screen/run.py [r2|r3|r4] [summary|auto|<prefix>]
```

Scorecard (fixed before any run): share of firm-time per decade, Zipf (Hill index of the top 1%),
no condensation (largest share < 0.05), MSB beta and zeta_q/zeta_1 on spells, straightness, tail index b
of sigma/sigmabar, rescaled-growth kurtosis by size.

| Round | File | Models | Outcome |
|---|---|---|---|
| 1 | `variants.py` | thesis GLV and brake-free variants (capacity, demographic, sublinear, min-flow, inertia), with and without random exit | at most 3.5/8; the x/m brake gives no Zipf, every brake-free variant condenses |
| 2 | `variants2.py` | couplings redrawn in time, share/scope fields | condensation or no size dependence |
| 3 | `variants3.py`, `theory_partner.py` | partner (granular) model and its mean-field reduction | ratios above q, b 6-7.7 |
| 4 | `variants4.py` | sales network (dynamic Leontief, customer switching) | best 4/8; ratios above q, no small firms |
| 5 | `variants5.py` | per-firm carrying capacity K_i (absolute, relative, saturating competition) | absolute: explodes; saturating: Zipf-like but beta ~ 0 |

Round 5 was smoke-tested only (scratch scripts, not registered in `run.py`). Lesson across rounds:
granular families give ratios at or above q under the MSB protocol, and a firm-specific scale removes
the size dependence of volatility. The model kept for the paper is the demographic-noise relative GLV
(`scripts/demographic/`).
