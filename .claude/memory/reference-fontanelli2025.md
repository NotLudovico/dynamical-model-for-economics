---
name: reference-fontanelli2025
description: Fontanelli-Napoletano-Secchi 2025 (LEM WP 2025/27) French data: size-volatility is steep (-0.5) for small firms, flat (~-0.05) above 350K euro
metadata:
  type: reference
---
LEM WP 2025/27, https://www.lem.sssup.it/WPLem/files/2025-27.pdf (bib key fontanelli2025). FICUS/FARE French manufacturing,
2000-2019, ~1.7M firm-years, deflated total sales. "Bins methodology": firm-year observations sorted by size into equal-count
bins (1000 pooled, 500 per year/sector), sd of growth per bin -- i.e. cross-sectional current-size conditioning (Stanley-type),
keeps short-lived firms. Result: two regimes, slope ~ -0.5 below 350K euro (about half the median firm; mean ~8-10M), ~ -0.05 to
-0.08 above; shape varies by year and sector. Explained by an islands-economy model with imperfect size-productivity
correlation (Spearman ~0.17). They note Moran (2020) found the same feature.
Contrast with our model under the same protocol (scratchpad stanley.png, 2026-10-05): ours is flat at small S and steepens
above ~0.05 of the mean -- opposite curvature. See [[project-size-volatility-exploration]].
