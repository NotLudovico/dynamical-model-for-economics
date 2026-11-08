---
name: reference-literature-novelty-2026-10
description: "2026-10-11 literature check of the thesis GLV's novelty — what Mallmin 2024, Arnoulx de Pirey-Bunin 2024, network-GLV and Taylor's-law-from-chaos papers already cover"
metadata:
  node_type: memory
  type: reference
  originSessionId: 70db33dd-6e72-477d-b4a2-8982a82c5c8d
  modified: 2026-10-11T15:57:21.627Z
---

Already done by others:
- Mallmin, Traulsen, De Monte, PNAS 2024 (arXiv 2306.11031): x' = x(1 - x - sum a x) + lam, fully connected; chaotic rare/abundant turnover, bimodality; focal-species reduction x' = x(g - x) + lam with OU field g = -k + u eta (k, u, tau); abundance exponent nu = 1 + k/(u^2 tau). Taylor's law only loosely (u ~ k across parameters). No multiscaling.
- Arnoulx de Pirey & Bunin, PRX 2024 (2306.13634): lam -> 0 limit, jump process (jump up, gradual decline), P(N) ~ 1/N, tau ~ |ln lam|, sigma^2 phi_top < 1. No Taylor's law, no networks.
- Network GLV, fixed points/stability only: Aguirre-Lopez JPhysA 2024 (HDMFT, replicator mapping); Patil, Altieri, Aguirre-Lopez 2026 (2605.21364); Lee, Lee, Lee, Suweis, Park 2026 (2607.18809, degree-dependent).
- Taylor's law from LV dynamics: Tchourine, Carballo-Pacheco, Vitkup bioRxiv 2025; Camacho-Mateu et al. PRE 2024.
- Al-Hiyasat, Swartz, Gore, Kardar 2026 (arXiv 2602.13423), READ 2026-10-12: GLV + multiplicative environmental noise sqrt(2T) N eta + migration D(<N>-N), fully connected, no turnover, no network; power-law abundances N^-(2+D/T) and Taylor's law <N^(1+p)> ~ <N>^(1+D/T) with ONE exponent for all p (no multiscaling); mean-field eq. = Bouchaud-Mezard wealth model + GLV. No growth-volatility-vs-size. Overlap small; cited in paper intro as alhiyasat2026 (arXiv misc entry; PNAS stamp not verified).
- Altieri et al. PRL 2021: demographic noise, equilibria/glassy only.
- Econ side: Fontanelli, Napoletano, Secchi 2025 (resource-misallocation model); no GLV-chaos model of MSB facts found.
Not found anywhere: chaotic phase on scale-free graphs with degree-resolved DMFT; relative (growing-aggregate) GLV chaos; demographic-noise renewal phase line; multiscaling of size-conditioned volatility from life cycles; the beta-vs-size-density no-go.
The thesis background chapter already cites mallmin2024, arnouxbunin2024, aguirrelopez, patil2026. Repo github.com/NotLudovico/dynamical-model-for-economics is public and indexed.
Related: [[project-article-goal]], [[project-capacity-model]].
