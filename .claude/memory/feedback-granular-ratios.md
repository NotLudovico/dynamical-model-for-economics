---
name: feedback-granular-ratios
description: "In the paper, granular models give ζ_q = qζ_1 (fixed-shape volatility law), not \"ratios >= q\""
metadata:
  node_type: memory
  type: feedback
  originSessionId: 5a35721c-3800-44b5-8ac5-81869fb77554
  modified: 2026-10-11T21:30:51.495Z
---

User (2026-10-11) corrected my paper outline: "granular models give constant ratios, not >= q".

**Why:** in the asymptotic granular model the distribution of v/E[v|S] does not depend on size, so ζ_q = qζ_1 exactly. Ratios above q appeared only in our own granular-like variants, where large firms Gaussianise at finite size. That is a property of those variants, not of granular models in general.
**How to apply:** in the intro and the mechanism section, contrast "granular → ratios = q (fixed shape)" with "heterogeneous fitness floor → ratios < q". Don't generalise the screen results (see [[project-sales-network-proposal]]) to granular models in the literature.
