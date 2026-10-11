---
name: reference-decline-beta-peak-instability
description: "msb._decline_beta / cq_one pick the fit start by argmax of binned median vol; on spells the top is flat, so beta jumps ~0.09 between economies"
metadata:
  node_type: memory
  type: reference
  originSessionId: d4091a3d-1da9-48f5-a252-1691f3c1d54e
  modified: 2026-10-02T19:36:40.365Z
---

On spell panels, `msb._decline_beta` and `beta_spells.cq_one` start the fit at the argmax of the binned median volatility. The top of that curve is flat: adjacent bins differ by 0.001–0.02 in log10 vol. Noise therefore moves the peak bin, and the decline branch is curved, so beta jumps. Check of 2026-10-02 (HDMFT law, alpha=6, tau=3.51): peak bin 6 gives beta 0.42, bin 7 gives 0.51. With the start fixed at bin 8 every economy gives 0.56–0.58 (sd ~0.007). Per-economy beta is bimodal, and the median over 8 economies effectively votes between the modes.

**How to apply:** beta gaps below ~0.09 between runs are not resolved by this estimator. Before comparing betas, pick the fit range once (the pooled peak, or fixed size quantiles) and apply it to every economy. Related: [[project-lifecycle-theory]].

The peak cannot just be dropped. Under spells σ(S̄) has a hump that peaks near S̄ ≈ 0.2, so an all-bins fit gives β ≈ 0.00 at α=6 and −0.05 at α=2.5. MSB appear to fit all 25 bins of a monotone curve that flattens at the top. The model's branch steepens instead: at α=6, β is 0.43 from the argmax, 0.69 from the top half and 0.78 above S̄ > 1. A fixed top-half cut is stable (sd 0.004 against 0.041). The fit-range choice was offered to the user on 2026-10-02 and is still undecided.
