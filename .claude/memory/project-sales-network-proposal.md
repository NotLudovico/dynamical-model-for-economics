---
name: project-sales-network-proposal
description: "2026-10-11 radical-model proposals for MSB+Zipf — min-stake contest model (superseded) and the sales-network (dynamic Leontief, customer switching) model; not yet simulated"
metadata:
  node_type: memory
  type: project
  originSessionId: 70db33dd-6e72-477d-b4a2-8982a82c5c8d
  modified: 2026-10-11T15:13:20.061Z
---

User asked for a radical model explaining MSB facts + Zipf together; rejected the first proposal ("give me something better").

1. Contest / yard-sale model (superseded): pairs meet at rate (x_i x_j)^theta, fair coin moves min(x_i,x_j). beta = mu/2 - theta (theta a free knob), no persistence, similar-size firms anticorrelated (zero-sum).
2. Sales-network model (current proposal): x_i = sum_j pi_ij a x_j + pi_i0 F (Leontief; total fixed by final demand F). State = shares pi; buyers drop one of k suppliers at rate omega and re-source proportional to x (entrant with prob phi); exogenous exit delta with preferential re-sourcing.
   Growth: g_i = sum_j w_ij g_j (sales-share-weighted buyer growth -> correlation with aggregate rises with size) + jumps from gaining/losing buyers (sigma^2 ~ (omega/k) H_i, H = customer Herfindahl).
   Buyers of a supplier are a uniform sample of firms -> chunk tail zp = 1 (Zipf), n ~ x -> granular map gives beta ~0.20, ratios ~1.71 2.17 2.46 with no fitted parameter.
   Closest literature: Kelly, Lustig, Van Nieuwerburgh "Firm volatility in granular networks" (NBER 2013).
   Test idea: Compustat major-customer (>10% sales) disclosures — customer concentration should carry the sigma/sigmabar dispersion and multiscaling.
Zipf tail balance worked out: with exogenous exit delta and proportional drift nu, the tail equation is nu*zeta - delta + O(x^-2beta) = 0, so zeta = delta/nu = 1/(1-phi) for any beta >= 0 (beta > 0 only makes it cleaner). Without killing, a stationary level-martingale has zeta = 1 - 2beta.
Risks: same family as the round-2 partner model (ratios > q, b 6-7.7), but that one used refreshed partners and Gaussian shocks; persistence needs adjustment lags.
Related: [[project-scope-model]], [[reference-msb2024-facts]], [[feedback-model-complexity]].

RESULT 2026-10-11 (scripts/screen/variants4.py, `run.py r4`, N=4000, 2 seeds, grid a {0,.5} x k {2,3,5} x kh {1,3,10}): FAILS, best 4/8.
- Ratios above q at every point (2.1-2.5, 3.2-4.3, 4.4-6): same failure as the partner model; the static Herfindahl map's ratios < q never show up.
- No small firms: 55-90% of firm-time sits in one decade (entry size = one slot); death on losing the last customer puts a floor at one slot.
- Zipf and beta trade off: Hill ~1.2 only at a=0, kh=10 (beta .17, top share .1); Hill ~3 gives beta .25-.43; a=.5 with many slots condenses (Hill .4-.7, top .26-.36).
- Upstream propagation gives pair correlation only .02-.06.
Lesson: every granular family tried (sub-units, partners, customers) gives ratios >= q under the MSB protocol; only the GLV life cycle (rise-plateau-fall) ever gave ratios < q.
