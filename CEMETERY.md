# The cemetery

Everything that was tried and abandoned, with the reason. The story of what survived is in the
notebooks (`notebooks/`, index in the README). Entries give the idea, the result, the cause of death, and where the code or
numbers are (when they were kept; some tests were scratch-only).

The code of most entries was removed on 2026-10-11. Every path below still exists in the commit
tagged `archive/pre-cleanup`: restore one with `git checkout archive/pre-cleanup -- <path>`. The
scripts, tests, run logs (`data/logs/`) and LaTeX notes that the notebooks replaced on 2026-10-12 are
in the commit tagged `archive/pre-notebooks`.

Two lessons recur:

1. **Self-limitation ties the volatility decline to a size cap.** In d ln x = Φ(t) − F(x), size
   dependence comes from the restoring rate, and the scale where it switches on is where sizes pile
   up. A steeper decline always meant a more bimodal size distribution (β ≈ 1.12 × slope of the size
   density over 103 runs).
2. **Granular volatility gives ratios ζ_q/ζ_1 ≥ q.** If a firm's volatility factors into a size part
   times a size-independent part, ζ_q = qζ_1 exactly; averaging over subunits, partners or customers
   pushes the ratios to q or above, and big firms Gaussianise. Removing self-limitation with quenched
   couplings lets the best niche take the whole economy.

---

## A. Estimators and protocols

**Balanced survivor panel.** Only firms present throughout the window. β depended on the window
length T (0.41 → 0.35 for T = 60 → 480 at α = 2.5) because the survivor filter has no long-window
limit. Rejected: an exponent that depends on the analyst's window is not a property of the model.
MSB use an unbalanced panel.

**Peak-started fit on spells.** Starting the fit at the peak of the binned curve made β jump by
about 0.09 between economies, because the top of the curve is flat. Replaced by the larger half of
spells, then by all bins once the demographic model removed the hump.

**Spells of the deterministic model, all bins.** Volatility rises with size below S̄ ≈ 0.2 (calm, brief
passes near the floor), so an all-bins fit gives β ≈ 0 (−0.07 to −0.02); the data-like exponent lives
only on the falling branch, under two decades. This is what motivated the demographic model.

**IPO-style listing.** Listing a firm from its first passage above an entry size gave β = 0.19 and
ratios 1.85, 2.49, 2.89 (α = 6), but the curve is a hockey stick: flat over the lower half, steep over
the upper half. Not a power law.

**Calibrated year (deterministic model).** At the headline step of 0.5 time units, growth within a
spell has lag-1 autocorrelation 0.7–0.95. Calibrating the step to zero autocorrelation shifts
volatility up 2–4× without changing the curve's shape; an apparent shallower hump was selection of
short spells. (`scripts/calibrated_year.py`, `data/logs/calibrated_year.log`)

**The b match.** The demographic model gives b = 4.40–4.44 at every parameter point for lives with
≥ 2 growth rates, close to 4.6. A null with iid Laplace growth and no firm differences gives 4.34, so
it is an estimator effect of short lives; at ≥ 20 growth rates the model gives 6.9. (`b_null.py`)

**Coupling normalised by each firm's degree (`powerlaw_rowavg`).** Not the standard disordered-LV
scaling; dropped on that ground, not on results.

## B. Variants of the thesis GLV

**Standard (absolute) GLV.** Has a fixed size scale; it grows only when cooperation is tuned to
criticality, and then diverges in finite time; its size–variance relation is a transient. Replaced by
the relative model in the thesis.

**λ = 0.** Without the floor the economy freezes.

**Pareto carrying capacities (static).** Condenses onto the top firm.

**High disorder.** σ = 2.6 (α = 3.5) fills five decades of sizes but volatility rises with size;
σ ≥ 3 condenses or freezes.

**Overnight scan (103 runs over α, σ, μ, λ, C, N).** No regime gives a long decline and a broad size
distribution together: β ≈ 1.12 × (slope of the size density) + 0.02, correlation 0.92.

**Firms as portfolios of products.** Gives β = 1/2, the central-limit value.

**Log-space demographic noise** (noise in ln S without the Itô drag, equivalent to λ = D/2 plus
square-root noise). By current size it looks good (β 0.21–0.35, data-like ratios at α = 2.5), but
on spells the hump comes back.

**Taylor-law noise √S^b.** Proposed to set β directly; not pursued, it adds a parameter that is a
direct knob for β.

**Sublinear self-limitation (x/m)^θ.** With linear interactions θ = 0.2 condenses (top firm 52%): a
weak self-term cannot hold a firm whose linear field crushes its neighbours. With interactions in the
same units it is an exact power transform of the θ = 1 model, so β scales by θ and the size range by
1/θ: nothing gained.

**Cooperative mean interaction (μ < 0).** Size follows degree (∝ k^1.2–1.9) and Hill ≈ 1 at α = 2.2, but
the top hub holds 37%, yearly volatility is 3–5 and rises with size, and σ = 1.5 freezes.

**Niche GLV on a 2D geometric graph.** Relative version condenses at N = 8000; the standard version
diverges for σ > √2 with mixed-sign couplings. No size–volatility result.

## C. Model screen, rounds 1–2 (`scripts/screen/`, `data/logs/screen_run*.log`)

Fixed scorecard: firm-time per decade, Zipf, no condensation, β, ratios, straightness, b, kurtosis.

**Round 1: brake-free variants** (no x/m term: capacity that follows x, demographic, inertia, min-flow,
sublinear; with and without random exit). Every brake-free variant condenses (top firm 0.9–1.0 of
the economy). Variants that keep the brake have no Zipf (Hill 4.5–6) and β between −0.05 and 0.15.
Best 3.5/8.

**Round 2: couplings redrawn in time, share and scope fields.** Annealing the couplings stops
permanent winners but gives either condensation (share fields) or no size dependence; scope with
partners redrawn within a year averages the Herfindahl index and pushes ratios to q.

## D. Granular and network mechanisms

**Slot model** (a firm is a set of links, each a killed geometric Brownian motion). Solvable (link
law, Herfindahl), but a lucky link makes a firm both big and concentrated, so volatility rises with
size. A version capping links by customer size also fails.

**Partner model** (round 3, `variants3.py`, `theory_partner.py`). Volatility from the Herfindahl
index of a firm's partners. The static map gives ratios below q with Zipf partners, but the dynamics
give ratios above q (2.1, 3.3, 4.6), b 6–7.7 and Gaussianising large firms: it is a granular model
with partners as subunits.

**Scope / cascade model.** Correlated shocks over a footprint of markets; with critical cascades
β = 1 − d_f/d with no free exponent. The full firm simulation condenses (top firm 34%), the granular
aggregate flattens β, kurtosis grows strongly with size, and there is no persistence.

**Reduced random-growth tests.** One SDE per firm with an exact current-size law. Random growth from
tiny entrants gives ratios above q and dilutes the life-panel β (0.2 by current size → 0.09 on lives).
Ratios below q need a volatility component whose spread grows with size.

**Sales network** (round 4, `variants4.py`). Sizes from a Leontief system; buyers switch suppliers
and re-source in proportion to size. Best 4/8: ratios above q at every setting, no small firms (death
on losing the last customer puts a floor at one customer), a Zipf/β trade-off, and the input–output
loop condenses with many slots.

**Contest ("yard-sale") model.** Firms meet at rate ∝ (x_i x_j)^θ and a fair coin moves the smaller
firm's size. Proposed, not run: β = μ/2 − θ is a free knob, there is no persistence, and zero-sum
contests make similar-size firms anticorrelated.

**Three-layer model** (markets × shares × firm portfolios). Rejected before running as too
complicated (7–8 parameters).

## E. Other dynamics

**Deterministic zero-sum exchange** (`scripts/zerosum/`). Antisymmetric exchange
x_i' = Σ_j W_ij (x_i x_j)^a / m^(2a−1) with entry and exit. At a = 1 sizes are broad and growth is
fat-tailed but volatility does not depend on size; at a < 1 current-size β ≈ 1 − a, but the life-panel
β is about 0 and the ratios are above q. No Zipf (Hill about 4).

**Per-firm carrying capacity K_i** (round 5, `variants5.py`; Gibrat K with exit). Absolute
competition: the giants dominate every firm's field, yearly volatility about 3, no Zipf. Field
relative to m: condenses. Saturating pressure Σ α x_j/(K_i + K_j): Zipf-like (Hill 0.8–1.0) but
β ≈ 0. A firm-specific scale makes volatility depend only on x/K, hence no size dependence.

## F. Superseded drafts and claims

- **"Multiscaling comes from degree heterogeneity"** (thesis, balanced panel): under the spell panel
  the deterministic model's bend came from life cycles instead. In the demographic model the bend
  does come from the degree, through the fitness floor (see notebook 05).
- **Deterministic model as the paper's main model**: replaced by the demographic model, which has no
  hump and twice the range.
- **Old reference point σ = 1.75, D = 0.1**: β = 0.19 at N = 8000 was inflated by a saturated graph;
  it converges to 0.13 at N ≥ 16k, hence the retune to σ = 1.70.
- **Standalone notes** `beta_theory`, `lifecycle_theory`, `dmft_theory` and `zero_growth_theory`
  under `thesis/standalone/` were superseded by `theory/theory.tex`, itself replaced by notebooks
  02 and 03 (its survivor-panel appendices, the adiabatic law and the degree channel, were not
  carried over; they are in `archive/pre-notebooks`).
- **Removed in the cleanup:** the `beta_*` and survivor-theory scripts behind `beta_theory.tex`; the
  spell scan and headline scripts (`scan_spells.py`, `headline_spells.py`, `plot_data_comparison.py`);
  the mean-degree cutoff and mechanism probes; the phase onset, attractor and divergence diagnostics;
  eleven exploratory life-cycle scripts (anatomy, dispersion, lifetime representation, ablations,
  long windows, survivor checks); the first demographic batch and its draft figures (`batch.py`,
  `paper_figures.py`); four thesis-era notebooks (`graph_role`, `meandeg_scaling`, `own_vs_mean`,
  `scale_fix`).
