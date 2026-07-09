# graph_role.ipynb — consolidating the graph-role experiments

**Date:** 2026-07-09

## Question

Four of the eight exploratory notebooks ask one question from different angles: *what role does the
interaction graph play in generating the firm-growth observables?* They are read separately today,
so the answer is assembled in the reader's head rather than on the page. This spec consolidates them
into one notebook that states the answer and shows the four legs that support it.

## Scope

Folds in, and then deletes, notebooks 1–4:

| # | notebook | leg |
|---|----------|-----|
| 1 | `degree_volatility` | does a firm's degree set its volatility? (δ, θ, partial OLS) |
| 2 | `topology_multiscaling` | does the degree distribution move β or the multiscaling? (+ `fc` control) |
| 3 | `field_granularity` | *why* is β degree-blind? (field variance flat, excess kurtosis ∼1/k) |
| 4 | `quench_test` | are the firm roles quenched into the couplings, or emergent? (ρ across ICs) |

Out of scope: `meandeg_scaling`, `meandeg_characterize`, `own_vs_mean`, `scale_fix`. These survive as
their own notebooks; `own_vs_mean` is about the normalization choice, not the graph, and `scale_fix`
is about size units.

## Architecture

`notebooks/graph_role.ipynb`, one section per leg:

```
0  md    the question; model equations; verdict table up front
1  code  portable setup; RECOMPUTE flag; locked operating point (mu=1.76, sigma=1.75, lam=1e-3, C=100)
2  code  shared helpers: loglog_bins, slope, degree_seq, owndeg_alpha, multiscaling
3  §1    degree -> volatility -> size triangle      3 panels
4  §2    the field a firm feels                     3 panels
5  §3    topology vs beta and multiscaling          4 panels
6  §4    quenched or emergent?                      bar chart
7  md    synthesis
```

### Compute policy

Every section is `load-or-compute`, keyed on a per-section tag:

```python
RECOMPUTE = set(os.environ.get("GRAPH_ROLE_RECOMPUTE", "").split(","))
def stale(tag, path):          # recompute if asked, or if the cache is absent
    return tag in RECOMPUTE or "all" in RECOMPUTE or not os.path.exists(path)
```

Default (`GRAPH_ROLE_RECOMPUTE` unset, caches present) the notebook loads `data/*.npz` and re-plots
in seconds. This is what makes deleting notebooks 1–4 safe: the compute that produced every cached
array lives in this notebook, exercised on demand, not lost.

### Shared helpers

`loglog_bins`, `slope`/`loglog_slope`, `degree_seq`, `owndeg_alpha` and `multiscaling` are currently
copy-pasted across the four notebooks with small drifts. They are defined once, in cell 2, and the
four sections call them. `owndeg_alpha` must stay a bit-identical mirror of `coupling()`'s
`powerlaw_owndeg` path (single rng stream: degrees, then couplings).

### Cache gap: experiment 1

`data/degree_volatility.npz` does not exist — the original notebook saved only a `.png`, and it ran
at self-described "quick settings" (N=1500, 4 seeds, smoke-length integration), weaker than the
other three. Section 1 is therefore re-run once at settings matched to `topology_multiscaling`
(N=2000, 20 seeds, tmax=300, window (230,290)), and writes `data/degree_volatility.npz` holding the
binned curves, a scatter subsample, and the scalars δ, θ, β_pool and the partial-OLS coefficients.
Leg 1 becomes as trustworthy as legs 2–4.

### Cache gaps: experiments 3 and 4 (accepted)

- `field_granularity.npz` stores binned curves and slopes but no raw scatter. The default figure
  draws the binned lines only; the scatter cloud returns under `RECOMPUTE=field`.
- `quench_test.npz` stores four scalars (`rhoS`, `rhoG`, `rho_diff`, `jaccard`). The default figure
  is the bar chart, which carries the result. The size/volatility scatter panels return under
  `RECOMPUTE=quench`. Recomputing is expensive (N=4000, 3 couplings × 6 ICs, tmax=250), so the
  scalars stand by default.

## The claim the notebook argues

Own-degree normalization is a **variance** normalization, and that single fact explains both results:

- **second moment** → the `1/sqrt(k)` coupling cancels the `sqrt(k)` of the neighbour sum, so the
  field variance every firm feels is flat in `k` → σ ⊥ k (δ≈0) → β is topology-invariant;
- **higher moments** → untouched, carrying a `~1/k` granularity (hubs average many neighbours into
  near-Gaussian noise; leaves track a few fat-tailed neighbours) → the multiscaling tracks the
  degree tail (pl2.5 → MSB-like, exponential → granular).

And the heterogeneity that carries the multiscaling is **quenched**: fix the couplings, vary only the
initial condition, and the same firms are big (ρ=0.93) and volatile (ρ=0.82). The graph and its
disorder assign the roles; the dynamics do not pick the winners. This is why disorder-averaged DMFT
order parameters miss the multiscaling.

Compressed: **the graph sets who, not how much.**

### CORRECTION (added after implementation)

The second bullet above is **wrong**, and building the notebook is what showed it. Reading the cached
arrays and re-running every section at matched settings:

1. **There is no `1/k` field granularity.** `field_granularity.npz` stores `slope_fexk = nan`, because
   all 22 binned values of the field's excess kurtosis are **negative** (≈ −0.5) — a log-log slope
   does not exist. Fit linearly against `log10(k)` instead and it is flat: −0.02 per decade over 1.15
   decades, where a true `1/k` decay would drop by a factor 14. The field is *sub*-Gaussian and
   degree-flat. The neighbours a firm sums are coupled trajectories of one chaotic economy, never
   independent draws, so they do not centrally-limit as `k` grows.
2. **The multiscaling does not track the degree tail** in the claimed direction. Every topology sits far
   above the granular line (min ζ₄/ζ₁ = 2.26 vs 1.0), so `exponential` is not granular. ζ₄/ζ₁
   *anticorrelates* with degree CV (Spearman ρ = −0.90, p = 0.04), and the `fc` control — **no graph
   at all** — has the largest ratio, 3.52 ± 0.10. The topology dilutes the multiscaling; it does not
   create it. `pl2.5` reading 2.90 against MSB's 2.9 is a coincidence of mid-pack placement (±0.26).

The corrected claim: own-degree neutralizes the degree channel at the **second and fourth** moments
alike. The graph sets no magnitude measured here — β is dynamics, the multiscaling is disorder plus
dynamics. What the coupling realization sets is the **assignment of roles** (§4, quenched, ρ=0.93).
"The graph sets who, not how much" survives, and is in fact stronger than intended.

**Where the multiscaling comes from is now an open question.** `fc` multiscales best, pointing at the
plain Gaussian-disorder relative GLV, where the DMFT already lives.

### A third correction: the definition of β

The folded-in `degree_volatility` notebook fitted `beta_pool` and its partial OLS across the **full**
size range, rather than on the decline branch `S > S*` where `msb._decline_beta` and the §3
multiscaling define β everywhere else in the repo.

**This was a latent bug, not a standing error.** At the notebook's own settings (N=1500, 4 seeds,
tmax=200, window 150–195) the peak of the binned σ(S) curve sits at bin 0 — there is no plateau, the
decline branch *is* the full range, and both estimators return β = 0.784 with b(size) = −0.68 → size
drives. Its published numbers and its conclusion were correct.

It breaks only once the immigration floor λ has time to accumulate a population of frozen firms. At
the upgraded settings adopted here (N=2000, 20 seeds, tmax=300, window 230–290) the peak moves to bin
2, the plateau holds 14% of the firms, and the full-range fit collapses to β = 0.107 with a *positive*
size partial (b = +0.02), reporting "degree is the stronger driver" — the opposite conclusion, with no
error raised and a plausible-looking number. On the decline branch: β = 0.856 (R² = 0.98), b = −0.892
(recovering −β), c = +0.147.

Note that `sv["beta"]`, the decline-branch estimator `msb.size_volatility` actually exports, is stable
across both regimes (0.816 vs 0.798). The old notebook computed and printed it; it simply used
`beta_pool` for the figure title, the θ·β check and leg 3. `graph_role` §1 restricts all three legs to
the decline branch and prints the full-range value alongside, so the gap is visible rather than latent.

## Verification

- The notebook executes top to bottom from a cold cache-present state in seconds, no errors.
- Executed once with `GRAPH_ROLE_RECOMPUTE=deg` to generate `data/degree_volatility.npz`, proving the
  compute path and the load path both work.
- Reported scalars agree with the values recorded in `notebooks/README.md` and MEMORY (β≈0.8, field
  variance slope ≈0, kurtosis slope ≈−1, ρ_S=0.93, ρ_σ=0.82, ζ₄/ζ₁ ordering pl2.5 > exponential).

## Consequences

- Delete `notebooks/{degree_volatility,topology_multiscaling,field_granularity,quench_test}.ipynb`.
- Rewrite `notebooks/README.md`: the arc drops from eight notebooks to five (`graph_role` plus the
  four out-of-scope ones), and the caveats section is preserved verbatim — the ζ_q fragility caveat
  applies to `graph_role` §3 now.
- `data/degree_volatility.npz` is a new tracked artifact.

## Deliberately not done

- No separate implementation plan document. This is one notebook and two file deletions; a plan doc
  would be ceremony.
- No recompute of legs 2–4. Their caches are current and their notebooks are being folded in, not
  reinterpreted.
