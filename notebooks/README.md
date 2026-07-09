# Degree, volatility & heterogeneity in the relative GLV

Exploratory notebooks studying how the disordered interaction structure generates the firm-growth
statistics of the **own-degree** relative Lotka–Volterra model, and how it compares to the
**mean-degree** normalization.

Each notebook is self-contained — the model equations are in its markdown, and it runs from the repo
root **or** from this folder (a location-portable setup cell walks up to the repo root). Figures and
arrays are written to `../data/`.

## The arc

Started from "what role does degree play in generating volatility?" and ended somewhere more
negative, and more interesting, than the question assumed:

1. a firm's **own degree does not set its volatility** — own-degree normalization equalizes the field
   variance across degrees, so σ⊥k and β=0.804±0.012 (per economy) is a firm limiting its *own* growth
   by its own size;
2. **β is topology-invariant.** The apparent degree-tail effect on the multiscaling (ρ=−0.90, p=0.04)
   is an artifact of *pooling economies*, an estimator with no limit. Measured per economy the model
   **barely multiscales at all**: ζ₄/ζ₁ = 3.77 against a no-multiscaling null of 4;
3. the firm-to-firm heterogeneity is **quenched** — frozen into the coupling realization, not emergent
   from the dynamics — which is why the disorder-averaged DMFT order parameters miss it;
4. the **mean-degree** normalization needs `C ∝ N` to stay size-independent, and even then gives
   weaker higher moments than own-degree — ⚠ but that comparison rests on *pooled* ζ_q (see caveats);
5. own-degree's exponents are ~4× too steep, but that is **one explained change of size units**
   (`size ≈ abundance^3.5`) that lands both the multiscaling on MSB *and* the size distribution on
   Zipf — ⚠ the multiscaling half of that claim also rests on pooled ζ_q.

## The notebooks

| # | notebook | question | key finding |
|---|----------|----------|-------------|
| 1 | `graph_role` | **what role does the graph play in generating the observables?** | **almost none.** σ⊥k; β invariant across topologies *and* under deleting the graph; ζ₄/ζ₁≈3.8 for every topology incl. `fc`; the field is degree-flat at both 2nd and 4th moment. What the couplings *do* set is **which firm gets which role** (quenched, ρ=0.93) |
| 2 | `meandeg_scaling` | what C(N) holds mean-degree β size-independent? | **C ∝ N** — the pl2.5 super-hub saturates at k_max=N−1, so N/C is the control ratio |
| 3 | `meandeg_characterize` | mean-degree at C∝N: does it work? | clean power laws, β≈0.34 (near-empirical magnitude), but **weaker multiscaling** (ζ₄/ζ₁≈2.1 vs own-degree ≈2.9) — ⚠ **pooled ζ_q, not yet re-run per economy** |
| 4 | `own_vs_mean` | own-degree vs mean-degree, head to head | σ(k) **flat vs rising** is the signature; mean-degree wins β magnitude — ⚠ the "own-degree wins multiscaling" half is **pooled ζ_q**, not yet re-run per economy |
| 5 | `scale_fix` | is the magnitude gap fixable by a normalization? | one factor a≈3.5 lands **exponents on MSB AND size dist on Zipf** (a_β≈a_dist) — an *explained* size-unit change |

Read `graph_role` (1) first: it is the consolidated graph story and it sets up why the *normalization*
(2–4) rather than the *topology* is the lever that moves β. `scale_fix` (5) is the closing result.

`graph_role.ipynb` is **load-or-compute**: by default it reads `../data/*.npz` and replots in seconds.
Set `GRAPH_ROLE_RECOMPUTE=deg,field,topo,quench` (or `all`) to re-integrate a section; a full recompute
is about 2.5 minutes on 8 cores.

## Honest caveats — read before quoting numbers

- **β always means the decline-branch slope.** The immigration floor λ gives σ(S) a rising plateau
  below the peak S\*; fitting across both branches returns β≈0.11, which is an artefact of mixing them
  and not a competing measurement. `msb._decline_beta`, `graph_role` §1 and §3's ζ₁ all restrict to
  S > S\*.
- **NEVER pool economies before conditioning on S — the pooled ζ_q has no limit.** Each realization of
  α is its own world (median σ spans ~1000×, each has its own S\*). Pooling n economies measures one
  economy with n disconnected components, and the resulting ζ₄/ζ₁ *slides with n*: 3.34 (n=5), 3.07
  (10), **2.89 (20)**, 2.84 (40), 2.53 (80), and the spread over repeated draws never shrinks
  (sd·√n grows 0.52→2.15, so there is no CLT). The pooled E[σ⁴|S] in a bin is dominated by its few most
  volatile firms — an extreme-value statistic, not an average. **The old "own-degree ζ₄/ζ₁ = 2.90 ≈
  MSB's 2.9" is the value the pooled curve happens to take at the 20 seeds those notebooks ran.**
  The per-economy median is well-behaved: 3.769, 95% CI [3.73, 3.83], flat across N=1000…8000.
- **The no-multiscaling null is ζ_q/ζ₁ = q, i.e. 1,2,3,4 — not 1,1,1,1.** Under the estimator used here
  (fit `log E[σ^q|S]` vs `log S`), a σ = A·S^−β with S-independent noise gives ζ_q = qβ exactly. The
  `1,1,1,1` line the old notebooks drew as "granular" is the null of the q-th-root convention. Reading
  one convention's reference against the other's estimator makes any rise in q look like multiscaling.
- **Topology dependence of the multiscaling is UNRESOLVED.** Five topologies give Spearman ρ=−0.30
  (p=0.62) against degree CV — underpowered — while `exponential` (4.23±0.12) and `pl2.5` (3.69±0.09)
  differ by ~4σ. Claim neither invariance nor an ordering from the current data.
- **The scale-fix agreement uses favorable empirical exponents** (Zipf ν≈1, β≈0.20). The
  reparameterization-invariant statement is β/ν ≈ 0.20, which sits at the *upper edge* of the
  empirical range (~0.12–0.20) — consistent, not a precision match.
- **No empirical firm data is in this repo.** Empirical exponents are literature values (Axtell 2001
  Zipf for P(S); MSB 2024 for β and the multiscaling). To pin the scale-fix you'd want β and P(S)
  from one dataset.
- The own-degree size distribution P(S) has a **floor-induced hump** (the immigration λ acts as a
  minimum-size friction); only its power-law *tail* is compared to Zipf.

## Superseded claims

`graph_role.ipynb` folds in and replaces four earlier notebooks (`degree_volatility`,
`field_granularity`, `topology_multiscaling`, `quench_test`), and in re-running them at matched
settings it **overturned three claims** this README used to make:

- ~~"the field's excess kurtosis carries a ∼1/k granularity"~~ — measured directly, the field's excess
  kurtosis is **flat in k and negative** (≈−0.5, sub-Gaussian): −0.02 per decade of k, where a true
  1/k decay would fall by a factor 14. The neighbours a firm sums over are coupled trajectories of one
  chaotic economy, not independent draws, so they never centrally-limit. Own-degree neutralizes the
  degree channel at the fourth moment too.
- ~~"the multiscaling character depends on the degree tail (pl2.5→MSB, exponential→granular)"~~ — the
  ordering was produced by **pooling economies**, whose ζ_q depends on the seed count. Per economy,
  ζ₄/ζ₁ is 3.75 (regular), 4.23 (exponential), 3.69 (pl2.5), 3.80 (pl3.5), 4.03 (fc), and `exponential`
  flips from lowest to highest. Whether any real topology dependence survives is unresolved (see
  caveats).

- ~~"own-degree's multiscaling sits on MSB (ζ₄/ζ₁ = 2.90 ≈ 2.9)"~~ — **the model barely multiscales.**
  Per economy, ζ₄/ζ₁ = 3.769 (CI [3.73, 3.83]) against the correct null of 4.0: ~6% concavity, where
  MSB report 27%. That is about **a fifth** of the observed multiscaling. The fully-connected control
  is at 4.03±0.03: *no multiscaling at all*. The old agreement was the pooled curve crossing 2.9 at 20
  seeds. Note the gap is not a finite-size effect: it is flat across N=1000…8000 (quenched 3.78, 3.83,
  3.61, 3.72) even as sd(β_e) falls 0.132→0.048.

**So the open question is no longer "why does the degree tail set the multiscaling". It is whether this
model produces MSB's multiscaling at all.** Per economy it makes about a fifth of it, and the
fully-connected limit makes none. Two ways to sharpen: (i) condition on S/S\*_e rather than raw S, to
test whether the residual pooled–quenched gap is a size-scale mismatch across realizations; (ii) push
the per-economy ζ_q to larger N with λ lowered to widen the ~1-decade fit range.

**Repo-wide consequence.** `scripts/msb_conditional.py`, `meandeg_characterize.ipynb` and
`own_vs_mean.ipynb` all report *pooled* ζ_q, and the thesis quotes them. Their multiscaling numbers are
contingent on how many seeds were pooled. In particular `meandeg_characterize`'s "mean-degree
multiscales more weakly than own-degree (ζ₄/ζ₁≈2.1 vs 2.9)" is exactly the kind of comparison this can
invent. Re-run per economy before quoting any of them.

A p-sweep notebook (dialing the coupling degree-exponent to try to strengthen the higher moments) was
removed earlier — **negative result**: no knob (ζ₄/ζ₁ ≈flat across the family), and it surfaced the
ζ_q N-fragility now recorded as a caveat above.
