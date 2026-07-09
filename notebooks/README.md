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
   variance across degrees, so σ⊥k and β≈0.86 is a firm limiting its *own* growth by its own size;
2. **β is topology-invariant**, and so, nearly, is the **multiscaling**: the fully-connected control
   (no graph at all) multiscales *most strongly*, and a heavier degree tail mildly **weakens** the
   effect. The multiscaling is made by the disorder and the dynamics, not by the topology;
3. the firm-to-firm heterogeneity is **quenched** — frozen into the coupling realization, not emergent
   from the dynamics — which is why the disorder-averaged DMFT order parameters miss it;
4. the **mean-degree** normalization needs `C ∝ N` to stay size-independent, and even then gives
   weaker (more granular) higher moments than own-degree;
5. own-degree's exponents are ~4× too steep, but that is **one explained change of size units**
   (`size ≈ abundance^3.5`) that lands both the multiscaling on MSB *and* the size distribution on
   Zipf.

## The notebooks

| # | notebook | question | key finding |
|---|----------|----------|-------------|
| 1 | `graph_role` | **what role does the graph play in generating the observables?** | **almost none, for the magnitudes.** σ⊥k; β invariant across topologies *and* under deleting the graph; multiscaling strongest in `fc`; the field is degree-flat at both 2nd and 4th moment. What the couplings *do* set is **which firm gets which role** (quenched, ρ=0.93) |
| 2 | `meandeg_scaling` | what C(N) holds mean-degree β size-independent? | **C ∝ N** — the pl2.5 super-hub saturates at k_max=N−1, so N/C is the control ratio |
| 3 | `meandeg_characterize` | mean-degree at C∝N: does it work? | clean power laws, β≈0.34 (near-empirical magnitude), but **weaker multiscaling** (ζ₄/ζ₁≈2.1 vs own-degree ≈2.9) |
| 4 | `own_vs_mean` | own-degree vs mean-degree, head to head | σ(k) **flat vs rising** is the signature; own-degree wins multiscaling + robustness, mean-degree wins β magnitude |
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
- **Multiscaling ζ_q absolutes are N/sample-fragile.** Own-degree's ζ₄/ζ₁ reads 2.9–3.5 across
  different runs. Trust the *ratios and orderings across conditions at fixed N*, not the exact values.
  In particular, `pl2.5` landing on ζ₄/ζ₁=2.90 against MSB's 2.9 is a coincidence of it sitting
  mid-pack (its own error bar is ±0.26), not a mechanism.
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
settings it **overturned two claims** this README used to make:

- ~~"the field's excess kurtosis carries a ∼1/k granularity"~~ — measured directly, the field's excess
  kurtosis is **flat in k and negative** (≈−0.5, sub-Gaussian): −0.02 per decade of k, where a true
  1/k decay would fall by a factor 14. The neighbours a firm sums over are coupled trajectories of one
  chaotic economy, not independent draws, so they never centrally-limit. Own-degree neutralizes the
  degree channel at the fourth moment too.
- ~~"the multiscaling character depends on the degree tail (pl2.5→MSB, exponential→granular)"~~ — every
  topology multiscales far above the granular line (min ζ₄/ζ₁ = 2.26 vs granular 1.0); `exponential` is
  not granular; and ζ₄/ζ₁ *anticorrelates* with degree CV (Spearman ρ=−0.90, p=0.04), with the
  graph-free `fc` control highest at 3.52. The degree tail dilutes the multiscaling rather than
  creating it.

**Where the multiscaling actually comes from is therefore open.** It is not degree, not the degree
tail, and not the field's non-Gaussianity. `fc` multiscales best, which points at the plain
Gaussian-disorder relative GLV — where the DMFT already lives.

A p-sweep notebook (dialing the coupling degree-exponent to try to strengthen the higher moments) was
removed earlier — **negative result**: no knob (ζ₄/ζ₁ ≈flat across the family), and it surfaced the
ζ_q N-fragility now recorded as a caveat above.
