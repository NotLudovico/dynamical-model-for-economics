# Degree, volatility & heterogeneity in the relative GLV

Exploratory notebooks studying how the disordered interaction structure generates the firm-growth
statistics of the **own-degree** relative Lotka–Volterra model, and how it compares to the
**mean-degree** normalization.

Each notebook is self-contained — the model equations are in its markdown, and it runs from the repo
root **or** from this folder (a location-portable setup cell walks up to the repo root). Figures and
arrays are written to `../data/`.

## The arc

Started from "what role does degree play in generating volatility?" and ended at a self-consistent
picture:

1. a firm's **own degree does not set its volatility** — own-degree normalization quiets it out;
2. **β is topology-invariant** and mechanistically explained (the field variance a firm feels is
   degree-flat), while the **multiscaling** (higher moments) is shaped by the degree *distribution*;
3. the firm-to-firm heterogeneity that carries the multiscaling is **quenched** — frozen into the
   coupling realization, not emergent from the dynamics — which is why the disorder-averaged DMFT
   order parameters miss it;
4. the **mean-degree** normalization needs `C ∝ N` to stay size-independent, and even then gives
   weaker (more granular) higher moments than own-degree;
5. own-degree's exponents are ~4× too steep, but that is **one explained change of size units**
   (`size ≈ abundance^3.5`) that lands both the multiscaling on MSB *and* the size distribution on
   Zipf.

## The notebooks

| # | notebook | question | key finding |
|---|----------|----------|-------------|
| 1 | `degree_volatility` | does a firm's degree set its volatility? | **No** — σ⊥k, S⊥k; β≈0.8 is self-limitation, not hub-quieting |
| 2 | `topology_multiscaling` | does the graph move β / the multiscaling? | β **invariant** across topologies; the **multiscaling character** depends on the degree tail (pl2.5→MSB, exponential→granular). Includes **fc** (no-topology control) |
| 3 | `field_granularity` | *why* is β degree-blind? | field mean & variance and growth σ are **flat in k** — own-degree equalizes the forcing. (Refuted a σ∼1/k kurtosis story for the multiscaling) |
| 4 | `quench_test` | is the heterogeneity quenched or emergent? | **Quenched** — same couplings, different ICs → same firms big (ρ=0.93) & volatile (ρ=0.82). The couplings assign the roles |
| 5 | `meandeg_scaling` | what C(N) holds mean-degree β size-independent? | **C ∝ N** — the pl2.5 super-hub saturates at k_max=N−1, so N/C is the control ratio |
| 6 | `meandeg_characterize` | mean-degree at C∝N: does it work? | clean power laws, β≈0.34 (near-empirical magnitude), but **weaker multiscaling** (ζ₄/ζ₁≈2.1 vs own-degree ≈2.9) |
| 7 | `own_vs_mean` | own-degree vs mean-degree, head to head | σ(k) **flat vs rising** is the signature; own-degree wins multiscaling + robustness, mean-degree wins β magnitude |
| 8 | `scale_fix` | is the magnitude gap fixable by a normalization? | one factor a≈3.5 lands **exponents on MSB AND size dist on Zipf** (a_β≈a_dist) — an *explained* size-unit change |

Suggested reading order is the arc above (1→8); `own_vs_mean` (7) is the capstone comparison and
`scale_fix` (8) the closing result.

## Honest caveats — read before quoting numbers

- **Multiscaling ζ_q absolutes are N/sample-fragile.** Own-degree's ζ₄/ζ₁ read 2.9–3.5 across
  different runs. Trust the *ratios and orderings across conditions*, not the exact values.
- **The scale-fix agreement uses favorable empirical exponents** (Zipf ν≈1, β≈0.20). The
  reparameterization-invariant statement is β/ν ≈ 0.20, which sits at the *upper edge* of the
  empirical range (~0.12–0.20) — consistent, not a precision match.
- **No empirical firm data is in this repo.** Empirical exponents are literature values (Axtell 2001
  Zipf for P(S); MSB 2024 for β and the multiscaling). To pin the scale-fix you'd want β and P(S)
  from one dataset.
- The own-degree size distribution P(S) has a **floor-induced hump** (the immigration λ acts as a
  minimum-size friction); only its power-law *tail* is compared to Zipf.

## Removed from this session

A p-sweep notebook (dialing the coupling degree-exponent to try to strengthen the higher moments) —
**negative result**: no knob (ζ₄/ζ₁ ≈flat across the family), and it surfaced the ζ_q N-fragility now
recorded as the first caveat above.
