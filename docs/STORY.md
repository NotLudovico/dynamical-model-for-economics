# The story

What this project claims, why, and where each piece lives. Last updated 2026-10-11. Everything that
was tried and abandoned is in [`CEMETERY.md`](CEMETERY.md).

## 1. The question

Large firms grow less erratically than small ones, but much more erratically than a sum of
independent parts would. In Compustat (Moran, Secchi & Bouchaud 2024, "MSB"):

- the growth volatility of a firm, v, against its lifetime-mean size S̄ scales as
  E[v^q | S̄] ~ S̄^-ζ_q over four decades, with ζ_q = 0.20, 0.39, 0.51, 0.58;
- the ratio curve ζ_q/ζ_1 = 2.0, 2.6, 2.9 bends below q (multiscaling);
- v divided by its size-conditional mean has a universal distribution with tail exponent b = 4.6;
- rescaled growth rates stay fat-tailed at every size (no Gaussianisation), multi-year growth is
  persistent, and a firm's co-movement with the aggregate rises with its size;
- firm sizes follow Zipf's law.

MSB show that granular models (a firm as a sum of subunits) cannot account for these facts. We ask
whether an economy of interacting firms can.

## 2. The model

Firms interact through a disordered generalised Lotka–Volterra (GLV) system on a scale-free graph,
with interactions and self-limitation measured in units of the mean firm m = ⟨x⟩:

    ẋ_i = x_i [1 − x_i/m + Σ_j W_ij x_j/m],   W_ij = A_ij (μ/C + σ z_ij/√C)

Every term is homogeneous of degree one, so the economy has no preferred size and grows at a rate
the dynamics selects. In the large-N limit the relative sizes obey the standard migration-supported
GLV with renormalised parameters; the fully connected case is the chaotic-turnover model of Mallmin,
Traulsen & De Monte (PNAS 2024). What is new here is the scale-free graph, the growth rate as an
output, and firm-growth observables.

Two versions:
- **Deterministic** (thesis model): an immigration floor λ(m − x_i) keeps firms from extinction.
- **Demographic** (main model for the paper): the floor is replaced by birth–death noise
  √(D m x_i) dW_i, and a firm that reaches zero is replaced by a newborn of size D m. No extra
  parameter (D replaces λ).

Reference point: degree tail α = 2.5, σ = 1.70, D = 0.1, mean degree C = 200, μ = −1.76 (μ < 0 is
competition here; the code uses the opposite sign, `mu=1.76`), N ≥ 16000, one year = 0.5 time units.

## 3. Theory

1. **Phase diagram.** A heterogeneous dynamical mean-field theory gives the line σ_c above which the
   economy stops relaxing and keeps fluctuating, with a closed form to first order in the degree
   variance, σ_c ≈ √2 [1 − ½ Var κ (1 − μ²/2π)], and a growth boundary that hardly depends on the
   network. With demographic noise, an exact renewal law for one firm gives σ_c(D) and predicts
   turnover to 0.5% below σ_c and 1.5% above. (`relative_glv/hdmft.py`, `relative_glv/renewal.py`,
   `thesis/standalone/theory/theory.tex`)
2. **One firm is enough.** Driving a single firm with its measured field (plus its own noise)
   reproduces the economy's size–volatility statistics to 0.01 in β. The field has a degree-resolved
   Gaussian law: mean and amplitude set by the degree, amplitude ∝ √κ.
3. **Life cycles** (deterministic limit). The one-firm equation is solved exactly; a firm exits on
   the first fitness dip whose area exceeds ℓ_c = ln(√(s/2π)/S_thr) (96–98% accurate), which gives
   its lifetime law with no fitted constant. A firm's volatility over its life is its log range.
4. **Volatility law** (demographic model). v² ≈ V_F + ⟨(D/S²)(1 − e^(−SΔt))⟩: a fitness floor nearly
   independent of size plus demographic noise filtered by the firm's own relaxation. It matches to 5%;
   β ≈ 0.2 is the effective slope of a crossover from −1/2 to 0.
5. **Multiscaling mechanism** (the central result). Write v_i² = v_F,i² + B(S̄). Then exactly
   ζ_q = (q/2)⟨w⟩_q, with w = B/(v_F² + B) the noise share and ⟨·⟩_q the v^q-weighted average at fixed
   size. The ratio curve bends below q if and only if the fitness floor differs between firms. On the
   network the floor grows with degree: the fitted law v² = Aκ^g + B(S̄) gives g = 1.00, it predicts
   ratios 1.86, 2.56, 3.06 against 1.80, 2.42, 2.88 measured, and a graph with equal degrees gives
   1.97, 2.90, 3.78, close to q. (`scripts/demographic/multiscaling_theory.py`)

## 4. Results against the facts

Demographic model at the reference point, N = 16000–64000, lives with at least 5 growth rates,
all 25 bins:

| Fact | Model | Compustat |
|---|---|---|
| β | 0.20 | 0.20 |
| ζ_q | 0.20, 0.38, 0.54, 0.66 | 0.20, 0.39, 0.51, 0.58 |
| Hump at small sizes | none | none |
| Range of the decline | 2 decades | 4 decades |
| Volatility, smallest → largest bin | 1.11 → 0.41 | 1.22 → 0.22 |
| Co-movement with the aggregate | rises with size | rises with size |
| Co-movement between similar-size firms | rises with size | rises with size |
| Tail b of v/E[v\|S̄], lives ≥ 20 | 6.9 | 4.6 |
| Rescaled growth across sizes | tails fatten (kurtosis 1 → 15) | roughly universal |
| Multi-year persistence | largest firms only | at all sizes |
| Zipf | no (Hill 4–5) | yes |

Robustness (`scripts/demographic/robustness.py`): the shape survives N = 8k–64k, seeds and the
neighbourhood in σ, D, α and C; β converges by N ≈ 16k; σ, D and C set its level.

## 5. Limits, and why

- **Range and Zipf.** In a one-variable model the self-limitation that produces the volatility
  decline also caps firm size. Softening it to (x/m)^θ multiplies β by θ and divides the size range
  by θ, so β × range is fixed (about 0.3 in the model against 0.8 in Compustat). Giving each firm its
  own scale removes the size dependence altogether. See the no-go section of the paper and the
  capacity models in the cemetery.
- **MSB's baseline sample** (lives with ≥ 2 growth rates) gives ratios above q: very short lives are
  dominated by newborns' noise.
- **Large firms are too volatile**, tails fatten with size and persistence is confined to the top,
  because small firms are driven by near-Gaussian, mean-reverting birth–death noise and large firms
  by rare, persistent fitness dips.
- **Aggregate volatility** is several times too large and does not fall as N^-1/2.
- **b**: the apparent match (4.4 at ≥ 2 growth rates) is an estimator effect of short lives
  (`scripts/demographic/b_null.py`).

## 6. Novelty against the literature

Already published: chaotic turnover and the single-species reduction (Mallmin et al. 2024), the jump
picture at vanishing immigration (Arnoulx de Pirey & Bunin, PRX 2024), fixed points of GLV on
networks (Aguirre-López 2024; Patil, Altieri & Aguirre-López 2026), Taylor's law from chaotic LV
dynamics (several 2024–2026 papers). Not found: the chaotic phase on scale-free graphs with its
degree-resolved theory, the renewal phase line, the multiscaling mechanism, the no-go, and any
chaotic-interaction model tested against MSB.

## 7. Where things are

| What | Where |
|---|---|
| Model, theory solvers, statistics | `relative_glv/` (tested in `tests/`) |
| Scripts (one folder per part of the story) | `scripts/` (see `scripts/README.md`) |
| Demographic studies | `scripts/demographic/` |
| Phase diagram | `scripts/phase/` |
| Deterministic life-cycle theory | `scripts/lifecycle/`, `thesis/standalone/theory/theory.tex` |
| Thesis figure generators | `scripts/thesis/` (map in `thesis/FIGURES.md`) |
| Paper draft | `thesis/standalone/paper/paper.tex` (built around the deterministic model; to be rewritten; its script citations predate the cleanup) |
| Demographic draft | `thesis/standalone/demographic/demographic.tex` |
| Figures | `thesis/assets/figures/` |
| Run logs with the numbers above | `data/logs/` |
| Abandoned models | [`CEMETERY.md`](CEMETERY.md); code removed, restore with `git checkout archive/pre-cleanup -- <path>` |

## 8. Next steps

1. Production run at the reference point (N = 32000, 6–8 economies) for the paper's data section.
2. Rewrite the paper around the demographic model, with the multiscaling mechanism as the main result.
3. Missing theory: field statistics above σ_c (two-time theory with noise), β(σ, D), co-movement
   rising with size, the 2-decade range.
