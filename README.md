# relative-glv

The relative (scale-invariant) generalised Lotka-Volterra model of firm growth. Interactions act on the mean firm, giving a self-consistently growing economy without finite-time blow-up. The model reproduces the stylized facts of firm-size dynamics observed in empirical firm-growth studies: a symmetric fat-tailed growth-rate distribution, and a size-variance relation whose form (volatility declining with size) matches real data, though not yet its exponent.

## The Model

The relative GLV describes competition among firms through a nonlinear replicator equation:

$$\dot{x}_i = x_i \left[1 - \frac{x_i}{m} - \frac{(\alpha x)_i}{m}\right],\quad m = \langle x \rangle = \frac{M}{N}$$

Here $x_i$ is the absolute size of firm $i$, $m$ is the mean firm size, $\alpha$ is the interaction matrix (disorder), and $M = \sum_i x_i$ is the total aggregate size. The key innovation is that competition scales with the mean firm $m$, not with the total aggregate $M^2$ (as in the standard GLV), so the aggregate $M$ can grow exponentially without diverging in finite time.

**Sign convention:** $\mu > 0$ denotes mean competition (lowers individual firm growth).

**Numerical stability:** We integrate in a share-plus-log-scale decomposition, $x_i = M w_i$ with $w_i$ on the simplex and $\ln M$ separate. This split is unconditionally stable: absolute abundances are never formed, so there is no overflow even when $M$ diverges. See `integrate()` in the model documentation for details.

## Key Results

- **Symmetric fat-tailed growth.** The distribution of firm-specific growth rates is symmetric around zero and fat-tailed (tent-shaped, closer to Laplace than Gaussian). See the notebook for growth distributions across parameter regimes.

- **Size-variance exponent.** Firm growth volatility decays with firm size as $\sigma(S) \sim S^{-\beta}$. The own-degree model reproduces the *form* of this relation but with $\beta \approx 0.6$-0.8, about $3\times$ the empirical $\beta \approx 0.15$-0.20 observed by Moran, Santos, and Bouchaud. This gap is structural: it is robust to parameters, network degree, and demographic noise, matching the MSB argument that the empirical exponent needs size-growing correlations beyond a single-level interaction model.

- **Phase transition.** The relaxed (single fixed point) to fluctuating (chaotic) phase boundary sits at $\sigma_c = \sqrt{2}$, independent of the mean competition $\mu$. Below it the relative sizes relax to a fixed point; above it they fluctuate persistently, and that fluctuating phase is where the MSB tent appears.

- **DMFT validation.** The steady-state observables (survival fraction, growth rate, shape moments) agree closely with a derived Dynamic Mean-Field Theory solver in the relaxed phase: sub-percent at low $\sigma$, within a few percent approaching $\sigma_c$. See `solve_fixed_point()` and test results.

See `story.ipynb` for detailed analysis: phase space, growth distributions, size-variance curves, the D2 multiscaling test (moment exponents vs $q$), numerical validation against DMFT, and a two-time DMFT appendix (fully-connected Gaussian limit) covering the fluctuating phase above $\sigma_c$.

## Repo Layout

```
relative-glv/
  README.md                      This file
  LICENSE                        MIT license
  pyproject.toml                 Package configuration
  story.ipynb                    Explainer notebook: figures, analysis, narrative
  relative_glv/
    __init__.py                  Package entry point; exports the public API: coupling, integrate, growth_rate, survivors, rescale, size_volatility, tent_stats, solve_fixed_point, sigma_c, solve_twotime
    model.py                     Relative GLV dynamics: coupling, integrate, growth_rate
    msb.py                       Firm-growth statistics: rescale, size_volatility, tent_stats
    dmft.py                      DMFT solver: solve_fixed_point, sigma_c, solve_twotime
  scripts/
    compute.py                   Heavy compute: generate phase diagrams and validation data
    phase_owndeg_fine.py         (mu, sigma) phase diagram, own-degree normalization
    msb_conditional.py           MSB conditional statistics (D1/D2/D3) over many economies
    plot_growing_*.py            Thesis main-text figure generators (see below)
    plot_dmft_*.py               Thesis DMFT-appendix figure generators (see below)
  figures/                       Regenerable figure output (gitignored; canonical copies in the thesis)
  docs/
    dmft-derivation.md           Full DMFT self-consistency derivation
  tests/
    test_model.py                Unit tests: coupling, integrate, consistency
    test_msb.py                  Unit tests: MSB statistics computation
    test_dmft_solver.py          Unit tests: DMFT solver and Gaussian moments
    test_validation.py           Integration tests: DMFT vs simulation agreement
  data/
    msb.npz                      Growth distributions and size-variance curves
    phase_diagram.npz            Phase-space observables (survival, growth, fluct.)
    dmft_validation.npz          Matched fully-connected simulation vs DMFT (backs the notebook validation figure)
    dmft_twotime.npz             Two-time DMFT autocorrelation and observables (fully-connected Gaussian limit, backs the notebook appendix figure)
```

## Reproducing the thesis figures

This is the code repository for the thesis. Each figure is generated by one script;
output lands in `figures/` (the canonical copies are committed in the thesis itself).

| thesis figure | script |
|---|---|
| `phase_diagram.png` (main) | `scripts/phase_owndeg_fine.py` (own-degree, N=4000) |
| `growing_growth_churn.png` | `scripts/plot_growing_growth_churn.py` |
| `growing_stationary_msb.png` | `scripts/plot_growing_stationary_msb.py` |
| `growing_multiscaling.png` | `scripts/plot_growing_multiscaling.py` (reads `data/msb_conditional.npz`) |
| `growing_conditional.png` | `scripts/plot_growing_conditional.py` (reads `data/msb_conditional.npz`) |
| `dmft_phase.png` (appendix) | `scripts/plot_dmft_phase.py` |
| `dmft_validation.png` (appendix) | `scripts/plot_dmft_validation.py` |

The Stage-1 critical-model appendix figures come from an earlier, exploratory study on the
standard (absolute) GLV and are not reproduced here; they are shown as static figures in the
thesis appendix.

## Install and Run

```bash
# Install the package and dev dependencies
uv sync --extra dev

# Read the explainer notebook
uv run jupyter notebook story.ipynb

# Run the test suite (DMFT solver, model consistency, validation vs simulation)
uv run pytest

# (Optional) Regenerate the heavy data files
uv run python scripts/compute.py
```

## Provenance and Citation

The full mathematical derivation is in the thesis appendix and in `docs/dmft-derivation.md`.

This README and the notebook present the main results. For the complete derivation of the DMFT self-consistency equations, the connection to the Aguirre-Lopez process, and proofs of the $\mu$-independence of shape observables and $\sigma_c$, see the derivation document.

## License

MIT. See LICENSE for details.
