# relative-glv

The relative (scale-invariant) generalised Lotka-Volterra model of firm growth. Interactions act on the mean firm, giving a self-consistently growing economy without finite-time blow-up. The model reproduces the stylized facts of firm-size dynamics observed in empirical firm-growth studies: a symmetric fat-tailed growth-rate distribution and a size-variance scaling exponent in the range of real data.

## The Model

The relative GLV describes competition among firms through a nonlinear replicator equation:

$$\dot{x}_i = x_i \left[1 - \frac{x_i}{m} - \frac{(\alpha x)_i}{m}\right],\quad m = \langle x \rangle = \frac{M}{N}$$

Here $x_i$ is the absolute size of firm $i$, $m$ is the mean firm size, $\alpha$ is the interaction matrix (disorder), and $M = \sum_i x_i$ is the total aggregate size. The key innovation is that competition scales with the mean firm $m$, not with the total aggregate $M^2$ (as in the standard GLV), so the aggregate $M$ can grow exponentially without diverging in finite time.

**Sign convention:** $\mu > 0$ denotes mean competition (lowers individual firm growth).

**Numerical stability:** We integrate in a share-plus-log-scale decomposition, $x_i = M w_i$ with $w_i$ on the simplex and $\ln M$ separate. This split is unconditionally stable: absolute abundances are never formed, so there is no overflow even when $M$ diverges. See `integrate()` in the model documentation for details.

## Key Results

- **Symmetric fat-tailed growth.** The distribution of firm-specific growth rates is symmetric around zero and fat-tailed (tent-shaped, closer to Laplace than Gaussian). See the notebook for growth distributions across parameter regimes.

- **Size-variance exponent.** Firm growth volatility decays with firm size as $\sigma(S) \sim S^{-\beta}$. The model yields $\beta_\infty \approx 0.17$ in the limit $N \to \infty$, approaching the empirical range of $\beta \approx 0.15$-0.20 observed by Moran, Santos, and Bouchaud.

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

The full mathematical derivation is in the thesis appendix and the technical document:

> `glv/docs/superpowers/specs/2026-06-21-relative-glv-dmft-derivation.md` in the parent monograph repository.

This README and the notebook present the main results. For the complete derivation of the DMFT self-consistency equations, the connection to the Aguirre-Lopez process, and proofs of the $\mu$-independence of shape observables and $\sigma_c$, see the derivation document.

## License

MIT. See LICENSE for details.
