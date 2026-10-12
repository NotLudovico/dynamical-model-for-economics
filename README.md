# A dynamical model for economics

Can an economy of **interacting** firms explain how firm-growth volatility depends on size? A
generalised Lotka–Volterra model of firms on a scale-free network, with interactions and
self-limitation measured in units of the mean firm, birth–death noise and turnover:

$$dx_i = x_i\Big[1 - \frac{x_i}{m} + \sum_j J_{ij}\frac{x_j}{m}\Big]dt + \sqrt{D\,m\,x_i}\,dW_i,\qquad m = \langle x\rangle,$$

with a newborn of size $Dm$ replacing every firm that reaches zero. It reproduces the size–volatility
exponents of Compustat ($\zeta_q = 0.21, 0.39, 0.55, 0.68$ against $0.20, 0.39, 0.51, 0.58$) and a ratio
curve $\zeta_q/\zeta_1$ that bends below $q$, with a derived mechanism: a firm's growth variance is a
fitness floor plus a demographic part, $\zeta_q = (q/2)\,b\,\langle w\rangle_q$, and the curve bends iff the
floor differs between firms of equal size — which on the network it does, through the degree. It does
not give Zipf's law or a four-decade range, and the reason is structural.

## Notebooks

Theory, code and figures together. Run them in order; each loads cached results from `data/` and
recomputes what is missing.

| notebook | content | runtime with cache |
|---|---|---|
| [`01_model`](notebooks/01_model.ipynb) | the facts, the model, relative sizes, a small economy, the MSB protocol | 40 s |
| [`02_phase_diagram`](notebooks/02_phase_diagram.ipynb) | HDMFT, closed-form $\sigma_c$, zero growth, chaos vs trapping; renewal law and $\sigma_c(D)$ | 2.5 min |
| [`03_life_cycles`](notebooks/03_life_cycles.ipynb) | deterministic limit: fitness law, one-firm surrogate, kill rule, lifetime law | 1 min |
| [`04_volatility_law`](notebooks/04_volatility_law.ipynb) | one firm is enough; field law; $v^2\simeq V_F+B$ | 20 s |
| [`05_multiscaling`](notebooks/05_multiscaling.ipynb) | the multiscaling identity and its test; regular-graph ablation | 1.5 min |
| [`06_facts`](notebooks/06_facts.ipynb) | every MSB fact, collective bursts, robustness, the $b$ null, no Zipf | 2 min |

Without the cache, the reference economies (notebooks 04–05, about an hour each) and the robustness
runs (notebook 06, 90 economies, hours on many cores) are simulated on first use.

## Layout

```
notebooks/      the six notebooks above, and economy.py: the code they share (graphs, the demographic
                integrator, the MSB life-panel estimators, the stored reference economies); code
                that only one notebook uses lives in that notebook
data/           cached results (small ones tracked; data/demographic/ is large and gitignored)
paper/          PRE draft (paper.tex, sm.tex, bib.bib); figures/ is written by the notebooks
CEMETERY.md     every model and protocol tried and abandoned, and why
tesi.pdf        the submitted thesis (deterministic model)
```

## Install and run

```bash
uv sync                      # numpy, scipy, networkx, matplotlib, joblib, jupyter
uv run jupyter lab           # open notebooks/
```

## License

MIT. See LICENSE for details.
