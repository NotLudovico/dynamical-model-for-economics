# A dynamical model for economics

Companion code to the article on how the growth volatility of interacting firms depends on their size.

Firm growth volatility falls with size as $E[v^q\mid\bar S]\sim\bar S^{-\zeta_q}$, with ratios $\zeta_q/\zeta_1$
below $q$. Whenever a firm's growth variance is a floor plus a part that falls with size,
$\zeta_q=(q/2)\,b\,\langle w\rangle_q$, and the ratios bend below $q$ only if the floors differ between firms of
equal size. The model is an economy in which they do: a generalised Lotka–Volterra economy of firms on a
scale-free network, with interactions and self-limitation in units of the mean firm, birth–death noise and
turnover:

$$dx_i = x_i\Big[1 - \frac{x_i}{m} + \sum_j J_{ij}\frac{x_j}{m}\Big]dt + \sqrt{D\,m\,x_i}\,dW_i,\qquad m = \langle x\rangle,$$

with a newborn of size $Dm$ replacing every firm that reaches zero. A firm's floor holds the climb after its
entry and the collapse before its exit, spread over its life, so floors differ with the length of a life and
with the degree of its node. At the minimum life length that matches the data (five growth rates), the
model gives $\zeta_q = 0.209, 0.400, 0.561, 0.685$ against Compustat's $0.20, 0.39, 0.51, 0.58$, and ratios
$1.92, 2.69, 3.28$ against $1.95, 2.55, 2.90$. It caps firm size, so it has no Zipf law and its decline
spans two decades rather than four.

## Notebooks

Theory, code and figures together. Every number and figure in the paper is printed or written by one of them or by the scripts below.
Each loads cached results from `data/` and recomputes what is missing.

| notebook | content | in the paper |
|---|---|---|
| [`01_model`](notebooks/01_model.ipynb) | the facts, the model, relative sizes, a small economy, the panel protocol | background |
| [`02_phase_diagram`](notebooks/02_phase_diagram.ipynb) | mean-field theory, $\sigma_c$, zero growth, renewal law, where the facts hold | Figs. strip and phase; mean-field and constant-field appendices |
| [`04_volatility_law`](notebooks/04_volatility_law.ipynb) | one firm is enough; the field; $v^2\simeq V_F+B$ | Sec. one firm; reduction appendix |
| [`05_multiscaling`](notebooks/05_multiscaling.ipynb) | the identity and its test; regular graph; all lives | Fig. mechanism, table of cases; identity appendix |
| [`06_facts`](notebooks/06_facts.ipynb) | every MSB fact, bursts, robustness, the fits, rescaled growth | Sec. data, Figs. facts, fits and tails; robustness and protocol appendices |
| [`07_floor_theory`](notebooks/07_floor_theory.ipynb) | what the floor is made of, lifetimes, the field's statistics, predictions | Secs. floors and predictions; field statistics |

The reference economies (notebooks 04, 05, 07) take about an hour each to simulate without the cache, and the
robustness runs of notebook 06 (80 economies) hours on many cores.

## Scripts

| script | what it does | in the paper |
|---|---|---|
| [`two_time_dmft.py`](scripts/two_time_dmft.py) | stationary two-time mean-field closure solved by sampling, with or without measured inputs | closure appendix |
| [`phase_onset.py`](scripts/phase_onset.py) | replica test of the noisy instability line | constant-field appendix |
| [`msb_curves.py`](scripts/msb_curves.py) | reads the Compustat rescaled-growth densities from the vector paths of the MSB figures | protocol appendix |

## Layout

```
notebooks/      the notebooks above, and economy.py: the code they share (graphs, the demographic integrator,
                the panel estimators, the stored reference economies and their life table)
scripts/        the three scripts above
data/           cached results (small ones tracked; data/demographic/ is large and gitignored)
paper/          the article (paper.tex with appendices, bib.bib); figures/ is written by the notebooks;
                derivations.tex holds working notes written for an earlier draft
CEMETERY.md     every model and protocol tried and abandoned, and why
```

## Install and run

```bash
uv sync                      # numpy, scipy, networkx, matplotlib, joblib, jupyter
uv run jupyter lab           # open notebooks/
cd paper && tectonic -X compile paper.tex
```

## License

MIT. See LICENSE for details.
