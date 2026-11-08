---
name: feedback-notebooks-not-scripts
description: User wants theory+code+plots in Jupyter notebooks, no tests, single main branch, thesis kept only as PDF
metadata:
  type: feedback
---

On 2026-10-12 the user restructured the repo: scripts/ and tests/ removed, six notebooks in notebooks/
(01 model ... 06 facts) explain the theory in markdown with code and plots inline. No package: code
shared by several notebooks is in notebooks/economy.py; code used by one notebook lives in that notebook
(e.g. the HDMFT/renewal solvers in 02, the per-economy panel in 06). Only one git branch, `main`. The thesis is kept only as tesi.pdf (LaTeX
removed); the paper draft stays as LaTeX in paper/.

**Why:** "functions are simple enough" for tests; notebooks show "theory, code and plot in a single view".
**How to apply:** add new analyses as notebook sections (cache heavy results in data/, recompute if
missing), not as scripts or tests; don't create extra branches. Related: [[project-demographic-paper]].
