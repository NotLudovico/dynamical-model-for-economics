---
name: reference-alpha-pl-global
description: model.coupling(kind="powerlaw") reads the degree exponent from module global model._ALPHA_PL (default 2.5); set it in the process that builds the graph
metadata:
  type: reference
---

`relative_glv.model.coupling(kind="powerlaw")` takes the degree exponent from the module global `model._ALPHA_PL`, read at call time (default 2.5, or env `RGLV_ALPHA_PL`). Setting it inside joblib workers does not set it in the main process. On 2026-10-02 this put the HDMFT solves in `mf_law.py` on the alpha=2.5 graph for every alpha. The resulting artefact ("HDMFT correlation time independent of alpha") was misdiagnosed as solver noise before the bug was found.

**How to apply:** whenever a graph or degree sequence is built, set `model._ALPHA_PL = alpha` in that same process. Suspect this first if results come out identical across alpha. Related: [[project-lifecycle-theory]].
