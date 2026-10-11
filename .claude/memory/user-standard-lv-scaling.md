---
name: user-standard-lv-scaling
description: User rejects the rowavg coupling variant; only the standard disordered-LV normalization (sigma/sqrt(C) per edge) counts
metadata:
  node_type: memory
  type: feedback
  originSessionId: 4185de35-c8d4-4ace-9730-ee2c380f9cf6
  modified: 2026-09-30T21:09:56.041Z
---

Don't propose `powerlaw_rowavg` robustness checks or build arguments on it. The user: "I dont care about rowavg, it's not the standard scaling used in the lotka volterra models."

**Why:** the model follows the standard disordered-LV normalization (per-edge mean mu/C, std sigma/sqrt(C)), and the user treats that as the physically meaningful choice.
**How to apply:** if a referee-style normalization objection comes up, answer it by citing the standard scaling of the disordered-LV literature, not with alternative normalizations. Related: [[project-article-goal]].
