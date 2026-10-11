---
name: feedback-no-sensitivity-scans
description: user doesn't care about parameter sensitivity (e.g. beta strongly dependent on sigma); don't push fine sensitivity scans as a referee risk
metadata:
  type: feedback
---
User said "i dont care about sensitivity" (2026-10-05) when I flagged that beta and the curve shape change fast with sigma (1.6 -> 1.9) in the demographic-noise model.

**Why:** a mechanism paper only needs a regime where the model works; sigma setting the level of beta is acceptable to them.
**How to apply:** report where a result holds, don't propose fine sensitivity scans or frame sensitivity as a blocker. Still report finite-size (N) checks and protocol dependence, which the user did engage with. See [[project-demographic-noise]].
