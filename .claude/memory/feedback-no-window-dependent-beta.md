---
name: feedback-no-window-dependent-beta
description: User dislikes beta depending on the observation window T; observables must have a window-free limit
metadata:
  node_type: memory
  type: feedback
  originSessionId: 129eb78b-f87d-407b-bebc-95a6ac144e36
  modified: 2026-10-02T12:03:06.725Z
---

The user rejected the beta note's "window dependence" result (beta and eps_k varying with the observation window T through the survivor ensemble n=T/2tau): "this thing of β depending on the window i dont like it".

**Why:** an exponent that changes with the analyst's window isn't a property of the model. The dependence came from the balanced-panel survivor filter, which has no T→∞ limit. See [[project-msb-protocol-spells]].

**How to apply:** don't present window dependence as a prediction or feature. Prefer observables with a stationary, window-free limit that mirror the empirical protocol. Related: [[project-article-goal]].
