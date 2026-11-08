---
name: project-article-goal
description: "Thesis is already submitted; current work targets a journal article (PRE); log of the 2026-10-11/12 cleanups (STORY/CEMETERY docs, notebooks restructure, main only)"
metadata:
  node_type: memory
  type: project
  originSessionId: 4185de35-c8d4-4ace-9730-ee2c380f9cf6
  modified: 2026-09-30T21:09:52.536Z
---

The thesis in `thesis/` was already submitted (as of 2026-09-30). Ongoing work, e.g. the standalone note `thesis/standalone/beta_theory/beta_theory.tex`, is toward an article publication, not thesis revisions.

**Why:** the user corrected a suggestion to fold the theory note back into the thesis.
**How to apply:** don't propose edits to thesis chapters; frame advice around the paper (claim, referee risks, which results go in the main text and which in the SI). The user doesn't want the `powerlaw_rowavg` normalization pursued: it isn't the standard LV scaling (see [[user-standard-lv-scaling]]).

2026-10-11: user chose the deterministic model as main + one demographic-noise section, physics-journal framing (PRE, revtex) with the firm application. Draft rewritten in place: thesis/standalone/paper/paper.tex (claim: life cycles give the multiscaling shape and the volatility-spread tail; the self-limitation cap means no Zipf, Sec. "Why there is no Zipf law"; equivalence of the relative model to the standard GLV at large N credited to Mallmin/Bunin). Red \todo{} marks numbers that are scratch-only and need repo scripts. altieri2021 and gabaix1999 copied into thesis/bib.bib. No TeX on the machine and docker needs sudo, so the draft has not been compiled.

Calibrated-year test 2026-10-11 (scripts/calibrated_year.py, data/calibrated_year/, same economies as headline_spells): the paper's Table I samples a "year" of dt=0.5 where within-spell lag-1 growth autocorr is .67-.95. Lag-1 crosses 0 at dt ~3.5 (alpha 6), ~1.8 (alpha 3.5 s1.9), ~2.5 (alpha 2.5), ~1.4 (alpha 3.5 lam 1e-2).
At the calibrated step: hump halves (smallest-bin/peak .22-.35 -> .45-.56; smallest-bin vol ~1, Compustat 1.22), all-bins beta turns positive (alpha 6 dt 3/4: .13/.17, ratios 2.04 2.87 3.41 / 2.05 2.93 3.57), upper-half beta steepens (alpha 6 .56 -> .69-.71; alpha 3.5 .29 -> .38), upper-half ratios stay data-like at alpha 6 and 3.5 (alpha 3.5 dt 2: 1.87 2.57 3.09). Still humped (smallest-fifth local slope ~ -0.3), range shrinks to ~2 decades, large firms too volatile (top bin .5-1.3 vs .22), peak overshoots (2-2.4 vs 1.22). alpha 3.5 local slopes at dt 2: -.29 .06 .25 .36 .35 (flattens at top like data); alpha 6 still steepens to .9. alpha 2.5 ratios broken at every dt.

CORRECTION (same day, user spotted it): at matched Sbar the calibrated step lifts volatility by x2.7->4.4 (alpha 6), x1.5->2.5 (alpha 3.5), x1.9->3.4 (alpha 2.5) from small to large sizes, i.e. mostly a level shift, slightly MORE at large sizes. The 'hump halves' and 'smallest bin ~1 like Compustat' came from selection: the coarser step drops spells too short to have 2 growth rates (smallest Sbar 0.009 -> 0.025). Calibrating the year does not fix the hump; it is a protocol fix only.

Cleanup 2026-10-11 (pushed to origin/theory-phase-diagram, commits 81f6231..a4be538): run logs moved to data/logs/ (old paths data/*.log in other memories now live there); data/calibrated_year/ (3 GB) gitignored; added scripts/screen/README.md, scripts/demographic/README.md and docs/STATUS.md (research status, main model = demographic at alpha 2.5, sigma 1.70, D 0.1). Since 2026-11-08 the repo copy .claude/memory/ and the local memory dir hold the same merged notes.
2026-10-11: user wants "a few files with a clear story and a cemetery file". Added docs/STORY.md (narrative, results table, limits, where code lives, next steps) and docs/CEMETERY.md (all abandoned models/protocols with result + reason); docs/STATUS.md removed; README starts from these. Keep both updated when results change or a model is dropped.
Script cleanup 2026-10-11 (commit 08ddf2b, pushed to origin main and theory-phase-diagram; tag archive/pre-cleanup = state before, pushed): scripts/ = demographic/ (main), phase/, lifecycle/ (beta_spells moved here), thesis/ (thesis figure generators + notebooks incl. stale story.ipynb). 51 files removed (screen, zerosum, beta_*, scans, probes, 11 lifecycle, batch/paper_figures, factsheet, 4 notebooks). Shared helpers now relative_glv.msb.{lives,zeta_bins,hill,excess_kurtosis,tail_exponent} and relative_glv.model.configuration_coupling; msb_facts.py simulates its own economies at sigma 1.70 N 16k. User wants few files: docs/STORY.md + docs/CEMETERY.md + scripts/README.md are the maps.

2026-10-12: repo restructured into notebooks (see [[feedback-notebooks-not-scripts]]); every other branch and the archive/* tags were deleted locally and on origin at the user's request. Only `main` exists; removed code is reachable only through main's history (`git log --all -- <path>`).
