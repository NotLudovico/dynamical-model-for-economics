---
name: feedback-figures
description: User wants fewer, more beautiful paper figures; the shared style in economy.py and the 3+2 figure set
metadata:
  type: feedback
---

User (2026-10-12): "less but more beautiful plots". Paper now has 3 main figures (fig_phase, fig_mechanism, fig_facts,
3 panels max) + 2 SM (figS_phase_lines, figS_reduction), vector PDF in paper/figures, written by notebooks 02/04/05/06
via economy.paper_style(), panel(), save_paper(), PAL, FIFTHS, COL1/COL2 (Computer Modern 8 pt, revtex widths).
Colours: blue = model/power-law graph, orange = regular graph, aqua = Compustat (validated all-pairs; aqua always
with a marker or direct label); size fifths = sequential blue ramp; reference-point star in ink.

**Why:** dense multi-panel PNGs (5-panel facts figure with a twin axis) read as cluttered.
**How to apply:** numbers that a dropped panel showed go in tables (Table I, S6), not new panels; direct labels over
legends; no twin axes; check every new figure by rendering it before saving. Related: [[project-demographic-paper]].
