---
name: feedback-parallel-screening
description: User wants many candidate modifications proposed and checked in parallel on their machine (12 cores, 15 GB, AMD Polaris GPU) via a shared harness and fixed scorecard
metadata:
  type: feedback
---
User (2026-10-11): "propose every modification that comes to your mind and write a code to check them in parallel so we take advantage of this powerful pc". Agreed conditions: one harness with every variant written in x, a scorecard fixed before running, at most one new parameter per variant on a coarse grid, a ranked table that includes the failures, and winners confirmed at larger N and with more seeds.

**Why:** serial one-model-at-a-time testing was slow; compute is cheap and the user would rather see a broad screen.
**How to apply:** add variants to scripts/screen/run.py GRID (models in scripts/screen/variants.py, scorecard in score.py; results cached in data/screen/, gitignored). Run with joblib on 12 workers. The GPU is an AMD RX 470-590 (gfx803): no ROCm/torch support and no OpenCL ICD installed, so use the CPU unless large-N batching makes OpenCL (opencl-mesa + pyopencl, float32) worth installing. Related: [[feedback-minimal-parameters]], [[feedback-write-models-in-x]].
