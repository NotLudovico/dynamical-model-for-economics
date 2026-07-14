"""Stationary relative firm-size distribution for the thesis mean-degree GLV.

Each economy is measured separately over a late stationary window.  CCDFs are
then averaged with equal economy weights; firms are never pooled across the
disconnected simulation economies.  ``ACTIVE_MIN`` is only a reporting cut
used for the tail panel, not a graph-degree cutoff.

Run from the repository root:
    uv run python scripts/size_distribution.py

For a quick check:
    SD_SMOKE=1 uv run python scripts/size_distribution.py
"""
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relative_glv.model import coupling, integrate
from relative_glv.size_distribution import economy_ccdf, hill_tail_index

SMOKE = os.environ.get("SD_SMOKE", "0") == "1"
MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N = 1200 if SMOKE else int(os.environ.get("SD_N", 10000))
SEEDS = 2 if SMOKE else int(os.environ.get("SD_SEEDS", 10))
TMAX, N_EVAL = (100.0, 400) if SMOKE else (250.0, 700)
LATE = (75.0, 98.0) if SMOKE else (180.0, 240.0)
NJOBS = int(os.environ.get("SD_NJOBS", 2 if SMOKE else 6))
CDEG = max(2, round(100 * N / 4000))
ACTIVE_MIN = float(os.environ.get("SD_ACTIVE_MIN", 0.1))
TAIL_FRACTION = float(os.environ.get("SD_TAIL_FRACTION", 0.02))
OUTDIR = Path(os.environ.get("SD_OUTDIR", ROOT / "data"))
TAG = os.environ.get("SD_TAG", "_smoke" if SMOKE else "")


def simulate(seed):
    alpha = coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=CDEG)
    result = integrate(alpha, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                       method="RK45", rtol=1e-4, atol=1e-7)
    if not result["success"]:
        return None
    late = (result["t"] >= LATE[0]) & (result["t"] <= LATE[1])
    return (N * result["W"][:, late]).ravel().astype(np.float32)


def mean_interval(by_economy):
    mean = by_economy.mean(axis=0)
    if by_economy.shape[0] == 1:
        return mean, mean, mean
    half_width = student_t.ppf(0.975, by_economy.shape[0] - 1) * (
        by_economy.std(axis=0, ddof=1) / np.sqrt(by_economy.shape[0])
    )
    return mean, np.clip(mean - half_width, 0.0, 1.0), np.clip(mean + half_width, 0.0, 1.0)


def plot_distribution(full, active, grid, active_grid, tail_index, tail_start, output):
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    full_mean, full_low, full_high = mean_interval(full["by_economy"])
    active_mean, active_low, active_high = mean_interval(active["by_economy"])

    left, right = axes
    left.plot(grid, full_mean, color="#4472c4", label="equal-weight economy mean")
    left.fill_between(grid, full_low, full_high, color="#4472c4", alpha=0.18, label="95% t interval")
    left.axvline(ACTIVE_MIN, color="#555555", linestyle="--", linewidth=1,
                label=rf"reporting cut $S={ACTIVE_MIN:g}$")
    left.set(xscale="log", yscale="log", xlabel=r"relative firm size $S_i=Nw_i$",
             ylabel=r"$P(S_i \geq S)$", title="All late-window firm observations")
    left.legend(fontsize=7.5)

    right.plot(active_grid, active_mean, color="#c0504d", label="active firms only")
    right.fill_between(active_grid, active_low, active_high, color="#c0504d", alpha=0.18,
                       label="95% t interval")
    reference_x = float(np.median(tail_start))
    reference_y = float(np.interp(reference_x, active_grid, active_mean))
    zipf = reference_y * (active_grid / reference_x) ** -1
    right.plot(active_grid, zipf, "--", color="#555555", linewidth=1,
               label="Zipf guide (CCDF slope −1)")
    right.axvline(reference_x, color="#555555", linestyle=":", linewidth=1)
    right.set(xscale="log", yscale="log", xlabel=r"relative firm size $S_i=Nw_i$",
              ylabel=rf"$P(S_i \geq S\mid S_i\geq {ACTIVE_MIN:g})$",
              title=rf"Upper tail; mean Hill exponent $\hat{{\mu}}={tail_index.mean():.2f}$")
    right.legend(fontsize=7.5)
    figure.suptitle(
        f"Mean-degree relative GLV size distribution (N={N}, {len(tail_index)} economies, "
        f"$\\mu$={-MU}, $\\sigma$={SIGMA})",   # thesis sign convention (mu<0 competitive)
        fontsize=11,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    print(
        f"size distribution | N={N}, C={CDEG}, seeds={SEEDS}, mu={MU}, sigma={SIGMA}, "
        f"late={LATE}, active cut={ACTIVE_MIN}",
        flush=True,
    )
    sizes_by_economy = [sizes for sizes in Parallel(n_jobs=NJOBS, backend="loky")(
        delayed(simulate)(seed) for seed in range(SEEDS)
    ) if sizes is not None]
    if not sizes_by_economy:
        raise RuntimeError("all simulations failed")

    grid = np.logspace(-6, np.log10(N), 360)
    active_by_economy = [sizes[sizes >= ACTIVE_MIN] for sizes in sizes_by_economy]
    if any(sizes.size < 2 for sizes in active_by_economy):
        raise RuntimeError("active reporting cut removed an entire economy")
    active_grid = np.logspace(np.log10(ACTIVE_MIN), np.log10(N), 360)
    full = economy_ccdf(sizes_by_economy, grid)
    active = economy_ccdf(active_by_economy, active_grid)
    tail_start = np.array([np.quantile(sizes, 1 - TAIL_FRACTION) for sizes in active_by_economy])
    tail_index = np.array([
        hill_tail_index(sizes[sizes >= cutoff])
        for sizes, cutoff in zip(active_by_economy, tail_start, strict=True)
    ])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    output = OUTDIR / f"size_distribution{TAG}.npz"
    np.savez_compressed(
        output,
        N=N, C=CDEG, seeds=len(sizes_by_economy), mu=MU, sigma=SIGMA, lam=LAM,
        late=np.array(LATE), active_min=ACTIVE_MIN, tail_fraction=TAIL_FRACTION,
        grid=grid, full_ccdf=full["mean"], full_ccdf_by_economy=full["by_economy"],
        active_grid=active_grid, active_ccdf=active["mean"],
        active_ccdf_by_economy=active["by_economy"], tail_start=tail_start,
        tail_index=tail_index,
    )
    image = output.with_suffix(".png")
    plot_distribution(full, active, grid, active_grid, tail_index, tail_start, image)
    print(
        f"completed {len(sizes_by_economy)}/{SEEDS} economies; "
        f"mean Hill CCDF exponent={tail_index.mean():.3f} ± {tail_index.std(ddof=1):.3f}; "
        f"tail starts at S={np.median(tail_start):.2f}",
    )
    print(f"saved {output} and {image}")


if __name__ == "__main__":
    main()
