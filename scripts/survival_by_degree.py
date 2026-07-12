"""Study survival probability conditional on network degree.

The experiment uses the thesis mean-degree-normalised power-law network at the
locked operating point. A firm is counted as surviving when its relative share
remains above ``FLOOR`` throughout the late stationary window.

Run from the repository root:
    uv run python scripts/survival_by_degree.py

For a quick reproducibility check:
    SBD_SMOKE=1 uv run python scripts/survival_by_degree.py
"""
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relative_glv.degree_survival import survival_by_degree_by_economy
from relative_glv.model import coupling, integrate

SMOKE = os.environ.get("SBD_SMOKE", "0") == "1"
MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N = 1200 if SMOKE else int(os.environ.get("SBD_N", 8000))
SEEDS = 2 if SMOKE else int(os.environ.get("SBD_SEEDS", 10))
TMAX, N_EVAL = (100.0, 400) if SMOKE else (250.0, 700)
LATE = (75.0, 98.0) if SMOKE else (180.0, 240.0)
NJOBS = int(os.environ.get("SBD_NJOBS", 2 if SMOKE else 6))
NBIN = int(os.environ.get("SBD_NBIN", 10))
FLOOR = float(os.environ.get("SBD_FLOOR", 1e-6))
CDEG = max(2, round(100 * N / 4000))
MIN_DEGREE = int(os.environ["SBD_MIN_DEGREE"]) if "SBD_MIN_DEGREE" in os.environ else None
OUTDIR = Path(os.environ.get("SBD_OUTDIR", ROOT / "data"))
TAG = os.environ.get("SBD_TAG", "_smoke" if SMOKE else "")


def simulate(seed):
    alpha = coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=CDEG,
                     min_degree=MIN_DEGREE)
    result = integrate(alpha, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                       method="RK45", rtol=1e-4, atol=1e-7)
    if not result["success"]:
        return None
    late = (result["t"] >= LATE[0]) & (result["t"] <= LATE[1])
    return {
        "degree": np.diff(alpha.indptr).astype(float),
        "survived": result["W"][:, late].min(axis=1) > FLOOR,
        "seed": seed,
    }


def configure_survival_axes(axis, degree, ci_high):
    degree = np.asarray(degree, dtype=float)
    ci_high = np.asarray(ci_high, dtype=float)
    axis.set_xscale("log")
    axis.set_xlim(left=0.8 * degree.min())
    axis.set_ylim(-0.005, 1.15 * ci_high.max())


def plot_summary(summary, overall, *, N, economies, min_degree, output):
    figure, axis = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    axis.plot(summary["degree"], summary["probability"], "o-", color="#c0504d",
              label=r"mean $P(\mathrm{survive}\mid k)$ across economies")
    axis.fill_between(summary["degree"], summary["ci_low"], summary["ci_high"],
                      color="#c0504d", alpha=0.18, label="economy-level 95% t interval")
    axis.axhline(overall, color="#555555", linestyle="--", linewidth=1,
                 label=f"overall = {overall:.2f}")
    configure_survival_axes(axis, summary["degree"], summary["ci_high"])
    axis.set(xlabel="degree $k$", ylabel="survival probability", title="Survival conditional on degree")
    axis.legend(fontsize=8)
    topology = "legacy power law" if min_degree is None else rf"shifted power law ($k_{{\min}}$={min_degree})"
    figure.suptitle(f"Mean-degree GLV: {topology} (N={N}, {economies} economies, "
                     f"$\\mu$={MU}, $\\sigma$={SIGMA})", fontsize=11)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main():
    print(
        f"survival by degree | N={N}, C={CDEG}, "
        f"k_min={MIN_DEGREE if MIN_DEGREE is not None else 'legacy'}, seeds={SEEDS}, "
        f"mu={MU}, sigma={SIGMA}, late={LATE}",
        flush=True,
    )
    records = [record for record in Parallel(n_jobs=NJOBS, backend="loky")(
        delayed(simulate)(seed) for seed in range(SEEDS)
    ) if record is not None]
    if not records:
        raise RuntimeError("all simulations failed")

    degree = np.concatenate([record["degree"] for record in records])
    survived = np.concatenate([record["survived"] for record in records])
    seed = np.concatenate([
        np.full(record["degree"].size, record["seed"], dtype=int) for record in records
    ])
    edges = np.logspace(np.log10(degree.min()), np.log10(degree.max() + 1), NBIN + 1)
    summary = survival_by_degree_by_economy(degree, survived, seed, bin_edges=edges)
    overall = float(np.mean([record["survived"].mean() for record in records]))
    rho_by_economy = np.array([
        spearmanr(record["degree"], record["survived"]).statistic for record in records
    ])
    rho = float(np.nanmean(rho_by_economy))
    populated = summary["count"] > 0
    low, high = np.flatnonzero(populated)[[0, -1]]
    print(f"completed {len(records)}/{SEEDS} economies; overall survival={overall:.3f}")
    print(
        f"P(survive|k): {summary['probability'][low]:.3f} at k≈{summary['degree'][low]:.0f} "
        f"-> {summary['probability'][high]:.3f} at k≈{summary['degree'][high]:.0f}; "
        f"mean per-economy Spearman rho={rho:+.3f}",
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    output = OUTDIR / f"survival_by_degree{TAG}.npz"
    np.savez(
        output,
        N=N, C=CDEG, min_degree=-1 if MIN_DEGREE is None else MIN_DEGREE,
        seeds=len(records), mu=MU, sigma=SIGMA, lam=LAM,
        late=np.array(LATE), floor=FLOOR, overall=overall, spearman_rho=rho,
        spearman_rho_by_economy=rho_by_economy,
        bin_edges=edges, raw_degree=degree, raw_survived=survived, raw_seed=seed,
        **summary,
    )

    image = output.with_suffix(".png")
    plot_summary(summary, overall, N=N, economies=len(records), min_degree=MIN_DEGREE,
                 output=image)
    print(f"saved {output} and {image}")


if __name__ == "__main__":
    main()
