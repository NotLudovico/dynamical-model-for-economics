"""Measure true threshold re-entry in the thesis mean-degree model.

A re-entry requires an active -> inactive -> active sequence within the late
window, so first-time activation does not count as churn.
"""
import os
from pathlib import Path

import matplotlib
import numpy as np
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relative_glv.degree_survival import reentry_masks, survival_by_degree_by_economy
from relative_glv.model import coupling, integrate

ROOT = Path(__file__).resolve().parents[1]
SMOKE = os.environ.get("RBD_SMOKE", "0") == "1"
N = 1200 if SMOKE else int(os.environ.get("RBD_N", 10000))
SEEDS = 2 if SMOKE else int(os.environ.get("RBD_SEEDS", 10))
N_JOBS = int(os.environ.get("RBD_NJOBS", 2 if SMOKE else 6))
TMAX, N_EVAL = (100.0, 400) if SMOKE else (250.0, 700)
LATE = (75.0, 98.0) if SMOKE else (180.0, 240.0)
MU, SIGMA, LAM, FLOOR = 1.76, 1.75, 1e-3, 1e-6
CDEG = max(2, round(N / 40))
TAG = os.environ.get("RBD_TAG", "_smoke" if SMOKE else "_N10000")
OUTDIR = Path(os.environ.get("RBD_OUTDIR", ROOT / "data"))


def simulate(seed):
    alpha = coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=CDEG)
    result = integrate(alpha, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                       method="RK45", rtol=1e-4, atol=1e-7)
    if not result["success"]:
        return None
    late = (result["t"] >= LATE[0]) & (result["t"] <= LATE[1])
    active = result["W"][:, late] > FLOOR
    return {
        "degree": np.diff(alpha.indptr).astype(float),
        "ever_active": active.any(axis=1),
        "seed": seed,
        **reentry_masks(active),
    }


def _summary(records, edges, denominator, outcome):
    degree = np.concatenate([record["degree"][record[denominator]] for record in records])
    event = np.concatenate([record[outcome][record[denominator]] for record in records])
    seed = np.concatenate([
        np.full(record[denominator].sum(), record["seed"], dtype=int) for record in records
    ])
    return survival_by_degree_by_economy(degree, event, seed, bin_edges=edges)


def main():
    print(f"re-entry study | N={N}, C={CDEG}, seeds={SEEDS}, late={LATE}", flush=True)
    records = [record for record in Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(simulate)(seed) for seed in range(SEEDS)
    ) if record is not None]
    if not records:
        raise RuntimeError("all simulations failed")

    degree = np.concatenate([record["degree"] for record in records])
    edges = np.logspace(np.log10(degree.min()), np.log10(degree.max() + 1), 10 + 1)
    reentry = _summary(records, edges, "exited", "reentered")
    exit_rate = _summary(records, edges, "ever_active", "exited")
    active_rate = _summary(records, edges, "persistent", "persistent")
    exited = np.array([record["exited"].sum() for record in records])
    reentered = np.array([record["reentered"].sum() for record in records])
    ever_active = np.array([record["ever_active"].sum() for record in records])
    print(f"completed {len(records)}/{SEEDS} economies")
    print(f"ever active: {np.mean(ever_active / N):.3f}; exited: {np.mean(exited / ever_active):.3f}; "
          f"re-entered after exit: {np.mean(reentered / np.maximum(exited, 1)):.3f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    output = OUTDIR / f"reentry_by_degree{TAG}.npz"
    np.savez(
        output,
        N=N, C=CDEG, seeds=len(records), mu=MU, sigma=SIGMA, lam=LAM, floor=FLOOR,
        late=np.array(LATE), bin_edges=edges, raw_degree=degree,
        raw_ever_active=np.concatenate([record["ever_active"] for record in records]),
        raw_exited=np.concatenate([record["exited"] for record in records]),
        raw_reentered=np.concatenate([record["reentered"] for record in records]),
        **{f"reentry_{key}": value for key, value in reentry.items()},
        **{f"exit_{key}": value for key, value in exit_rate.items()},
        **{f"active_{key}": value for key, value in active_rate.items()},
    )

    figure, axis = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    axis.semilogx(reentry["degree"], reentry["probability"], "o-", color="#c0504d",
                 label=r"$P(\mathrm{re\!\!\! - \! enter}\mid\mathrm{exited}, k)$")
    axis.fill_between(reentry["degree"], reentry["ci_low"], reentry["ci_high"],
                      color="#c0504d", alpha=0.18, label="economy-level 95% t interval")
    axis.set(xlabel="degree $k$", ylabel="re-entry probability", ylim=(-0.02, 1.02),
             title="Re-entry after a threshold exit")
    axis.legend(fontsize=8)
    figure.suptitle(f"Mean-degree GLV (N={N}, {len(records)} economies, $\\mu$={-MU}, $\\sigma$={SIGMA})")  # thesis sign convention
    image = output.with_suffix(".png")
    figure.savefig(image, dpi=150)
    print(f"saved {output} and {image}")


if __name__ == "__main__":
    main()
