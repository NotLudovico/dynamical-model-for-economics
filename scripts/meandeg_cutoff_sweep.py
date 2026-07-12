"""MSB observables for one mean-degree power-law cutoff arm.

Run matched arms with identical seeds, then compare the tagged NPZ files:

    MCS_N=10000 MCS_SEEDS=10 MCS_MIN_DEGREE=1 MCS_TAG=_N10000_k1 \
        uv run python scripts/meandeg_cutoff_sweep.py

Omit ``MCS_MIN_DEGREE`` for the legacy graph construction. The interaction
ensemble is always ``kind='powerlaw'`` with mean degree C=N/40.
"""
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import kurtosis

from relative_glv.model import coupling, integrate
from relative_glv.msb import size_volatility, tent_stats

ROOT = Path(__file__).resolve().parents[1]
SMOKE = os.environ.get("MCS_SMOKE", "0") == "1"
N = 1500 if SMOKE else int(os.environ.get("MCS_N", 10000))
SEEDS = 2 if SMOKE else int(os.environ.get("MCS_SEEDS", 10))
N_JOBS = int(os.environ.get("MCS_NJOBS", 6))
TMAX, N_EVAL = (100.0, 400) if SMOKE else (250.0, 700)
LATE = (75.0, 98.0) if SMOKE else (180.0, 240.0)
MU, SIGMA, LAM = 1.76, 1.75, 1e-3
CDEG = max(2, round(N / 40))
MIN_DEGREE = int(os.environ["MCS_MIN_DEGREE"]) if "MCS_MIN_DEGREE" in os.environ else None
TAG = os.environ.get("MCS_TAG", "_smoke" if SMOKE else "")
OUTDIR = Path(os.environ.get("MCS_OUTDIR", ROOT / "data"))
NB = 20
MIN_DECLINE = 50 if SMOKE else 200


def _conditional_stats(size, volatility, growth):
    order = np.argsort(size)
    bins = np.array_split(order, NB)
    bin_size = np.array([size[index].mean() for index in bins])
    bin_volatility = np.array([np.mean(volatility[index]) for index in bins])
    decline = size > bin_size[int(np.argmax(bin_volatility))]
    if decline.sum() < MIN_DECLINE:
        return None

    size_decline, vol_decline = size[decline], volatility[decline]
    order = np.argsort(size_decline)
    bins = np.array_split(order, max(6, NB // 2))
    bin_size = np.array([size_decline[index].mean() for index in bins])
    moments = []
    for moment in (1, 2, 3, 4):
        values = np.array([np.mean(vol_decline[index] ** moment) for index in bins])
        moments.append(-np.polyfit(np.log10(bin_size), np.log10(values), 1)[0])
    moments = np.array(moments)

    mean_volatility = np.interp(size_decline, bin_size, [
        np.mean(vol_decline[index]) for index in bins
    ])
    rescaled_volatility = vol_decline / mean_volatility

    residual_growth = growth - growth.mean(axis=1, keepdims=True)
    unmixed_growth = residual_growth / volatility[:, None]
    order = np.argsort(size_decline)
    size_bins = np.array_split(order, 6)
    kurtosis_by_size = np.array([
        kurtosis(unmixed_growth[decline][index].ravel()) for index in size_bins
    ])
    kurtosis_size = np.array([size_decline[index].mean() for index in size_bins])
    trend = np.polyfit(np.log10(kurtosis_size), kurtosis_by_size, 1)[0]

    return {
        "ratio": moments / moments[0],
        "d1_tail": np.percentile(rescaled_volatility, 99) / np.median(rescaled_volatility),
        "d3_trend": trend,
        "d3_kurtosis": kurtosis_by_size,
    }


def simulate(seed):
    alpha = coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=CDEG,
                     min_degree=MIN_DEGREE)
    result = integrate(alpha, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                       method="RK45", rtol=1e-4, atol=1e-7)
    if not result["success"]:
        return None
    summary = size_volatility(result["W"], result["t"], window=LATE, dt=0.5, n_bins=NB)
    if summary["growth"].size == 0 or not np.isfinite(summary["beta"]):
        return None
    conditional = _conditional_stats(summary["Sbar"], summary["vol"], summary["growth"])
    if conditional is None:
        return None
    return {
        "beta": summary["beta"],
        "survival": summary["Sbar"].size / N,
        "bowley": tent_stats(summary["growth"])["bowley"],
        "exkurt": tent_stats(summary["growth"])["exkurt"],
        **conditional,
    }


def main():
    label = "legacy" if MIN_DEGREE is None else str(MIN_DEGREE)
    print(
        f"mean-degree cutoff arm: N={N}, C={CDEG}, k_min={label}, seeds={SEEDS}, "
        f"window={LATE}",
        flush=True,
    )
    records = [record for record in Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(simulate)(seed) for seed in range(SEEDS)
    ) if record is not None]
    fields = ("beta", "survival", "bowley", "exkurt", "ratio", "d1_tail", "d3_trend", "d3_kurtosis")
    output = {field: np.array([record[field] for record in records]) for field in fields}
    print(f"completed {len(records)}/{SEEDS} economies")
    if not records:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        path = OUTDIR / f"meandeg_cutoff{TAG}.npz"
        np.savez(
            path,
            N=N, C=CDEG, min_degree=-1 if MIN_DEGREE is None else MIN_DEGREE,
            seeds=0, attempted_seeds=SEEDS, mu=MU, sigma=SIGMA, lam=LAM,
            late=np.array(LATE), **output,
        )
        print(f"saved {path}")
        return
    print(f"beta={np.median(output['beta']):.3f} ± {np.std(output['beta']):.3f}")
    print(f"survival={np.mean(output['survival']):.3f} ± {np.std(output['survival']):.3f}")
    print(f"zeta ratios={np.mean(output['ratio'], axis=0)}")
    print(f"D1 tail={np.median(output['d1_tail']):.2f}; D3 trend={np.median(output['d3_trend']):+.2f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"meandeg_cutoff{TAG}.npz"
    np.savez(
        path,
        N=N, C=CDEG, min_degree=-1 if MIN_DEGREE is None else MIN_DEGREE,
        seeds=len(records), attempted_seeds=SEEDS, mu=MU, sigma=SIGMA, lam=LAM,
        late=np.array(LATE), **output,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
