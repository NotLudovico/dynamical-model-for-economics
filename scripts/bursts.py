"""Spread between economies and collective bursts (Appendix: protocol) for the production run (N 32000).
1. Exponents of each production economy (lives with >= 5 growth rates): spread between the quiet economies against a
   bootstrap over lives within one economy, and the distance of the burst economy in units of that spread.
2. Bursts: graph 7 re-run with other noise and with half the step, graphs 0 and 4 as controls; per sample the growth
   rate of the mean firm, the largest firm and its node.  Each run takes 8-12 min; results are cached.
Run: uv run python scripts/bursts.py"""
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import numpy as np
from joblib import Parallel, delayed
from economy import DATA, configuration_coupling, integrate_demographic, zeta_bins

PROD = DATA / "demographic" / "robustness"
OUT = DATA / "demographic" / "bursts"
BURST = 0.3                                   # aggregate volatility per year above which an economy is excluded
RUNS = {"g7_n7": (7, 7, 0.01), "g7_n107": (7, 107, 0.01), "g7_n207": (7, 207, 0.01), "g7_n7_dt005": (7, 7, 0.005),
        "g0_n100": (0, 100, 0.01), "g4_n4": (4, 4, 0.01)}   # label: graph seed, noise seed, step


def spreads(n_boot=60, seed=0):
    rng = np.random.default_rng(seed)
    rows, boot, agg = [], [], []
    for s in range(8):
        d = np.load(PROD / f"prod_seed{s}.npz")
        m = d["n"] >= 5
        Sb, v = d["Sbar"][m].astype(float), d["v"][m].astype(float)
        z = zeta_bins(Sb, v)[0]
        rows.append(np.r_[z, z[1:] / z[0]])
        agg.append(float(d["aggvol"]))
        if agg[-1] <= BURST:
            b = []
            for _ in range(n_boot):
                i = rng.integers(0, Sb.size, Sb.size)
                zb = zeta_bins(Sb[i], v[i])[0]
                b.append(np.r_[zb, zb[1:] / zb[0]])
            boot.append(np.std(b, 0))
    R, agg = np.array(rows), np.array(agg)
    quiet = R[agg <= BURST]
    mean, sd = quiet.mean(0), quiet.std(0)
    names = ["zeta1", "zeta2", "zeta3", "zeta4", "zeta2/zeta1", "zeta3/zeta1", "zeta4/zeta1"]
    print(f"{'':12} {'mean':>7} {'sd between':>11} {'sd bootstrap':>13}")
    for k, nm in enumerate(names):
        print(f"{nm:12} {mean[k]:7.3f} {sd[k]:11.3f} {np.mean(boot, 0)[k]:13.4f}")
    for s in range(8):
        print(f"economy {s}: aggregate volatility {agg[s]:.3f}, zeta1 {(R[s, 0] - mean[0]) / sd[0]:+.1f} sd, "
              f"zeta4/zeta1 {(R[s, 6] - mean[6]) / sd[6]:+.1f} sd")


def run(label, graph, noise, dt):
    f = OUT / f"{label}.npz"
    if f.exists():
        return f
    a = configuration_coupling(32000, 1.76, 1.70, alpha=2.5, mean_degree=200, seed=graph)
    r = integrate_demographic(a, D=0.1, tmax=580.0, t_start=100.0, dt_save=0.5, dt=dt, seed=noise, compact=True)
    S = r["S"].astype(float)
    fbar = np.array([S[:, k] @ (1.0 - S[:, k] - a @ S[:, k]) / S.shape[0] for k in range(S.shape[1])])
    np.savez(f, fbar=fbar, maxS=S.max(0), top=np.argmax(S, 0), deg=np.diff(a.indptr))
    return f


def bursts():
    OUT.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=len(RUNS))(delayed(run)(k, *v) for k, v in RUNS.items())
    for label in RUNS:
        d = np.load(OUT / f"{label}.npz")
        big = d["maxS"] > 50
        print(f"{label:12} aggregate volatility {np.std(d['fbar'][:-1] * 0.5):.3f}  largest firm: median "
              f"{np.median(d['maxS']):.0f}, max {d['maxS'].max():.0f}  above 50 in {big.mean():.0%} of samples, "
              f"nodes {Counter(d['top'][big].tolist()).most_common(2)}")


if __name__ == "__main__":
    spreads()
    bursts()
