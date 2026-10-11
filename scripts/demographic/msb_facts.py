"""Every MSB 2024 fact on the demographic-noise model at the reference point.

Simulates SEEDS economies (alpha 2.5, sigma 1.70, D 0.1, N 16000, C 200, burn-in 100, window 480,
step 0.5 as the year), keeps the trajectories in data/demographic/msb_facts/, and measures the facts,
numbered as in MSB 2024 (arXiv:2404.15226):
  F1  E[v^q | Sbar] ~ Sbar^-zeta_q over all 25 bins: zeta = 0.20, 0.39, 0.51, 0.58; lives with >= 2
      growth rates (baseline) and >= 20 (robustness); volatility as sqrt(pi/2) MAD and as SD.
  F2  the largest bin lies above the fitted line; mean volatility 1.22 (smallest) -> 0.22 (largest).
  F3  sigma/sigmabar(Sbar) collapses across bins, unimodal, inverse-gamma-like with tail
      x^-(1+b), b = 4.6: KS distance between size groups, inverse-gamma MLE shape, Hill index.
  F4  rescaled growth (g - gbar_i)/sigma_i has fat, stretched-exponential tails, the same at every
      size (no Gaussianisation): excess kurtosis and tail exponent by size fifth.
  F5  multi-year rescaled growth has fatter tails than the iid convolution of one-year growth.
  F6  correlation with the aggregate and between firms of similar size rises with size.
Writes data/demographic/msb_facts.npz for the figure (plot_msb_facts.py).  The figure in
thesis/assets/figures/demographic_msb_facts.png was made at the old point (sigma 1.75, N 8000).
Run: uv run python scripts/demographic/msb_facts.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy import stats
from relative_glv.msb import lives, hill, excess_kurtosis as exkurt, tail_exponent

ALPHA, SIGMA, D = 2.5, 1.70, 0.1
N, C, MU, START, TOBS, DT, SEEDS = 16000, 200, 1.76, 100.0, 480.0, 0.5, 3
TRAJ = os.path.join(ROOT, "data", "demographic", "msb_facts")
OUT = os.path.join(ROOT, "data", "demographic", "msb_facts.npz")
MSB_ZETA = np.array([0.20, 0.39, 0.51, 0.58])


def zeta_fit(Sb, v, nb=25):
    Pb = np.array_split(np.argsort(Sb), nb)
    lx = np.log([Sb[p].mean() for p in Pb])
    out, resid = [], None
    for q in (1, 2, 3, 4):
        ly = np.log([np.mean(v[p] ** q) for p in Pb])
        c = np.polyfit(lx, ly, 1)
        out.append(-c[0])
        if q == 1:
            resid = ly - np.polyval(c, lx)
    return np.array(out), resid, np.exp(lx), np.array([v[p].mean() for p in Pb]), Pb


def simulate(seed):
    """One economy at the reference point: relative sizes S, death counters L, aggregate growth fbar."""
    from relative_glv import model
    f = os.path.join(TRAJ, f"seed{seed}.npz")
    if not os.path.exists(f):
        a = model.configuration_coupling(N, MU, SIGMA, alpha=ALPHA, mean_degree=C, seed=seed)
        r = model.integrate_demographic(a, D=D, tmax=START + TOBS, t_start=START, dt_save=DT, seed=seed, compact=True)
        S = r["S"]
        fb = np.array([float(S[:, k].astype(float) @ (1.0 - S[:, k] - a @ S[:, k].astype(float))) / N
                       for k in range(S.shape[1])])
        np.savez_compressed(f, S=S, L=r["life"], fbar=fb)
    return f


def main():
    os.makedirs(TRAJ, exist_ok=True)
    from joblib import Parallel, delayed
    files = Parallel(n_jobs=SEEDS)(delayed(simulate)(s) for s in range(SEEDS))
    data = [np.load(f) for f in files]
    print("loading lives ...", flush=True)
    per = [lives(d["S"].astype(float), d["L"], 1) for d in data]
    allv = [x for e in per for x in e]
    Sb = np.array([x[0] for x in allv]); v = np.array([x[1] for x in allv]); n = np.array([x[2] for x in allv])
    sd = np.array([x[3].std(ddof=1) if x[2] >= 2 else np.nan for x in allv])
    res = {}

    print("\nF1/F2  E[v^q | Sbar] ~ Sbar^-zeta_q, all 25 bins     (MSB: 0.20 0.39 0.51 0.58; vol 1.22 -> 0.22)")
    for label, vol in (("MAD", v), ("SD", sd)):
        for nm in (2, 5, 20):
            sel = (n >= nm) & np.isfinite(vol) & (vol > 0)
            z, r, bs, bv, _ = zeta_fit(Sb[sel], vol[sel])
            print(f"  {label} lives>={nm:2d}: zeta {' '.join(f'{x:.3f}' for x in z)}  ratios "
                  f"{' '.join(f'{x:.2f}' for x in z[1:] / z[0])}  largest bin vs line {r[-1]:+.3f}  "
                  f"vol {bv[0]:.2f} -> {bv[-1]:.2f}  ({sel.sum()} lives)")
            res[f"bins_{label}_{nm}"] = np.array([bs, bv]); res[f"zeta_{label}_{nm}"] = z

    print("\nF3  collapse of sigma/sigmabar across size        (MSB: collapses, unimodal, inverse gamma b = 4.6)")
    for nm in (2, 20):
        sel = n >= nm
        _, _, _, _, Pb = zeta_fit(Sb[sel], v[sel])
        vv = v[sel]; r = np.empty(vv.size); grp = np.empty(vv.size, int)
        for i, p in enumerate(Pb):
            r[p] = vv[p] / vv[p].mean(); grp[p] = i // 5
        ks = max(stats.ks_2samp(r[grp == i], r[grp == j]).statistic for i in range(5) for j in range(i + 1, 5))
        b_ig, _, _ = stats.invgamma.fit(r, floc=0)
        h, e = np.histogram(np.log(r), bins=60)
        modes = int(np.sum((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]) & (h[1:-1] > 0.05 * h.max())))
        print(f"  lives>={nm:2d}: max KS distance between size fifths {ks:.3f}; inverse-gamma shape b = {b_ig:.2f}; "
              f"Hill top 5% {hill(r, 0.05):.2f}; local maxima of log-density {modes}")
        res[f"ratio_{nm}"] = r.astype(np.float32); res[f"group_{nm}"] = grp.astype(np.int8)

    print("\nF4  rescaled growth by size fifth (lives>=10)     (MSB: fat stretched-exp tails, same at every size)")
    L10 = [x for x in allv if x[2] >= 10]
    sb = np.array([x[0] for x in L10]); fifth = np.digitize(sb, np.quantile(sb, [0.2, 0.4, 0.6, 0.8]))
    k1, te, Z = [], [], []
    for f5 in range(5):
        z = np.concatenate([x[3] / x[1] for x, ff in zip(L10, fifth) if ff == f5])
        k1.append(exkurt(z)); te.append(tail_exponent(z)); Z.append(z[:400_000].astype(np.float32))
    print("  excess kurtosis: " + " ".join(f"{x:.2f}" for x in k1))
    print("  stretched-exponential tail exponent (1 = exponential, 2 = Gaussian): " + " ".join(f"{x:.2f}" for x in te))
    for f5 in range(5):
        res[f"z_{f5}"] = Z[f5]

    print("\nF5  persistence: excess kurtosis of rescaled k-step growth vs iid k1/k (lives>=20)")
    L20 = [x for x in allv if x[2] >= 20]
    sb = np.array([x[0] for x in L20]); fifth = np.digitize(sb, np.quantile(sb, [0.2, 0.4, 0.6, 0.8]))
    pers = np.zeros((5, 4, 2))
    for f5 in range(5):
        rr = [x[3] / x[1] for x, ff in zip(L20, fifth) if ff == f5]
        kk1 = exkurt(np.concatenate(rr))
        for j, k in enumerate((1, 2, 3, 4)):
            zk = np.concatenate([np.convolve(a, np.ones(k), "valid")[::k] for a in rr if a.size >= k])
            pers[f5, j] = exkurt(zk / zk.std()), kk1 / k
        print(f"  fifth {f5}: " + "  ".join(f"k={k}: {pers[f5, j, 0]:.2f} (iid {pers[f5, j, 1]:.2f})" for j, k in enumerate((1, 2, 3, 4))))
    res["persistence"] = pers

    print("\nF6  co-movement by size fifth                     (MSB: rises with size)")
    W, rng = 40, np.random.default_rng(0)
    cagg = [[] for _ in range(5)]; cpair = [[] for _ in range(5)]
    for d in data:
        S = d["S"].astype(float); L = d["L"]; G = d["fbar"][:-1] * DT
        g = np.diff(np.log(S), axis=1) + G[None, :]
        for a in range(0, S.shape[1] - W - 1, W):
            alive = np.all(L[:, a:a + W + 1] == L[:, [a]], axis=1)
            idx = np.flatnonzero(alive)
            if idx.size < 100:
                continue
            ms = S[idx, a:a + W + 1].mean(1)
            q = np.digitize(ms, np.quantile(ms, [0.2, 0.4, 0.6, 0.8]))
            agg = G[a:a + W]
            for f5 in range(5):
                ii = idx[q == f5]
                if ii.size < 20:
                    continue
                ii = rng.choice(ii, min(300, ii.size), replace=False)
                gg = g[ii, a:a + W]
                cagg[f5] += [np.corrcoef(row, agg)[0, 1] for row in gg if row.std() > 0]
                c = np.corrcoef(gg)
                cpair[f5].append(np.nanmean(c[np.triu_indices_from(c, 1)]))
    ca = [np.nanmean(x) for x in cagg]; cp = [np.nanmean(x) for x in cpair]
    print("  corr with aggregate:            " + " ".join(f"{x:.3f}" for x in ca))
    print("  mean pairwise corr within fifth: " + " ".join(f"{x:.4f}" for x in cp))
    res["corr_agg"] = np.array(ca); res["corr_pair"] = np.array(cp)
    np.savez_compressed(OUT, **res)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
