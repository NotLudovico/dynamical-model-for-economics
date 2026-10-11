"""Robustness of the demographic-noise model's MSB facts: system size, seeds, mean degree and the
neighbourhood of the reference point (alpha 2.5, sigma 1.75, D 0.1, C 200, mu 1.76 code convention).

Each job simulates one economy (burn-in 100, window 480, sampled every 0.5 = one year) and
analyses it in the worker; only per-life arrays and summary statistics are kept, in
data/demographic/robustness/<label>_seed<k>.npz.  Graphs with N >= 32000 (and the retune jobs)
use model.configuration_coupling, the numpy configuration model.

Per economy: lives (Sbar, MAD and SD volatility, growth count); by size fifth the excess kurtosis
and stretched-exponential tail exponent of rescaled growth (lives >= 10), the k-step persistence
ratio (lives >= 20, k = 2, 3, 4), correlation with the aggregate and mean pairwise correlation
within the fifth (40-step windows); aggregate volatility per step; lag-1 autocorrelation at steps
of 0.5, 1 and 2; size Hill index, firm-time per decade, largest share; turnover.

Run: uv run python scripts/demographic/robustness.py           (all jobs, 12 workers)
     uv run python scripts/demographic/robustness.py prod      (paper production run at the new reference point)
     uv run python scripts/demographic/robustness.py summary
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

OUT = os.path.join(ROOT, "data", "demographic", "robustness")
MU, START, TOBS, DT = 1.76, 100.0, 480.0, 0.5
REF = dict(alpha=2.5, sigma=1.75, D=0.1, N=8000, C=200)
JOBS = (  # label, overrides of REF, seeds
    ("ref", {}, range(8)),
    ("N16k", {"N": 16000}, range(4)),
    ("N32k", {"N": 32000}, range(6)),
    ("N64k", {"N": 64000}, range(2)),
    ("D.05", {"D": 0.05}, range(4)),
    ("D.2", {"D": 0.2}, range(4)),
    ("s1.7", {"sigma": 1.7}, range(4)),
    ("s1.8", {"sigma": 1.8}, range(4)),
    ("a3", {"alpha": 3.0}, range(4)),
    ("C100", {"C": 100}, range(4)),
    ("C400", {"C": 400}, range(4)),
)
RETUNE = tuple(  # aim: beta ~ 0.2 at a converged size (N = 16000); numpy graph builder for every job
    (f"rt_s{sg}_D{D:g}", {"N": 16000, "sigma": sg, "D": D, "fast": True}, range(4))
    for sg in (1.70, 1.72) for D in (0.1, 0.15, 0.2)
) + (("rt_s1.75_D0.1", {"N": 16000, "fast": True}, range(2)),)   # same point as N16k, other graph builder
PROD = (  # paper reference point (sigma 1.70 after the retune), N = 32000
    ("prod", {"N": 32000, "sigma": 1.70, "fast": True}, range(8)),
)
BURST = 0.3   # economies whose aggregate volatility per step exceeds this are in a collective burst (prod seed 7:
              # 0.54, the other seven 0.03-0.23); the paper's facts exclude them and say so


def kept(ds):
    """Drop economies in a collective burst (aggregate volatility above BURST)."""
    return [d for d in ds if float(d["aggvol"]) <= BURST]
MSB_ZETA = np.array([0.20, 0.39, 0.51, 0.58])


def params(over):
    p = dict(REF); p.update(over); return p


def path(label, seed):
    return os.path.join(OUT, f"{label}_seed{seed}.npz")


def _segs(Li, T):
    return [s for s in np.split(np.arange(T), np.flatnonzero(np.diff(Li)) + 1) if s.size >= 3]


def analyse(S, L, fbar, N, cap=400_000, seed=0):
    """Two streaming passes over firms, so memory stays O(N T) even for 10^7 lives."""
    from relative_glv.msb import hill, tail_exponent
    out = {}
    lnS = np.log(S)
    T = S.shape[1]
    Sb, V, SD, NN = [], [], [], []
    for i in range(S.shape[0]):                                   # pass 1: per-life statistics
        segs = _segs(L[i], T)
        if not segs:
            continue
        g = [np.diff(lnS[i, s]).astype(float) for s in segs]
        Sb.append(np.array([S[i, s].mean() for s in segs], np.float32))
        V.append(np.array([np.sqrt(np.pi / 2) * np.abs(x - x.mean()).mean() for x in g], np.float32))
        SD.append(np.array([x.std(ddof=1) for x in g], np.float32)); NN.append(np.array([x.size for x in g], np.int32))
    out["Sbar"], out["v"], out["sd"], out["n"] = (np.concatenate(Sb), np.concatenate(V), np.concatenate(SD),
                                                  np.concatenate(NN))
    del Sb, V, SD, NN
    ok = out["v"] > 0
    q10 = np.quantile(out["Sbar"][(out["n"] >= 10) & ok], [0.2, 0.4, 0.6, 0.8])
    q20 = np.quantile(out["Sbar"][(out["n"] >= 20) & ok], [0.2, 0.4, 0.6, 0.8])
    cnt10 = np.bincount(np.digitize(out["Sbar"][(out["n"] >= 10) & ok], q10), weights=out["n"][(out["n"] >= 10) & ok], minlength=5)
    keep = np.minimum(1.0, cap / np.maximum(cnt10, 1))
    rng = np.random.default_rng(seed)
    m10 = np.zeros((5, 3)); sub = [[] for _ in range(5)]          # count, sum z^2, sum z^4
    m20 = np.zeros((5, 4, 3))                                     # k = 1..4
    for i in range(S.shape[0]):                                   # pass 2: rescaled growth by size fifth
        for s in _segs(L[i], T):
            g = np.diff(lnS[i, s]).astype(float)
            if g.size < 10:
                continue
            r = g - g.mean(); v = np.sqrt(np.pi / 2) * np.abs(r).mean()
            if v <= 0:
                continue
            z = r / v; sb = S[i, s].mean(); f = int(np.digitize(sb, q10))
            m10[f] += (z.size, (z ** 2).sum(), (z ** 4).sum())
            if rng.random() < keep[f]:
                sub[f].append(z.astype(np.float32))
            if g.size >= 20:
                f2 = int(np.digitize(sb, q20))
                for j, k in enumerate((1, 2, 3, 4)):
                    zk = z[: z.size // k * k].reshape(-1, k).sum(1)
                    m20[f2, j] += (zk.size, (zk ** 2).sum(), (zk ** 4).sum())
    kurt = lambda m: m[..., 2] * m[..., 0] / m[..., 1] ** 2 - 3
    out["kurt"] = kurt(m10)
    out["tailexp"] = np.array([tail_exponent(np.concatenate(x)) for x in sub])
    k20 = kurt(m20)
    out["persist"] = np.array([[k20[f, j] / (k20[f, 0] / k) for j, k in ((1, 2), (2, 3), (3, 4))] for f in range(5)])
    G = fbar[:-1] * DT
    out["aggvol"] = float(np.std(G))
    g = np.diff(np.log(S), axis=1) + G[None, :].astype(np.float32)
    W, rng = 40, np.random.default_rng(0)
    ca = [[] for _ in range(5)]; cp = [[] for _ in range(5)]
    for a in range(0, S.shape[1] - W - 1, W):
        idx = np.flatnonzero(np.all(L[:, a:a + W + 1] == L[:, [a]], axis=1))
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
            gg = g[ii, a:a + W].astype(float)
            ok = gg.std(1) > 0
            ca[f5] += [np.corrcoef(row, agg)[0, 1] for row in gg[ok]]
            c = np.corrcoef(gg[ok]); cp[f5].append(np.nanmean(c[np.triu_indices_from(c, 1)]))
    out["corr_agg"] = np.array([np.nanmean(x) for x in ca]); out["corr_pair"] = np.array([np.nanmean(x) for x in cp])
    lags = []
    for k in (1, 2, 4):
        Sk, Lk = S[:, ::k], L[:, ::k]
        gk = np.diff(np.log(Sk), axis=1)
        same = (Lk[:, 2:] == Lk[:, 1:-1]) & (Lk[:, 1:-1] == Lk[:, :-2])
        lags.append(np.corrcoef(gk[:, :-1][same], gk[:, 1:][same])[0, 1])
    out["lag1"] = np.array(lags)
    allS = S[:, ::5].ravel().astype(float)
    out["hill1"] = hill(allS, 0.01)
    lx = np.log10(allS)
    out["decades"] = np.array([np.mean((lx >= lo) & (lx < lo + 1)) for lo in range(-4, 2)])
    out["topshare"] = float(S.max(0).mean() / N)
    out["turnover"] = float((L[:, -1] - L[:, 0]).sum() / N / TOBS)
    return out


def job(label, over, seed):
    f = path(label, seed)
    if os.path.exists(f):
        return label, seed, "cached"
    t0 = time.time()
    p = params(over)
    from relative_glv import model
    if p["N"] >= 32000 or p.get("fast"):
        a = model.configuration_coupling(p["N"], MU, p["sigma"], alpha=p["alpha"], mean_degree=p["C"], seed=seed)
    else:
        model._ALPHA_PL = p["alpha"]
        a = model.coupling(p["N"], MU, p["sigma"], kind="powerlaw", seed=seed, mean_degree=p["C"])
    r = model.integrate_demographic(a, D=p["D"], tmax=START + TOBS, t_start=START, dt_save=DT, seed=seed, compact=True)
    S, L = r["S"], r["life"]
    del r
    fbar = np.array([float(S[:, k].astype(float) @ (1.0 - S[:, k] - a @ S[:, k].astype(float))) / p["N"]
                     for k in range(S.shape[1])])
    out = analyse(S, L, fbar, p["N"])
    out["secs"] = time.time() - t0
    np.savez_compressed(f, **out)
    return label, seed, f"{out['secs'] / 60:.1f} min"


# ------------------------------------------------------------------ summary
def summary():
    from scipy import stats
    from relative_glv.msb import hill, zeta_bins
    for label, over, seeds in JOBS:
        ds = [np.load(path(label, s)) for s in seeds if os.path.exists(path(label, s))]
        if not ds:
            continue
        if label == "prod":
            n0 = len(ds); ds = kept(ds)
            print(f"\n[prod: {n0 - len(ds)} of {n0} economies in a collective burst (aggregate volatility > {BURST}) excluded]")
        p = params(over)
        print(f"\n=== {label}: alpha {p['alpha']} sigma {p['sigma']} D {p['D']} N {p['N']} C {p['C']}  "
              f"({len(ds)} economies, {np.mean([d['secs'] for d in ds]) / 60:.0f} min each)")
        cat = lambda k: np.concatenate([d[k] for d in ds])
        Sb, v, sd, n = cat("Sbar").astype(float), cat("v").astype(float), cat("sd").astype(float), cat("n")
        for nm, vol, tag in ((2, v, "MAD"), (5, v, "MAD"), (10, v, "MAD"), (20, v, "MAD"), (5, sd, "SD")):
            sel = (n >= nm) & (vol > 0)
            z, res, v0, v1, dec, loc = zeta_bins(Sb[sel], vol[sel])
            Ze = np.array([zeta_bins(d["Sbar"][(d["n"] >= nm)].astype(float), (d["v"] if tag == "MAD" else d["sd"])[(d["n"] >= nm)].astype(float))[0]
                           for d in ds])
            print(f"  {tag} n>={nm:2d}: zeta {' '.join(f'{x:.3f}' for x in z)} (beta sd {Ze[:, 0].std():.3f})  "
                  f"ratios {' '.join(f'{x:.2f}' for x in z[1:] / z[0])}  local {' '.join(f'{x:.2f}' for x in loc)}  "
                  f"top-bin {res:+.2f}  vol {v0:.2f}->{v1:.2f}  {dec:.1f} dec")
        sel = n >= 10
        P = np.array_split(np.argsort(Sb[sel]), 25); vv = v[sel]; r = np.empty(vv.size); grp = np.empty(vv.size, int)
        for i, pp in enumerate(P):
            r[pp] = vv[pp] / vv[pp].mean(); grp[pp] = i // 5
        ks = max(stats.ks_2samp(r[grp == i], r[grp == j]).statistic for i in range(5) for j in range(i + 1, 5))
        r2 = v[n >= 2]; P2 = np.array_split(np.argsort(Sb[n >= 2]), 25); rr2 = np.empty(r2.size)
        for pp in P2:
            rr2[pp] = r2[pp] / r2[pp].mean()
        m = lambda k: np.mean([d[k] for d in ds], 0); s_ = lambda k: np.std([d[k] for d in ds], 0)
        print(f"  sigma/sigmabar: KS max (n>=10) {ks:.3f}; Hill b top 5%: n>=2 {hill(rr2, 0.05):.2f}, n>=10 {hill(r, 0.05):.2f}")
        print(f"  kurtosis by fifth {' '.join(f'{x:.1f}' for x in m('kurt'))};  tail exp {' '.join(f'{x:.2f}' for x in m('tailexp'))}")
        print(f"  persistence ratio k=3 by fifth {' '.join(f'{x:.2f}' for x in m('persist')[:, 1])}")
        print(f"  corr agg {' '.join(f'{x:.3f}' for x in m('corr_agg'))};  pair {' '.join(f'{x:.3f}' for x in m('corr_pair'))}")
        print(f"  aggregate vol/step {m('aggvol'):.4f} ± {s_('aggvol'):.4f};  lag1 at 0.5/1/2: {' '.join(f'{x:+.2f}' for x in m('lag1'))};  "
              f"turnover {m('turnover'):.3f}")
        print(f"  sizes: Hill top1% {m('hill1'):.2f}; firm-time per decade [-4..2): {' '.join(f'{100 * x:.0f}' for x in m('decades'))}%; "
              f"top share {m('topshare'):.4f}")
    print(f"\nMSB: zeta {MSB_ZETA}, vol 1.22 -> 0.22 over 4 decades, top bin above line, b 4.6, no Gaussianisation, "
          "persistence, co-movement rising with size")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:2] == ["retune"]:                 # robustness.py retune [summary]
        JOBS = RETUNE; del sys.argv[1]
    elif sys.argv[1:2] == ["prod"]:                 # robustness.py prod [summary]
        JOBS = PROD; del sys.argv[1]
    if sys.argv[1:] == ["summary"]:
        summary(); sys.exit()
    from joblib import Parallel, delayed
    jobs = [(l, o, s) for l, o, seeds in JOBS for s in seeds]
    jobs.sort(key=lambda j: -params(j[1])["N"] * params(j[1])["C"])            # largest first
    for r in Parallel(n_jobs=10, verbose=5)(delayed(job)(*j) for j in jobs):
        print(*r, flush=True)
    summary()
