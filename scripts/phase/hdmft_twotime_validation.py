"""Two-time HDMFT vs simulation on the power-law graph, fluctuating phase.

Thesis operating point (mu_c=1.76, sigma=1.75, lam=1e-3), N=8000, C=200, pl2.5.
Both sides are measured with the same protocol: late window [180, 240], share floor
1e-6 for survival, Delta ln S at dt=0.5 for volatility. The HDMFT classes and kernel
are built from the degree sequence of the simulated graph (seed 0).

Run: uv run python scripts/phase/hdmft_twotime_validation.py   -> data/hdmft_twotime_validation.npz
     HTV_SMOKE=1 ... (code-path check only)
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate, growth_rate
from relative_glv.msb import tent_stats
from relative_glv.hdmft import (bin_degree_distribution, soft_configuration_kernel,
                                solve_twotime_heterogeneous)

SMOKE = os.environ.get("HTV_SMOKE", "0") == "1"
MU, SIGMA, LAM = 1.76, 1.75, 1e-3
N, C = (2000, 50) if SMOKE else (8000, 200)
SEEDS = 2 if SMOKE else int(os.environ.get("HTV_SEEDS", 6))
TMAX, LATE, DT = 240.0, (180.0, 240.0), 0.5
FLOOR = 1e-6 * N                                   # share floor 1e-6, in units of S = N w
N_CLS = 10 if SMOKE else 50
EDGES = np.logspace(np.log10(40), np.log10(8000), 13)


def per_firm(lnS_grid, S_window):
    """Survival (min over window above floor), mean size and MAD volatility per firm."""
    live = S_window.min(1) > FLOOR
    g = np.diff(lnS_grid, axis=1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    return live, np.exp(lnS_grid).mean(1), vol, g


def simulate(seed):
    a = coupling(N, MU, SIGMA, kind="powerlaw", seed=seed, mean_degree=C)
    k = np.diff(a.indptr).astype(float)
    r = integrate(a, tmax=TMAX, n_eval=961, lam=LAM, seed=seed,
                  method="RK45", rtol=1e-4, atol=1e-7)
    t, S = r["t"], N * r["W"]
    win = (t >= LATE[0]) & (t <= LATE[1])
    tg = np.arange(LATE[0], LATE[1] + 1e-9, DT)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(s, 1e-300))) for s in S])
    live, Sbar, vol, g = per_firm(lnS, S[:, win])
    return dict(k=k, live=live, Sbar=Sbar, vol=vol, g=g[live].ravel(),
                g_eff=growth_rate(t, r["lnM"], frac=0.25))


def theory(k):
    b = bin_degree_distribution(k, N_CLS)
    ker = soft_configuration_kernel(b["degree_fractions"], b["weights"], b["connectance"])["kernel"]
    r = solve_twotime_heterogeneous(MU, SIGMA, ker, b["connectance"], lam=LAM,
                                    n_per_class=40 if SMOKE else 160,
                                    Nt=int(TMAX / 0.25), dt=0.25,
                                    iters=6 if SMOKE else 30, burn=3 if SMOKE else 15,
                                    verbose=True)
    t, U = r["t"], r["U"]                          # U (n_cls, n_per, Nt)
    kk = np.repeat(b["representative_degrees"], U.shape[1])
    U = U.reshape(-1, U.shape[-1])
    win = (t >= LATE[0]) & (t <= LATE[1])
    step = int(round(DT / (t[1] - t[0])))
    lnS = np.log(np.maximum(U[:, win][:, ::step], 1e-300))
    live, Sbar, vol, g = per_firm(lnS, U[:, win])
    late = t >= LATE[0]
    return dict(k=kk, live=live, Sbar=Sbar, vol=vol, g=g[live].ravel(),
                g_eff=float(r["g_t"][late].mean()), err=r["err"], hist=r["hist"])


def binned(k, live, Sbar, vol):
    """Survival P(live|k); median size and volatility of survivors, in median units."""
    idx = np.digitize(k, EDGES) - 1
    nb = EDGES.size - 1
    surv = np.array([live[idx == b].mean() if (idx == b).sum() else np.nan for b in range(nb)])
    Sn, vn = Sbar / np.median(Sbar[live]), vol / np.median(vol[live])
    med = lambda x: np.array([np.median(x[(idx == b) & live]) if ((idx == b) & live).sum() >= 5
                              else np.nan for b in range(nb)])
    return surv, med(Sn), med(vn)


if __name__ == "__main__":
    t0 = time.time()
    sims = Parallel(n_jobs=min(SEEDS, 8))(delayed(simulate)(s) for s in range(SEEDS))
    print(f"sims done {(time.time() - t0) / 60:.1f} min", flush=True)
    th = theory(sims[0]["k"])
    print(f"theory done {(time.time() - t0) / 60:.1f} min  rel.err={th['err']:.3f}", flush=True)

    sim_b = np.array([binned(s["k"], s["live"], s["Sbar"], s["vol"]) for s in sims])  # (seeds,3,nb)
    th_b = np.array(binned(th["k"], th["live"], th["Sbar"], th["vol"]))
    g_sim = np.concatenate([s["g"] for s in sims])
    ts, tt = tent_stats(g_sim), tent_stats(th["g"])
    geff = np.array([s["g_eff"] for s in sims])
    surv = np.array([s["live"].mean() for s in sims])
    print(f"g_eff     sim {geff.mean():+.3f} ± {geff.std():.3f}   theory {th['g_eff']:+.3f}")
    print(f"survival  sim {surv.mean():.3f} ± {surv.std():.3f}   theory {th['live'].mean():.3f}")
    print(f"exkurt    sim {ts['exkurt']:.2f}   theory {tt['exkurt']:.2f}")
    mids = np.sqrt(EDGES[:-1] * EDGES[1:])
    names = ("P(surv|k)", "S/med", "vol/med")
    for i, nm in enumerate(names):
        print(nm)
        for m, a, b in zip(mids, np.nanmedian(sim_b[:, i], 0), th_b[i]):
            print(f"   k={m:7.0f}  sim {a:6.3f}  theory {b:6.3f}")
    if not SMOKE:
        np.savez("data/hdmft_twotime_validation.npz", mids=mids, sim_b=sim_b, th_b=th_b,
                 geff_sim=geff, geff_th=th["g_eff"], surv_sim=surv, surv_th=th["live"].mean(),
                 g_sim=g_sim, g_th=th["g"], hist=th["hist"])
