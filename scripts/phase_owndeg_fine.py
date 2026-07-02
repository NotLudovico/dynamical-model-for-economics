"""Fine-resolution PHASE diagram of the own-degree relative GLV (regime-coloured,
not beta-coloured; mu<0 = competitive, thesis convention).

Regimes: DIVERGENT (integration fails/explodes) / FROZEN (shares relax) /
SHRINKING (g_eff<0) / FLUCTUATING+GROWING (the MSB regime). Robust to exploding
sims: per-seed wall-clock cap + finite-check + broad except -> bad runs become
NaN (flagged divergent), never crash the job; per-row checkpoint to the npz.

    uv run python scripts/phase_owndeg_fine.py            # compute (~2-3h) + plot
    uv run python scripts/phase_owndeg_fine.py --plot     # replot from the npz
    uv run python scripts/phase_owndeg_fine.py --smoke
    uv run python scripts/phase_owndeg_fine.py --grid 20 --seeds 10   # finer + more seeds

--grid/--seeds write to a tagged npz+png (phase_owndeg_fine_<nS>x<nM>_<seeds>s.*), so the
banked default figure is never clobbered until you copy the new one over it.
"""
import sys, os, time, signal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate, growth_rate
from relative_glv.msb import size_volatility

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NPZ = os.path.join(DATA, "phase_owndeg_fine.npz")
PNG_OUT = os.path.join(DATA, "phase_owndeg_fine.png")
N, LAM = 4000, 1e-3
TMAX, N_EVAL, LATE, DT = 400.0, 800, (320.0, 390.0), 0.5
GRID = 14                                            # cells per axis (override with --grid)
SIGMAS = np.round(np.linspace(1.40, 1.95, GRID), 4)
MUS = np.round(np.linspace(0.0, 2.5, GRID), 4)       # code convention (mu>0 competitive)
SEEDS, N_JOBS, TIMEOUT = 5, 8, 90
OP = (1.76, 1.75)                                    # operating point (mu_code, sigma)
FREEZE_CHURN = 0.035


class _TO(Exception):
    pass


def _alarm(_s, _f):
    raise _TO()


def run_one(sigma, mu, seed):
    """Explosion-safe: wall-clock cap + finite check + broad except -> NaN on any
    failure (flagged divergent later), never raises."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        a = coupling(N, mu, sigma, kind="powerlaw_owndeg", seed=seed)
        r = integrate(a, tmax=TMAX, n_eval=N_EVAL, lam=LAM, seed=seed,
                      method="RK45", rtol=1e-4, atol=1e-7)
        signal.alarm(0)
        if (not r["success"]) or (not np.isfinite(r["W"]).all()) or (not np.isfinite(r["lnM"]).all()):
            return [np.nan] * 5
        sv = size_volatility(r["W"], r["t"], window=LATE, dt=DT)
        return [float(sv["growth"].std()), growth_rate(r["t"], r["lnM"], frac=0.3),
                sv["vol"].size / N, float(sv["beta"]), float(sv["r2"])]
    except Exception:
        signal.alarm(0)
        return [np.nan] * 5


def compute():
    t0 = time.time()
    nS, nM = len(SIGMAS), len(MUS)
    raw = np.full((nS, nM, SEEDS, 5), np.nan)
    print(f"[phase-fine] own-degree N={N} lam={LAM}  {nS}x{nM} grid x {SEEDS} seeds  "
          f"(cap {TIMEOUT}s/seed, explosion-safe, per-row checkpoint)", flush=True)
    for i, s in enumerate(SIGMAS):
        for j, m in enumerate(MUS):
            raw[i, j] = np.array(Parallel(n_jobs=N_JOBS)(
                delayed(run_one)(s, m, k) for k in range(SEEDS)))
        np.savez(NPZ, sigmas=SIGMAS, mus=MUS, seeds=SEEDS, N=N, lam=LAM,
                 op=np.array(OP), raw=raw)                          # checkpoint each row
        nd = int((np.isfinite(raw[i, :, :, 0]).mean(1) < 0.5).sum())
        print(f"  sigma={s:.3f} done  (divergent cols this row: {nd}/{nM})  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"[phase-fine] saved {os.path.basename(NPZ)}  ({(time.time()-t0)/60:.1f} min)", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.interpolate import RegularGridInterpolator
    d = np.load(NPZ)
    sig, mus, raw, op = d["sigmas"], d["mus"], d["raw"], d["op"]
    integ = np.isfinite(raw[..., 0]).mean(2)
    med = np.nanmedian(raw, axis=2)
    churn, geff = med[..., 0], med[..., 1]
    nS, nM = len(sig), len(mus)

    # classify each cell into a regime: 0 frozen, 1 fluctuating+growing, 2 shrinking, 3 divergent
    code = np.empty((nS, nM), int)
    for i in range(nS):
        for j in range(nM):
            if integ[i, j] < 0.5:
                code[i, j] = 3
            elif (not np.isfinite(churn[i, j])) or churn[i, j] < FREEZE_CHURN:
                code[i, j] = 0
            elif np.isfinite(geff[i, j]) and geff[i, j] < 0:
                code[i, j] = 2
            else:
                code[i, j] = 1

    # RAW cells -- no despeckle, no interpolation: every square is one measured cell.
    from matplotlib.colors import ListedColormap
    COLS = ["#e8e8e8", "#4878a8", "#c0744f", "#3a3a3a"]   # frozen, fluctuating, shrinking, divergent
    mu_plot = -mus                                        # mu<0 (competitive) convention
    order = np.argsort(mu_plot)                           # ascending: most competitive on the left
    mu_s = mu_plot[order]; code_s = code[:, order]

    def edges(a):
        a = np.asarray(a, float); m = (a[:-1] + a[1:]) / 2
        return np.concatenate([[2 * a[0] - m[0]], m, [2 * a[-1] - m[-1]]])

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.pcolormesh(edges(mu_s), edges(sig), code_s, cmap=ListedColormap(COLS),
                  vmin=-0.5, vmax=3.5, shading="flat", edgecolors="white", linewidth=0.3)
    COL = np.array([matplotlib.colors.to_rgb(c) for c in COLS])

    # grow/shrink boundary: g_eff=0 contour of the MEASURED grid (divergent cells = fast-growing)
    geff_s = geff[:, order].copy()
    fin = np.isfinite(geff_s)
    geff_s[~fin] = np.nanmax(geff_s[fin]) if fin.any() else 0.0
    ax.contour(mu_s, sig, geff_s, levels=[0.0], colors="black", linewidths=2.2)
    ax.scatter([-op[0]], [op[1]], marker="*", s=520, color="black", zorder=6)
    ax.set_xlabel(r"mean interaction  $\mu$   ($\mu<0$ competitive)", fontsize=12)
    ax.set_ylabel(r"interaction disorder  $\sigma$", fontsize=12)
    ax.set_title(f"Phase diagram of the own-degree relative GLV  (N={int(d['N'])})", fontsize=12)
    # label each region directly on the plot (no legend box)
    ax.text(-2.05, 1.45, "frozen", color="black", fontsize=12, ha="center", va="center")
    ax.text(-0.95, 1.71, "fluctuating\n+ growing\n(MSB)", color="white", fontsize=11,
            ha="center", va="center", weight="bold")
    ax.text(-2.28, 1.73, "shrinking", color="white", fontsize=10.5, ha="center", va="center",
            weight="bold")
    ax.text(-0.45, 1.93, "divergent", color="white", fontsize=11, ha="center", va="center",
            weight="bold")
    ax.text(-0.95, 1.46, r"$g_{\rm eff}=0$", color="black", fontsize=10, ha="left", va="center")
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=150, bbox_inches="tight")
    print(f"[phase-fine] figure -> {PNG_OUT}")


def _argval(flag, cast, default):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        SIGMAS = np.array([1.45, 1.7, 1.9]); MUS = np.array([0.0, 1.0, 2.0]); SEEDS = 2
    elif "--grid" in sys.argv or "--seeds" in sys.argv:
        GRID = _argval("--grid", int, GRID)
        SEEDS = _argval("--seeds", int, SEEDS)
        SIGMAS = np.round(np.linspace(1.40, 1.95, GRID), 4)
        MUS = np.round(np.linspace(0.0, 2.5, GRID), 4)
        # tagged outputs so the banked default figure is never clobbered
        tag = f"_{len(SIGMAS)}x{len(MUS)}_{SEEDS}s"
        NPZ = os.path.join(DATA, f"phase_owndeg_fine{tag}.npz")
        PNG_OUT = os.path.join(DATA, f"phase_owndeg_fine{tag}.png")
    if "--plot" not in sys.argv:
        compute()
    plot()
