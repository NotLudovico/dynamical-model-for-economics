"""Fine-resolution PHASE diagram of the own-degree relative GLV (regime-coloured,
not beta-coloured; mu<0 = competitive, thesis convention).

Regimes: DIVERGENT (integration fails/explodes) / FROZEN (shares relax) /
SHRINKING (g_eff<0) / FLUCTUATING+GROWING (the MSB regime). Robust to exploding
sims: per-seed wall-clock cap + finite-check + broad except -> bad runs become
NaN (flagged divergent), never crash the job; per-row checkpoint to the npz.

    uv run python scripts/phase_owndeg_fine.py            # compute (~2-3h) + plot
    uv run python scripts/phase_owndeg_fine.py --plot     # replot from the npz
    uv run python scripts/phase_owndeg_fine.py --smoke
"""
import sys, os, time, signal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed
from relative_glv.model import coupling, integrate, growth_rate
from relative_glv.msb import size_volatility

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NPZ = os.path.join(DATA, "phase_owndeg_fine.npz")
N, LAM = 4000, 1e-3
TMAX, N_EVAL, LATE, DT = 400.0, 800, (320.0, 390.0), 0.5
SIGMAS = np.round(np.linspace(1.40, 1.95, 14), 4)
MUS = np.round(np.linspace(0.0, 2.5, 14), 4)        # code convention (mu>0 competitive)
SEEDS, N_JOBS, TIMEOUT = 5, 6, 90
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

    # despeckle: reassign an isolated cell to its neighbour-majority (>=6 of 8 agree), 2 passes.
    # removes single near-frozen/divergent cells that flicker at a phase boundary (seed noise).
    clean = code.copy()
    for _ in range(2):
        s0 = clean.copy()
        for i in range(1, nS - 1):
            for j in range(1, nM - 1):
                nb = np.delete(s0[i-1:i+2, j-1:j+2].ravel(), 4)
                v, c = np.unique(nb, return_counts=True)
                if c.max() >= 6 and v[c.argmax()] != s0[i, j]:
                    clean[i, j] = v[c.argmax()]

    # smooth regions: interpolate each regime's indicator, take argmax on a fine mesh
    MU = np.linspace(mus[0], mus[-1], 420)
    SG = np.linspace(sig[0], sig[-1], 420)
    MM, SS = np.meshgrid(MU, SG)
    pts = np.column_stack([SS.ravel(), MM.ravel()])
    ind = np.stack([RegularGridInterpolator((sig, mus), (clean == r).astype(float),
                    bounds_error=False, fill_value=None)(pts).reshape(SS.shape) for r in range(4)])
    reg = ind.argmax(0)
    COL = np.array([matplotlib.colors.to_rgb(c)            # frozen, fluctuating, shrinking, divergent
                    for c in ["#aebfd4", "#3f9b6e", "#e2948c", "#2f2f2f"]])
    img = COL[reg][:, ::-1]                                # flip mu axis: mu<0 (competitive) on the left

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.imshow(img, origin="lower", aspect="auto", extent=[-mus[-1], -mus[0], sig[0], sig[-1]])
    fge = RegularGridInterpolator((sig, mus), np.nan_to_num(geff), bounds_error=False, fill_value=None)
    ax.contour(-MM, SS, fge(pts).reshape(SS.shape), levels=[0.0], colors="#6e1414", linewidths=1.4)
    ax.axhline(np.sqrt(2), color="#1f6f8b", ls="--", lw=1.1)
    ax.text(-mus[-1] + 0.05, np.sqrt(2) + 0.012, r"$\sigma_c=\sqrt{2}$ (DMFT)",
            color="#1f6f8b", fontsize=8.5, va="bottom")
    ax.scatter([-op[0]], [op[1]], marker="*", s=520, edgecolor="gold",
               facecolor="none", linewidth=2.6, zorder=6)
    ax.annotate("operating point", (-op[0], op[1]), (-op[0], op[1] - 0.085),
                fontsize=9, color="gold", ha="center", weight="bold")
    ax.set_xlabel(r"mean interaction  $\mu$   ($\mu<0$ competitive)", fontsize=12)
    ax.set_ylabel(r"interaction disorder  $\sigma$", fontsize=12)
    ax.set_title(f"Phase diagram of the own-degree relative GLV  (N={int(d['N'])})", fontsize=12)
    leg = [Patch(facecolor=COL[1], label="fluctuating + growing (MSB regime)"),
           Patch(facecolor=COL[0], label="frozen (shares relax)"),
           Patch(facecolor=COL[2], label=r"shrinking ($g_{\rm eff}<0$)"),
           Patch(facecolor=COL[3], label="divergent (explodes)")]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    fig.tight_layout()
    out = os.path.join(DATA, "phase_owndeg_fine.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[phase-fine] figure -> {out}")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        SIGMAS = np.array([1.45, 1.7, 1.9]); MUS = np.array([0.0, 1.0, 2.0]); SEEDS = 2
    if "--plot" not in sys.argv:
        compute()
    plot()
