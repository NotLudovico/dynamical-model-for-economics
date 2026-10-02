"""Parameter scan under the MSB unbalanced panel (msb.spell_volatility): is there a region where
beta ~ 0.15-0.20 and zeta_q/zeta_1 ~ 2.0, 2.6, 2.9 hold together?

Grid: alpha (degree tail) x sigma (disorder) x lam (immigration floor), mu = 1.76 (code convention),
N = 8000, C = N/40, 8 economies per point. The listing threshold follows the floor: share 8 lam / N
(= 1e-6 at lam = 1e-3, the thesis value). Per economy:
  spells >= 2 and >= 20 growth rates at T_obs = 240 and 480 from t = 180 (convergence check)
  the thesis survivor estimator on [180, 240] (protocol comparison)
  tent stats of the pooled listed growth rates (fat tails under the same panel)
One file per grid point in data/scan_spells/, so an interrupted scan resumes where it stopped.

Run:      uv run python scripts/scan_spells.py
Summary:  uv run python scripts/scan_spells.py summary
Smoke:    SCAN_SMOKE=1 uv run python scripts/scan_spells.py
"""
import os, sys, time, signal, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
from joblib import Parallel, delayed
from relative_glv import model
from relative_glv.msb import size_volatility, spell_volatility, tent_stats
from beta_spells import cq_one

SMOKE = os.environ.get("SCAN_SMOKE") == "1"
MU, START, DT = 1.76, 180.0, 0.5
N, SEEDS = (2000, 2) if SMOKE else (8000, 8)
C = N // 40
TOBS = (30.0, 60.0) if SMOKE else (240.0, 480.0)
ALPHAS, SIGMAS, LAMS = ((2.5,), (1.75,), (1e-3,)) if SMOKE else ((2.5, 3.0, 3.5, 6.0), (1.6, 1.75, 1.9),
                                                                (1e-4, 1e-3, 1e-2))
TIMEOUT = 600 if SMOKE else 3600                    # s per economy; a stuck integration is dropped
OUT = os.path.join(ROOT, "data", "scan_smoke" if SMOKE else "scan_spells")
DATA_BETA, DATA_RATIO = (0.15, 0.20), np.array([2.0, 2.6, 2.9])


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def zeta(S, v):
    z = cq_one(S, v)
    return np.full(4, np.nan) if z is None else z


def economy(alpha, sigma, lam, seed):
    """Row: [beta, z1..z4, n, cens] per (T_obs, min_growth), then survivor [beta, z1..z4, n],
    then [exkurt, bowley, churn]. None if the integration fails or times out."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        model._ALPHA_PL = alpha
        a = model.coupling(N, MU, sigma, kind="powerlaw", seed=seed, mean_degree=C)
        tmax = START + max(TOBS)
        r = model.integrate(a, tmax=tmax, n_eval=int(tmax / DT) + 1, lam=lam, seed=seed,
                            method="RK45", rtol=1e-4, atol=1e-7)
        signal.alarm(0)
    except _Timeout:
        return None
    if not r["success"] or not np.isfinite(r["W"]).all():
        return None
    t, W, floor = r["t"], r["W"], 8 * lam / N
    row = []
    for T in TOBS:
        for m in (2, 20):
            sp = spell_volatility(W, t, window=(START, START + T), dt=DT, floor=floor, min_growth=m)
            row += [sp["beta"], *zeta(sp["Sbar"], sp["vol"]), sp["Sbar"].size, sp["censored"].mean()]
    sv = size_volatility(W, t, window=(START, START + 60.0), dt=DT)
    row += [sv["beta"], *zeta(sv["Sbar"], sv["vol"]), sv["Sbar"].size]
    S = N * W[:, t >= START - 1e-9]
    ok = (S[:, :-1] > N * floor) & (S[:, 1:] > N * floor)
    g = np.diff(np.log(np.maximum(S, 1e-300)), axis=1)[ok]
    ts = tent_stats(g)
    row += [ts["exkurt"], ts["bowley"], g.std()]
    return row


def cols():
    c = [f"{k}_T{int(T)}_m{m}" for T in TOBS for m in (2, 20)
         for k in ("beta", "z1", "z2", "z3", "z4", "n", "cens")]
    return c + ["beta_surv", "z1_surv", "z2_surv", "z3_surv", "z4_surv", "n_surv", "exkurt", "bowley", "churn"]


def path(alpha, sigma, lam):
    return os.path.join(OUT, f"a{alpha}_s{sigma}_l{lam:.0e}.npz")


def ratios(R, c, tag):
    z = R[:, [c[f"z{q}{tag}"] for q in (1, 2, 3, 4)]]
    z = z[np.isfinite(z).all(1)]
    return z.mean(0)[1:] / z.mean(0)[0] if len(z) else np.full(3, np.nan)


def summary():
    c = {k: i for i, k in enumerate(cols())}
    T = int(max(TOBS))
    print(f"spells >= 2 at T_obs={T} (dT: change since T={int(min(TOBS))}); survivor = thesis estimator; "
          f"* = beta in {DATA_BETA} and ratios within 0.3 of {DATA_RATIO}")
    print(f"{'alpha':>5} {'sigma':>5} {'lam':>6} {'n':>2} | {'beta':>5} {'dT':>6} {'zeta_q/zeta_1':>16} "
          f"| {'beta20':>6} {'ratios20':>16} | {'b_surv':>6} {'ratios_surv':>16} | {'exk':>4} {'churn':>5}")
    for alpha, sigma, lam in itertools.product(ALPHAS, SIGMAS, LAMS):
        if not os.path.exists(path(alpha, sigma, lam)):
            continue
        R = np.load(path(alpha, sigma, lam))["rows"]
        if R.size == 0:
            print(f"{alpha:5.1f} {sigma:5.2f} {lam:6.0e}  0 | all economies failed")
            continue
        b, b0 = np.nanmedian(R[:, c[f"beta_T{T}_m2"]]), np.nanmedian(R[:, c[f"beta_T{int(min(TOBS))}_m2"]])
        zr, zr20, zrs = ratios(R, c, f"_T{T}_m2"), ratios(R, c, f"_T{T}_m20"), ratios(R, c, "_surv")
        hit = DATA_BETA[0] <= b <= DATA_BETA[1] and np.all(np.abs(zr - DATA_RATIO) < 0.3)
        f = lambda z: " ".join(f"{x:5.2f}" for x in z)
        print(f"{alpha:5.1f} {sigma:5.2f} {lam:6.0e} {len(R):2d} | {b:5.3f} {b - b0:+6.3f} {f(zr)} "
              f"| {np.nanmedian(R[:, c[f'beta_T{T}_m20']]):6.3f} {f(zr20)} "
              f"| {np.nanmedian(R[:, c['beta_surv']]):6.3f} {f(zrs)} "
              f"| {np.nanmedian(R[:, c['exkurt']]):4.1f} {np.nanmedian(R[:, c['churn']]):5.3f}{'  *' if hit else ''}")


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        summary()
        sys.exit()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    grid = list(itertools.product(ALPHAS, SIGMAS, LAMS))
    for i, (alpha, sigma, lam) in enumerate(grid):
        if os.path.exists(path(alpha, sigma, lam)):
            continue
        rows = Parallel(n_jobs=min(SEEDS, 8))(delayed(economy)(alpha, sigma, lam, s) for s in range(SEEDS))
        rows = np.array([x for x in rows if x is not None], float).reshape(-1, len(cols()))
        np.savez(path(alpha, sigma, lam), rows=rows, cols=np.array(cols()))
        print(f"[{i + 1}/{len(grid)}] alpha={alpha} sigma={sigma} lam={lam:.0e}: {len(rows)}/{SEEDS} economies "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    summary()
