"""Code shared by the notebooks: the economy and how it is measured.

graphs        powerlaw_coupling (networkx, the thesis-era phase grid), configuration_coupling (numpy, large N)
dynamics      integrate_demographic: the model with birth-death noise and turnover, in relative sizes
measurement   lives, zeta_bins, hill: the Moran-Secchi-Bouchaud (MSB) life panel and its exponents
reference     the paper's reference point, its two stored economies and the per-life volatility law
plotting      tidy

Sign convention: mu > 0 is mean competition (the notebooks' text uses mu < 0).
"""
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "paper" / "figures"

MSB_ZETA = np.array([0.20, 0.39, 0.51, 0.58])           # Compustat, Moran-Secchi-Bouchaud 2024
MSB_RATIO = np.array([1.0, 1.95, 2.55, 2.90])

# reference point: alpha 2.5, sigma 1.70, D 0.1, N 16000, C 200; burn-in 100, window 480, one year = 0.5
ALPHA, SIGMA, D, N, C, MU = 2.5, 1.70, 0.1, 16000, 200, 1.76
START, TOBS, DT = 100.0, 480.0, 0.5
SEEDS = (0, 1)
TRAJ = DATA / "demographic" / "multiscaling"


# ---------------------------------------------------------------------------------------------- graphs
def _couple(ei, ej, n, mu, sigma, C_eff, rng):
    """Edge list -> couplings mu/C + sigma z/sqrt(C), z_ij and z_ji independent."""
    a, b = rng.normal(0, 1, ei.size), rng.normal(0, 1, ei.size)
    scale = sigma / np.sqrt(2 * C_eff)
    sym, anti = (a + b) / np.sqrt(2), (a - b) / np.sqrt(2)
    w_ij, w_ji = mu / C_eff + scale * (sym + anti), mu / C_eff + scale * (sym - anti)
    return sparse.csr_array((np.concatenate([w_ij, w_ji]), (np.concatenate([ei, ej]), np.concatenate([ej, ei]))),
                            shape=(n, n))


def powerlaw_coupling(n, mu, sigma, *, alpha=2.5, mean_degree=100, seed=0):
    """Couplings on a configuration-model graph with p(k) ~ k^-alpha (networkx, collapsed to a simple graph)."""
    import networkx as nx
    rng = np.random.default_rng(seed)
    C = min(mean_degree, n - 1)
    tail = (1 - rng.uniform(size=n)) ** (-1 / (alpha - 1))
    deg = np.maximum((C * (alpha - 2) / (alpha - 1) * tail).round().astype(int), 1)
    if deg.sum() % 2:
        deg[deg.argmin()] += 1
    g = nx.Graph(nx.configuration_model(deg.tolist(), seed=int(seed)))
    g.remove_edges_from(nx.selfloop_edges(g))
    A = nx.to_scipy_sparse_array(g, format="csr", dtype=float)
    A.data[:] = 1.0
    Au = sparse.triu(A, k=1).tocoo()
    return _couple(Au.row, Au.col, n, mu, sigma, float(np.asarray(A.sum(axis=1)).mean()), rng)


def configuration_coupling(n, mu, sigma, *, alpha=2.5, mean_degree=200, seed=0, regular=False):
    """Same ensemble as powerlaw_coupling (or every degree = mean_degree if regular), in numpy: stubs paired at
    random, self-loops dropped, multi-edges merged.  Scales to n ~ 1e5; different random stream."""
    rng = np.random.default_rng(seed)
    if regular:
        deg = np.full(n, mean_degree, dtype=np.int64)
    else:
        tail = (1 - rng.uniform(size=n)) ** (-1 / (alpha - 1))
        deg = np.maximum((mean_degree * (alpha - 2) / (alpha - 1) * tail).round().astype(np.int64), 1)
    if deg.sum() % 2:
        deg[deg.argmin()] += 1
    stubs = np.repeat(np.arange(n, dtype=np.int64), deg)
    rng.shuffle(stubs)
    i, j = stubs[0::2], stubs[1::2]
    keep = i != j
    i, j = np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])
    key = np.unique(i * n + j)
    return _couple(key // n, key % n, n, mu, sigma, 2.0 * key.size / n, rng)


# ---------------------------------------------------------------------------------------------- dynamics
def integrate_demographic(a, *, D, tmax, t_start=0.0, dt_save=0.5, dt=0.01, seed=0, compact=False):
    """dS_i = S_i (f_i - fbar) dt + sqrt(D S_i) dW_i - (S_i/N) sum_j sqrt(D S_j) dW_j,  f_i = 1 - S_i - (a S)_i,
    fbar = sum_i S_i f_i / N: the model x_i' = x_i (1 - x_i/m - (a x)_i/m) + sqrt(D m x_i) noise in relative sizes.
    A firm that reaches S <= 0 exits and a newborn of size D takes its node.  Euler-Maruyama, sum S = N restored
    after every step.  Returns t, S (N, n_save) and life (N, n_save), the death counter of each node: a life is a
    maximal run of samples with the same counter.  compact=True stores float32/int32."""
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    S = rng.uniform(0.5, 1.5, n)
    S *= n / S.sum()
    life = np.zeros(n, dtype=np.int64)
    every, nsteps = int(round(dt_save / dt)), int(round(tmax / dt))
    t_out, S_out, L_out = [], [], []
    for step in range(nsteps + 1):
        if step % every == 0 and step * dt >= t_start - 1e-9:
            t_out.append(step * dt)
            S_out.append(S.astype(np.float32) if compact else S.copy())
            L_out.append(life.astype(np.int32) if compact else life.copy())
        f = 1.0 - S - a @ S
        xi = np.sqrt(D * S * dt) * rng.standard_normal(n)
        S = S + S * (f - S @ f / n) * dt + xi - S * xi.sum() / n
        dead = S <= 0
        if dead.any():
            S[dead] = D
            life[dead] += 1
        S *= n / S.sum()
    return dict(t=np.array(t_out), S=np.array(S_out).T, life=np.array(L_out).T)


# ---------------------------------------------------------------------------------------------- measurement
def lives(S, L, step=1):
    """Lives on the `step`-subsampled grid (MSB unbalanced panel), window-censored ones included.  One tuple per
    life with >= 2 growth rates and positive volatility: (Sbar, v, n, g - mean g, g, node, first sample),
    v = sqrt(pi/2) times the mean absolute deviation of the growth rates g = Delta ln S."""
    S, L = S[:, ::step], L[:, ::step]
    lnS, T = np.log(S), S.shape[1]
    out = []
    for i in range(S.shape[0]):
        for seg in np.split(np.arange(T), np.flatnonzero(np.diff(L[i])) + 1):
            if seg.size < 3:
                continue
            g = np.diff(lnS[i, seg])
            v = np.sqrt(np.pi / 2) * np.abs(g - g.mean()).mean()
            if v > 0:
                out.append((S[i, seg].mean(), v, g.size, g - g.mean(), g, i, seg[0]))
    return out


def zeta_bins(Sbar, vol, n_bins=25, qs=(1, 2, 3, 4)):
    """E[vol^q | Sbar] ~ Sbar^-zeta_q over all equal-count bins.  Returns (zeta, largest-bin residual of the q = 1
    fit in log units, mean vol of the smallest and largest bin, decades spanned, local exponent per fifth)."""
    Sbar, vol = np.asarray(Sbar, float), np.asarray(vol, float)
    P = np.array_split(np.argsort(Sbar), n_bins)
    lx = np.log([Sbar[p].mean() for p in P])
    z, res = [], None
    for q in qs:
        ly = np.log([np.mean(vol[p] ** q) for p in P])
        c = np.polyfit(lx, ly, 1)
        z.append(-c[0])
        if q == qs[0]:
            res = ly - np.polyval(c, lx)
    mv = np.array([vol[p].mean() for p in P])
    w = max(n_bins // 5, 2)
    loc = [-np.polyfit(lx[i:i + w], np.log(mv[i:i + w]), 1)[0] for i in range(0, n_bins - w + 1, w)]
    return np.array(z), float(res[-1]), float(mv[0]), float(mv[-1]), float((lx[-1] - lx[0]) / np.log(10)), loc


def hill(z, p):
    """Hill estimate of the tail index from the top fraction p of the sample (at least 10 points)."""
    z = np.sort(np.asarray(z, float))[::-1]
    k = max(int(p * len(z)), 10)
    return float(1.0 / np.mean(np.log(z[:k] / z[k])))


# ---------------------------------------------------------------------------------------------- reference economies
def reference_coupling(seed, kind="powerlaw"):
    return configuration_coupling(N, MU, SIGMA, alpha=ALPHA, mean_degree=C, seed=seed, regular=(kind == "regular"))


def simulate_reference(seed, kind="powerlaw"):
    """Path of a stored reference economy (kind "powerlaw" or "regular"), simulated if missing (about an hour)."""
    f = TRAJ / f"{kind}_seed{seed}.npz"
    if not f.exists():
        TRAJ.mkdir(parents=True, exist_ok=True)
        a = reference_coupling(seed, kind)
        r = integrate_demographic(a, D=D, tmax=START + TOBS, t_start=START, dt_save=DT, seed=seed, compact=True)
        np.savez_compressed(f, S=r["S"], L=r["life"], deg=np.diff(a.indptr))
    return f


def reference_economy(seed, kind="powerlaw"):
    """Sizes S (N, T), death counters L, degrees, couplings a and the field Phi = 1 - aS - fbar."""
    d = np.load(simulate_reference(seed, kind))
    S, L, deg = d["S"].astype(float), d["L"], d["deg"]
    a = reference_coupling(seed, kind)
    assert np.array_equal(np.diff(a.indptr), deg), "coupling does not match the stored economy"
    aS = a @ S
    fbar = (S * (1.0 - S - aS)).sum(0) / N
    return S, L, deg, a, 1.0 - aS - fbar


def noise_free(ph, lnS0, dt=0.01):
    """ln S of dS = S(Phi - S)dt along padded fields ph (lives, n), Phi linear between samples."""
    per = int(round(DT / dt))
    lnS = lnS0.copy()
    det = [lnS.copy()]
    for k in range(ph.shape[1] - 1):
        p0, p1 = ph[:, k], ph[:, k + 1]
        for j in range(per):
            lnS = lnS + (p0 + (p1 - p0) * j / per - np.exp(lnS)) * dt
        det.append(lnS.copy())
    return np.array(det).T


def life_rows(seed, kind="powerlaw", nmin=20, chunk=20000):
    """(Sbar, var g, V_F, B, degree) for every life with >= nmin growth rates: V_F the growth variance of the
    noise-free one-firm equation along the life's measured field from its starting size, B = mean_k
    (D/S_k^2)(1 - exp(-S_k dt)) the filtered demographic part."""
    S, L, deg, _, Phi = reference_economy(seed, kind)
    T = S.shape[1]
    lv = [(i, seg) for i in range(S.shape[0]) for seg in np.split(np.arange(T), np.flatnonzero(np.diff(L[i])) + 1)
          if seg.size >= nmin + 1]
    lv.sort(key=lambda x: x[1].size)                     # chunks of similar length keep the padding small
    rows = []
    for c0 in range(0, len(lv), chunk):
        part = lv[c0:c0 + chunk]
        ph, x0 = np.zeros((len(part), part[-1][1].size)), np.empty(len(part))
        for k, (i, seg) in enumerate(part):
            ph[k, :seg.size] = Phi[i, seg]
            ph[k, seg.size:] = Phi[i, seg[-1]]
            x0[k] = S[i, seg[0]]
        det = noise_free(ph, np.log(x0))
        for k, (i, seg) in enumerate(part):
            x = S[i, seg]
            g, gd = np.diff(np.log(x)), np.diff(det[k, :seg.size])
            rows.append((x.mean(), g.var(), gd.var(), np.mean(D / x[:-1] ** 2 * (1 - np.exp(-x[:-1] * DT))), deg[i]))
    return np.array(rows)


def decomposition(kind="powerlaw"):
    """life_rows of both reference economies, cached in data/demographic/volatility_law[_regular].npz."""
    f = DATA / "demographic" / ("volatility_law.npz" if kind == "powerlaw" else f"volatility_law_{kind}.npz")
    if not f.exists():
        np.savez(f, rows=np.concatenate([life_rows(s, kind) for s in SEEDS]))
    return np.load(f)["rows"]


# ---------------------------------------------------------------------------------------------- plotting
def tidy(ax, xticks=None, yticks=None):
    """Plain-number ticks on log axes, light major grid, no top/right spines."""
    from matplotlib.ticker import NullFormatter, FormatStrFormatter
    for axis, ticks, scale in ((ax.xaxis, xticks, ax.get_xscale()), (ax.yaxis, yticks, ax.get_yscale())):
        if scale == "log":
            axis.set_minor_formatter(NullFormatter())
            axis.set_major_formatter(FormatStrFormatter("%g"))
            if ticks is not None:
                axis.set_ticks(ticks)
    ax.grid(True, which="major", color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
