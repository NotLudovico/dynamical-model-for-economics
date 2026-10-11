"""Moran-Santos-Bouchaud (MSB) firm-growth statistics of the relative GLV.

The MSB stylized facts this model reproduces:
  (1) growth rates g_i = Delta ln S_i (S_i = N w_i the relative firm size) follow
      a symmetric, fat-tailed, tent-shaped distribution -- closer to a Laplace
      than a Gaussian;
  (2) the volatility of growth falls with firm size as a power law,
      sigma(S) ~ S^-beta, with the empirical exponent beta ~ 0.15-0.20.

Because the aggregate M divides out of the shares, g_i = Delta ln S_i is purely
idiosyncratic (the common growth g_eff cancels), which is exactly the quantity
the firm-growth literature measures.
"""
import numpy as np
from scipy.stats import kurtosis


def rescale(g):
    """Centre and rescale growth rates by sqrt(pi/2)*MAD, NOT the std.

    The std is inflated by the fat tails and would mis-normalise the tent;
    sqrt(pi/2)*MAD equals the std for a Gaussian but is robust to heavy tails, so
    a Gaussian maps to unit variance while the model's excess kurtosis stays
    visible against the N(0,1) / Laplace reference curves.
    """
    g = np.asarray(g, float).ravel()
    g = g[np.isfinite(g)]
    scale = np.sqrt(np.pi / 2) * np.abs(g - g.mean()).mean()
    return (g - g.mean()) / (scale if scale > 0 else 1.0)


def _decline_beta(Sbar, vol, n_bins=20):
    """sigma(S) ~ S^-beta on the large-S DECLINE branch.

    Bin firms by mean size, take the median volatility per bin, find the peak,
    and least-squares fit the log-log slope from the peak rightward (the small-S
    side is a floor/plateau artefact and is excluded). Returns
    (beta, r2, bin_S, bin_vol, peak_index).
    """
    live = (vol > 0) & np.isfinite(vol) & (Sbar > 0)
    Sbar, vol = Sbar[live], vol[live]
    if Sbar.size < 50:
        return np.nan, np.nan, np.array([]), np.array([]), 0
    o = np.argsort(Sbar)
    P = np.array_split(np.arange(o.size), n_bins)
    bx = np.array([Sbar[o][p].mean() for p in P])
    by = np.array([np.median(vol[o][p]) for p in P])
    m = (bx > 0) & (by > 0)
    bx, by = bx[m], by[m]
    if bx.size <= 5:
        return np.nan, np.nan, bx, by, 0
    pk = int(np.argmax(by))
    if by.size - pk < 5:
        return np.nan, np.nan, bx, by, pk
    c = np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1)
    yh = np.polyval(c, np.log10(bx[pk:]))
    lo = np.log10(by[pk:])
    ss = np.sum((lo - lo.mean()) ** 2)
    r2 = float(1 - np.sum((lo - yh) ** 2) / ss) if ss > 0 else np.nan
    return float(-c[0]), r2, bx, by, pk


def upper_zeta(Sbar, vol, qs=(1, 2, 3, 4), lo_quantile=0.5, n_bins=10):
    """zeta_q from E[vol^q | Sbar] ~ Sbar^-zeta_q on the larger firms only: the firms above the
    lo_quantile of Sbar, in n_bins equal-count bins, bin means of vol^q, least squares in log-log.
    zeta_1 is beta.  A fixed quantile range replaces the peak search of _decline_beta, whose start
    bin jumps on the flat top of spell panels (beta moves by up to 0.09 between economies)."""
    ok = (vol > 0) & np.isfinite(vol) & (Sbar > 0)
    Sbar, vol = Sbar[ok], vol[ok]
    top = Sbar >= np.quantile(Sbar, lo_quantile)
    Sbar, vol = Sbar[top], vol[top]
    o = np.argsort(Sbar)
    P = np.array_split(o, n_bins)
    lx = np.log([Sbar[p].mean() for p in P])
    return np.array([-np.polyfit(lx, np.log([np.mean(vol[p] ** q) for p in P]), 1)[0] for q in qs])


def size_volatility(W, t, *, window, dt, n_bins=20):
    """Size-volatility relation sigma(S) ~ S^-beta from a shares trajectory.

    Relative sizes S_i = N w_i are regridded onto a fixed Delta-t = dt log grid
    over `window` (a fixed grid is required so volatility is not aliased by the
    integrator's adaptive steps). Per firm: mean size Sbar_i and MAD-volatility
    sigma_i = sqrt(pi/2) * mean|g - <g>|, g = Delta ln S_i. beta is the
    decline-branch slope (see _decline_beta). Only firms above the survival floor
    across the whole window contribute. Returns dict(Sbar, vol, beta, r2, bin_S,
    bin_vol, pk, growth).
    """
    N = W.shape[0]
    win = (t >= window[0]) & (t <= window[1])
    live = W[:, win].min(1) > 1e-6
    tg = np.arange(window[0], window[1] + 1e-9, dt)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12)))
                    for i in np.where(live)[0]])
    g = np.diff(lnS, axis=1)
    Sbar = np.exp(lnS).mean(1)
    vol = np.sqrt(np.pi / 2) * np.abs(g - g.mean(1, keepdims=True)).mean(1)
    beta, r2, bx, by, pk = _decline_beta(Sbar, vol, n_bins)
    return dict(Sbar=Sbar, vol=vol, beta=beta, r2=r2,
                bin_S=bx, bin_vol=by, pk=pk, growth=g)


def spell_volatility(W, t, *, window, dt, floor=1e-6, min_growth=2, n_bins=20):
    """size_volatility on the unbalanced panel of Moran-Secchi-Bouchaud (2024, Sec. 3.1).

    A firm is listed while its share is above `floor`. Every listed spell (a contiguous
    stretch above the floor, on the dt grid) counts as one firm with its own lifetime
    average size Sbar and MAD volatility around its own mean growth, if it has at least
    `min_growth` growth rates. Unlike size_volatility, no firm has to persist across the
    whole window, so the estimator has a long-window limit: only the spells cut by the
    window edges (`censored`) depend on it. Returns dict(Sbar, vol, firm, length,
    censored, beta, r2, bin_S, bin_vol, pk); `firm` is the row of W, `length` counts
    grid points.
    """
    N = W.shape[0]
    tg = np.arange(window[0], window[1] + 1e-9, dt)
    lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * w, 1e-300))) for w in W])
    on = np.pad(lnS > np.log(N * floor), ((0, 0), (1, 1)))
    i, j = np.nonzero(np.diff(on.astype(np.int8), axis=1))    # alternating start / end per firm
    a, b = j[::2], j[1::2]                                     # spell = grid points a .. b-1
    keep = b - a > min_growth
    Sbar, vol = [], []
    for f, s, e in zip(i[::2][keep], a[keep], b[keep]):
        x = lnS[f, s:e]
        g = np.diff(x)
        Sbar.append(np.exp(x).mean())
        vol.append(np.sqrt(np.pi / 2) * np.abs(g - g.mean()).mean())
    Sbar, vol = np.array(Sbar), np.array(vol)
    beta, r2, bx, by, pk = _decline_beta(Sbar, vol, n_bins)
    return dict(Sbar=Sbar, vol=vol, firm=i[::2][keep], length=(b - a)[keep],
                censored=((a == 0) | (b == tg.size))[keep],
                beta=beta, r2=r2, bin_S=bx, bin_vol=by, pk=pk)


def tent_stats(g):
    """Shape of the (rescaled) growth-rate distribution: Bowley skewness
    (quartile-based, robust) and excess kurtosis. A symmetric fat tent has
    bowley ~ 0 and excess kurtosis well above 0 (Laplace = 3, Gaussian = 0).
    """
    z = rescale(g)
    q1, q2, q3 = np.percentile(z, [25, 50, 75])
    bowley = (q3 + q1 - 2 * q2) / (q3 - q1)
    return dict(bowley=float(bowley), exkurt=float(kurtosis(z)))


# ---------------------------------------------------------------- life panel (demographic model)
def lives(S, L, step=1):
    """Lives of an economy on the `step`-subsampled grid (MSB unbalanced panel).

    S[i, t] are relative sizes and L[i, t] the death counter of node i, so a life is a maximal
    run of samples with the same counter (window-censored lives included).  Returns one tuple per
    life with at least 2 growth rates and positive volatility:
    (Sbar, v, n, g - mean(g), g, node, first sample index), with v = sqrt(pi/2) MAD.
    """
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


def hill(z, p):
    """Hill estimate of the tail index from the top fraction p of the sample (at least 10 points)."""
    z = np.sort(np.asarray(z, float))[::-1]
    k = max(int(p * len(z)), 10)
    return float(1.0 / np.mean(np.log(z[:k] / z[k])))


def excess_kurtosis(x):
    x = np.asarray(x, float)
    return float(np.mean(x ** 4) / np.mean(x ** 2) ** 2 - 3)


def tail_exponent(z):
    """Stretched-exponential exponent of |z|: slope of log(-log P(|z|>x)) against log x over
    P in [1e-4, 1e-1] (1 = exponential tail, 2 = Gaussian)."""
    a = np.sort(np.abs(np.asarray(z, float)))[::-1]
    p = np.arange(1, a.size + 1) / a.size
    m = (p > 1e-4) & (p < 1e-1) & (a > 0)
    return float(np.polyfit(np.log(a[m]), np.log(-np.log(p[m])), 1)[0])


def zeta_bins(Sbar, vol, n_bins=25, qs=(1, 2, 3, 4)):
    """MSB exponents over all equal-count bins of Sbar: E[vol^q | Sbar] ~ Sbar^-zeta_q.

    Returns (zeta, largest-bin residual of the q = 1 fit in log units, mean vol of the smallest and
    largest bin, number of decades spanned, local exponent over each fifth of the bins)."""
    Sbar, vol = np.asarray(Sbar, float), np.asarray(vol, float)
    P = np.array_split(np.argsort(Sbar), n_bins)
    lx = np.log([Sbar[p].mean() for p in P])
    z, res = [], None
    for q in qs:
        ly = np.log([np.mean(vol[p] ** q) for p in P])
        c = np.polyfit(lx, ly, 1); z.append(-c[0])
        if q == qs[0]:
            res = ly - np.polyval(c, lx)
    mv = np.array([vol[p].mean() for p in P])
    w = max(n_bins // 5, 2)
    loc = [-np.polyfit(lx[i:i + w], np.log(mv[i:i + w]), 1)[0] for i in range(0, n_bins - w + 1, w)]
    return np.array(z), float(res[-1]), float(mv[0]), float(mv[-1]), float((lx[-1] - lx[0]) / np.log(10)), loc
