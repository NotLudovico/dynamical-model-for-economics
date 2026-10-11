"""Fixed scorecard for the model screen (decided before the screen runs).  Input: X[r, i] = x_i/m
recorded every time unit, fid[r, i] the firm id at node i.

A firm is listed while x/m >= LIST (1e-3, the low end of the MSB size range).  A spell is a maximal run
of records with the same firm id, listed, with >= 2 growth rates; its volatility is sqrt(pi/2) MAD of
its yearly log growth about its own mean, Sbar its mean x/m (MSB 2024, Sec. 3.1).  Exponents use all
25 equal-count bins of Sbar.

Checks (target in brackets):
  small     share of all firm-time in each decade of x/m from 1e-3 to 1   [each >= 5%]
  zipf      Hill index of the top 1% of listed sizes                    [0.8 - 1.3]
  nocond    mean share of the largest firm                              [< 0.05]
  beta      MSB beta (spells, all bins)                                 [0.12 - 0.30]
  ratios    zeta_q/zeta_1, q = 2, 3, 4                                  [within 0.3 of 2.0 2.6 2.9]
  straight  spread of the local beta over the 5 fifths of bins          [< 0.25]
  tailb     tail index b of sigma/sigmabar (top 5%, pooled)             [3.5 - 6]
  kurt      rescaled-growth excess kurtosis, top / bottom Sbar fifth   [both > 1, ratio < 4]
"""
import numpy as np

LIST = 1e-3
DATA_RATIOS = np.array([2.0, 2.6, 2.9])


def _hill(z, p):
    z = np.sort(z)[::-1]
    k = max(int(p * len(z)), 10)
    return float(1.0 / np.mean(np.log(z[:k] / z[k])))


def score(X, fid):
    R, N = X.shape
    out = {}
    lx = np.log10(np.maximum(X, 1e-30))
    for lo in (-5, -4, -3, -2, -1, 0, 1):
        out[f"dec{lo}"] = float(np.mean((lx >= lo) & (lx < lo + 1)))
    listed = X >= LIST
    snap = X[::5][listed[::5]]
    out["hill1"], out["hill5"] = _hill(snap, 0.01), _hill(snap, 0.05)
    h = R // 2                                       # stationarity: Hill in each half of the window
    out["hill1_halves"] = [_hill(X[:h:5][listed[:h:5]], 0.01), _hill(X[h::5][listed[h::5]], 0.01)]
    out["top_share"] = float((X.max(1) / X.sum(1)).mean())
    # current-size volatility (growth over one record), both ends listed, same firm
    same = (fid[1:] == fid[:-1]) & listed[1:] & listed[:-1]
    g = np.log(np.maximum(X[1:], 1e-30)) - np.log(np.maximum(X[:-1], 1e-30))
    s0, gg = X[:-1][same], g[same]
    if s0.size > 1000:
        q = np.quantile(s0, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(s0, q[1:-1]), 0, 9)
        cs = np.array([np.log10(s0[idx == j].mean()) if (idx == j).any() else np.nan for j in range(10)])
        cv = np.array([np.log10(gg[idx == j].std()) if (idx == j).sum() > 1 else np.nan for j in range(10)])
        ok = np.isfinite(cs) & np.isfinite(cv)
        cs, cv = cs[ok], cv[ok]
        out["cur_slope"] = float(-np.polyfit(cs, cv, 1)[0]) if ok.sum() >= 3 else np.nan
        out["cur_curve"] = [round(float(a), 2) for a in cs], [round(float(10 ** b), 3) for b in cv]
    # spells
    Sb, vol, res, ln = [], [], [], []
    for i in range(N):
        ok = listed[:, i]
        key = np.where(ok, fid[:, i], -1)
        cut = np.flatnonzero(np.diff(key)) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, R]):
            if key[a] < 0 or b - a < 3:
                continue
            u = np.log(X[a:b, i].astype(float)); gr = np.diff(u); gc = gr - gr.mean()
            v = np.sqrt(np.pi / 2) * np.abs(gc).mean()
            if v <= 0:
                continue
            Sb.append(np.exp(u).mean()); vol.append(v); ln.append(len(gr)); res.append(gc / v)
    Sb, vol, ln = np.array(Sb), np.array(vol), np.array(ln)
    out["n_spells"], out["median_life"] = int(len(Sb)), float(np.median(ln)) if len(ln) else 0.0
    if len(Sb) < 500:
        return out
    o = np.argsort(Sb); bins = np.array_split(o, 25)
    lS = np.log([Sb[b].mean() for b in bins])
    z = np.array([-np.polyfit(lS, np.log([np.mean(vol[b] ** q) for b in bins]), 1)[0] for q in (1, 2, 3, 4)])
    mv = np.log([vol[b].mean() for b in bins])
    out["range_dec"] = float((lS[-1] - lS[0]) / np.log(10))
    out["beta"], out["ratios"] = float(z[0]), [float(r) for r in z[1:] / z[0]]
    out["local_beta"] = [float(-np.polyfit(lS[k:k + 5], mv[k:k + 5], 1)[0]) for k in range(0, 25, 5)]
    out["bins"] = [round(float(a / np.log(10)), 2) for a in lS], [round(float(np.exp(b)), 3) for b in mv]
    r = np.empty_like(vol)
    for b in bins:
        r[b] = vol[b] / vol[b].mean()
    out["tail_b"], out["cv"] = _hill(r, 0.05), float(np.std(r))
    fifth = np.empty(len(Sb), int)
    for k in range(5):
        for b in bins[5 * k:5 * k + 5]:
            fifth[b] = k
    ku = []
    for k in (0, 4):
        zz = np.concatenate([res[j] for j in np.flatnonzero((fifth == k) & (ln >= 10))] or [np.zeros(1)])
        ku.append(float(np.mean(zz ** 4) / np.mean(zz ** 2) ** 2 - 3) if zz.size > 100 else np.nan)
    out["kurt_lo"], out["kurt_hi"] = ku
    b1 = (fid[2:] == fid[1:-1]) & (fid[1:-1] == fid[:-2]) & listed[2:] & listed[1:-1] & listed[:-2]
    out["lag1"] = float(np.corrcoef(g[1:][b1], g[:-1][b1])[0, 1]) if b1.sum() > 100 else np.nan
    return out


def checks(s):
    """Pass/fail of each scorecard item (None if not measurable)."""
    c = {}
    c["small"] = all(s.get(f"dec{k}", 0) >= 0.05 for k in (-3, -2, -1))
    c["zipf"] = 0.8 <= s.get("hill1", 0) <= 1.3
    c["nocond"] = s.get("top_share", 1) < 0.05
    if "beta" in s:
        c["beta"] = 0.12 <= s["beta"] <= 0.30
        c["ratios"] = bool(np.all(np.abs(np.array(s["ratios"]) - DATA_RATIOS) <= 0.3))
        c["straight"] = float(np.ptp(s["local_beta"])) < 0.25
        c["tailb"] = 3.5 <= s["tail_b"] <= 6
        kl, kh = s["kurt_lo"], s["kurt_hi"]
        c["kurt"] = bool(kl > 1 and kh > 1 and max(kl, kh) / min(kl, kh) < 4) if np.isfinite(kl) and np.isfinite(kh) else False
    return c


def comove(X, fid, lm, W=20, rng=np.random.default_rng(0)):
    """Reported, not a check.  agg_vol: sd of the one-record growth of the total (ln m); pair_corr:
    mean pairwise correlation of firms' absolute one-record log growth, over windows of W records,
    among firms listed with the same id throughout (<= 300 per window)."""
    g = np.diff(np.log(np.maximum(X, 1e-30)), axis=0) + np.diff(lm)[:, None]
    out = {"agg_vol": float(np.std(np.diff(lm)))}
    cs = []
    for a in range(0, X.shape[0] - W, W):
        ok = np.all(fid[a:a + W] == fid[a], axis=0) & np.all(X[a:a + W] >= LIST, axis=0)
        idx = np.flatnonzero(ok)
        if idx.size < 20:
            continue
        idx = rng.choice(idx, min(300, idx.size), replace=False)
        c = np.corrcoef(g[a:a + W - 1, idx].T)
        cs.append(np.nanmean(c[np.triu_indices_from(c, 1)]))
    out["pair_corr"] = float(np.mean(cs)) if cs else np.nan
    return out
