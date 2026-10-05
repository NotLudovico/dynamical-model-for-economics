"""Figure of the stylised facts (needs facts.py output). Run: uv run python scripts/demographic/facts_plot.py"""
import os, numpy as np, matplotlib.pyplot as plt
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "demographic")
d = np.load(os.path.join(DATA, "facts.npz")); s = np.load(os.path.join(DATA, "facts_summary.npz"))
# all lives born inside the window: |g| and exit hazard by age; hazard by current size
A, Gabs, EX, SZ = [], [], [], []
for S, L in zip(d["S"], d["L"]):
    S = S.astype(float); T = S.shape[1]
    for i in range(S.shape[0]):
        segs = np.split(np.arange(T), np.flatnonzero(np.diff(L[i])) + 1)
        for k, seg in enumerate(segs[1:], 1):        # born inside the window
            x = S[i, seg]
            n = seg.size
            dies = k < len(segs) - 1
            A.append(np.arange(n) * 0.5); SZ.append(x)
            e = np.zeros(n, bool); e[-1] = dies; EX.append(e)
            gg = np.full(n, np.nan); gg[:-1] = np.abs(np.diff(np.log(x))); Gabs.append(gg)
A, Gabs, EX, SZ = map(np.concatenate, (A, Gabs, EX, SZ))
ab = np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 480])
ac = np.sqrt(ab[:-1] * ab[1:] + 0.0625)
haz = np.array([EX[(A >= lo) & (A < hi)].mean() for lo, hi in zip(ab[:-1], ab[1:])])
vol = np.array([np.nanmean(Gabs[(A >= lo) & (A < hi)]) for lo, hi in zip(ab[:-1], ab[1:])])
qs = np.quantile(SZ, np.linspace(0, 1, 21))
szc = np.sqrt(qs[:-1] * qs[1:]); hzs = np.array([EX[(SZ >= lo) & (SZ < hi)].mean() for lo, hi in zip(qs[:-1], qs[1:])])
print("hazard by age:", dict(zip(np.round(ac, 1), np.round(haz, 4))))
print("|g| by age:", dict(zip(np.round(ac, 1), np.round(vol, 3))))
print("hazard by size (20 quantiles):", np.round(hzs, 4))

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
z = s["z"]; h, e = np.histogram(z, np.linspace(-8, 8, 161), density=True); xc = 0.5 * (e[1:] + e[:-1])
ax[0, 0].semilogy(xc, h, color="#104281", lw=2, label="model (lives ≥ 20), rescaled by size bin")
ax[0, 0].semilogy(xc, np.exp(-xc**2 / 2) / np.sqrt(2 * np.pi), ":", color="#333", label="Gaussian")
b = np.sqrt(np.pi / 2) / np.sqrt(2)  # Laplace with same sqrt(pi/2)*MAD = 1 normalisation: MAD = b -> b = sqrt(2/pi)
b = np.sqrt(2 / np.pi); ax[0, 0].semilogy(xc, np.exp(-np.abs(xc) / b) / (2 * b), "--", color="#eb6834", label="Laplace (tent)")
ax[0, 0].set(ylim=(1e-5, 1), title=f"(1) growth-rate distribution: excess kurtosis {np.mean(z**4)/np.mean(z**2)**2-3:.1f}", xlabel="rescaled growth rate")
ax[0, 1].semilogx(s["sb"], s["mg"], "o-", color="#104281", lw=2)
ax[0, 1].axhline(0, color="#333", lw=0.8)
ax[0, 1].set(title="(2) Gibrat: mean growth vs current size (lives ≥ 20)", xlabel="current size S", ylabel="E[g | S] per step")
ls = s["ls"]; h, e = np.histogram(ls, 60, density=True)
ax[0, 2].semilogy(0.5 * (e[1:] + e[:-1]), h, color="#104281", lw=2, label="established firms (lives ≥ 20)")
ha, ea = np.histogram(np.log10(d["S"][:, :, -1].ravel()), 80, density=True)
ax[0, 2].semilogy(0.5 * (ea[1:] + ea[:-1]), ha, color="#86b6ef", lw=2, label="all firms, last snapshot")
ax[0, 2].set(title="(3) size distribution", xlabel="log10 S (units of the mean firm)", ylabel="density", ylim=(1e-4, 3))
L = np.sort(s["lifelen"]); surv = 1 - np.arange(L.size) / L.size
ax[1, 0].loglog(L, surv, color="#104281", lw=2)
ax[1, 0].set(title=f"(4) lifetimes of complete lives: median {np.median(L):.1f}, P(>10) = {np.mean(L > 10):.3f}", xlabel="lifetime (model time)", ylabel="P(lifetime > t)")
ax[1, 1].loglog(ac, haz, "o-", color="#104281", lw=2, label="exit hazard per step")
ax[1, 1].loglog(ac, vol, "s--", color="#2a78d6", lw=2, label="mean |g|")
ax[1, 1].set(title="(5) by age, all lives born in window", xlabel="age (model time)")
ax[1, 2].hist(s["ac1"], np.linspace(-1, 1, 81), color="#2a78d6")
ax[1, 2].axvline(np.median(s["ac1"]), color="#eb6834", lw=2, label=f"median {np.median(s['ac1']):+.2f}")
ax[1, 2].set(title="(6) lag-1 autocorrelation of growth within lives ≥ 20", xlabel="corr(g_t, g_t+1)")
for a in ax.ravel():
    a.legend(fontsize=8, frameon=False); a.grid(True, color="#e6e6e3", lw=0.6)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.suptitle("Other stylised facts: replacement demographic-noise model, α = 2.5, σ = 1.75, D = 0.1, N = 8000, 2 economies, growth step Δt = 0.5", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(DATA, "facts.png"), dpi=110)
