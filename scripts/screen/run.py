"""Model screen: run every variant of variants.py on the same graphs, score each run with score.py and
print one table.  Runs are independent, one per core; results are cached per run in data/screen/
(json score + npz trajectory), so re-running skips finished points.

Run (repo root):  uv run python scripts/screen/run.py            # full grid, 12 workers
                  uv run python scripts/screen/run.py summary    # table only
                  uv run python scripts/screen/run.py base       # only variants whose name starts with base
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np
from joblib import Parallel, delayed

OUT = os.path.join(ROOT, "data", "screen")
ALPHAS, SEEDS = (2.5, 6.0), (0, 1)
GRID = [  # (label, variant, extra params); one new parameter per variant at most
    ("base", "base", {}),
    ("base+exit.02", "base", {"delta": 0.02}),
    ("base+exit.1", "base", {"delta": 0.1}),
    ("base_demog D.01", "base_demog", {"D": 1e-2}),
    ("base_demog D.1", "base_demog", {"D": 1e-1}),
    ("nobrake", "nobrake", {}),
    ("nobrake+exit.02", "nobrake", {"delta": 0.02}),
    ("nobrake+exit.1", "nobrake", {"delta": 0.1}),
    ("capacity eps.1", "capacity", {"eps": 0.1}),
    ("capacity eps1", "capacity", {"eps": 1.0}),
    ("demog D.01", "demog", {"D": 1e-2}),
    ("demog D.1", "demog", {"D": 1e-1}),
    ("sublin a.6", "sublin", {"a": 0.6}),
    ("sublin a.8", "sublin", {"a": 0.8}),
    ("minflow", "minflow", {}),
    ("minflow_nb", "minflow_nb", {}),
    ("inertia tau.1", "inertia", {"tau0": 0.1}),
    ("inertia tau1", "inertia", {"tau0": 1.0}),
    # brake-free variants condense with quenched couplings; the same with random exit
    ("capacity eps1+exit.05", "capacity", {"eps": 1.0, "delta": 0.05}),
    ("demog D.1+exit.05", "demog", {"D": 1e-1, "delta": 0.05}),
    ("sublin a.8+exit.05", "sublin", {"a": 0.8, "delta": 0.05}),
    ("minflow_nb+exit.05", "minflow_nb", {"delta": 0.05}),
    ("inertia tau1+exit.05", "inertia", {"tau0": 1.0, "delta": 0.05}),
]
GRID2 = [  # round 2: couplings redrawn at rate 1/tau (variants2.py); exit 0.05 unless stated
    ("ann tau.3", "ann", {"tau": 0.3, "delta": 0.0}),
    ("ann tau3", "ann", {"tau": 3.0, "delta": 0.0}),
    ("ann_nb tau.3", "ann_nb", {"tau": 0.3}),
    ("ann_nb tau3", "ann_nb", {"tau": 3.0}),
    ("ann_nb mu0 tau.3", "ann_nb", {"tau": 0.3, "mu": 0.0}),
    ("ann_nb mu0 tau3", "ann_nb", {"tau": 3.0, "mu": 0.0}),
    ("ann_minflow tau.3", "ann_minflow", {"tau": 0.3}),
    ("ann_minflow tau3", "ann_minflow", {"tau": 3.0}),
    ("rebirth_nb", "rebirth_nb", {}),
    ("rebirth_base", "rebirth_base", {}),
    ("share tau.3", "share", {"tau": 0.3}),
    ("share tau3", "share", {"tau": 3.0}),
    ("scope_share g.3", "scope_share", {"gs": 0.3}),
    ("scope_share g.6", "scope_share", {"gs": 0.6}),
    ("scope_share g1", "scope_share", {"gs": 1.0}),
    ("scope g.6", "scope", {"gs": 0.6}),
]
GRID3 = [  # round 3: the partner model (variants3.py); gamma 0 is the no-scope control
    (f"partner g{g} s{s0}", "partner", {"gamma": g, "s0": s0})
    for g in (0.0, 0.5, 1.0) for s0 in (0.05, 0.2)
]
GRID4 = [  # round 4: the sales-network model (variants4.py); a = 0 switches off upstream propagation
    (f"salesnet a{a} k{k} kh{kh}", "salesnet", {"a": a, "k": k, "kh": kh})
    for a in (0.0, 0.5) for k in (2, 3, 5) for kh in (1, 3, 10)
]
if sys.argv[1:2] == ["r2"]:                   # run.py r2 [summary | auto | prefix]
    GRID = GRID2; del sys.argv[1]
elif sys.argv[1:2] == ["r3"]:                 # no graph: one dummy alpha, four seeds
    GRID, ALPHAS, SEEDS = GRID3, (0.0,), (0, 1, 2, 3); del sys.argv[1]
elif sys.argv[1:2] == ["r4"]:                 # no graph: one dummy alpha, two seeds
    GRID, ALPHAS, SEEDS = GRID4, (0.0,), (0, 1); del sys.argv[1]


def name(label, alpha, seed):
    return f"{label.replace(' ', '_')}_a{alpha}_s{seed}"


def job(label, variant, extra, alpha, seed):
    import traceback
    import variants, variants2, variants3, variants4
    simulate = variants2.simulate if variant in variants2.VARIANTS else variants.simulate
    if variant == "partner":
        simulate = lambda v, **k: variants3.simulate(**k)
    if variant == "salesnet":
        simulate = lambda v, **k: variants4.simulate(**k)
    from score import score, checks
    f = os.path.join(OUT, name(label, alpha, seed))
    if os.path.exists(f + ".json"):
        return
    t0 = time.time()
    try:
        if os.path.exists(f + ".npz"):                       # trajectory cached: rescore only
            d = np.load(f + ".npz"); X, fid, births = d["X"], d["fid"], float(d["births"])
        else:
            r = simulate(variant, alpha=alpha, seed=seed, **extra)
            if r is None:
                X = None
            else:
                X, fid, births = r["X"], r["fid"], r["births"]
                np.savez_compressed(f + ".npz", X=X, fid=fid, deg=r["deg"], t=r["t"], births=births,
                                    clipped=r.get("clipped", 0.0), lm=r.get("lm", np.zeros(0)),
                                    npart=r.get("npart", np.nan), H=r.get("H", np.zeros(0)))
        if X is None:
            s = {"failed": "overflow"}
        else:
            s = score(X, fid); s["births"] = births
            if os.path.exists(f + ".npz"):
                d = np.load(f + ".npz")
                s["clipped"] = float(d.get("clipped", 0.0)); s["npart"] = float(d.get("npart", np.nan))
                if d.get("lm", np.zeros(0)).size == X.shape[0]:
                    from score import comove
                    s.update(comove(X, fid, d["lm"]))
    except Exception:
        s = {"failed": traceback.format_exc(limit=3)}
    s.update(label=label, variant=variant, extra=extra, alpha=alpha, seed=seed, secs=time.time() - t0,
             checks=checks(s) if "failed" not in s else {})
    json.dump(s, open(f + ".json", "w"), default=float)


def _lag1(X, fid):
    from score import LIST
    lg = np.log(np.maximum(X, 1e-30)); g = np.diff(lg, axis=0); L = X >= LIST
    b = (fid[2:] == fid[1:-1]) & (fid[1:-1] == fid[:-2]) & L[2:] & L[1:-1] & L[:-2]
    return float(np.corrcoef(g[1:][b], g[:-1][b])[0, 1]) if b.sum() > 100 else np.nan


def rescore_one(path):
    """Year = shortest record step k in STEPS with |lag-1 growth autocorrelation| < 0.2 (annual
    Compustat growth is nearly uncorrelated); score at that step -> <name>.auto.json."""
    from score import score, checks
    out = path.replace(".npz", ".auto.json")
    if os.path.exists(out):
        return
    base = json.load(open(path.replace(".npz", ".json")))
    d = np.load(path); X, fid = d["X"], d["fid"]
    lags = {k: _lag1(X[::k], fid[::k]) for k in STEPS}
    k = next((k for k in STEPS if abs(lags[k]) < 0.2), STEPS[-1])
    try:
        s = score(X[::k], fid[::k])
    except Exception as e:
        s = {"failed": repr(e)}
    s.update({kk: base[kk] for kk in ("label", "variant", "extra", "alpha", "seed")}, year=k,
             lags={str(a): b for a, b in lags.items()}, checks=checks(s) if "failed" not in s else {})
    json.dump(s, open(out, "w"), default=float)


STEPS = (1, 2, 3, 5, 8)
MODE = ".json"


def summary():
    rows = []
    for label, variant, extra in GRID:
        for alpha in ALPHAS:
            ss = [json.load(open(p)) for p in (os.path.join(OUT, name(label, alpha, s) + MODE) for s in SEEDS)
                  if os.path.exists(p)]
            if not ss:
                continue
            if any(s.get("failed") for s in ss):
                rows.append((label, alpha, None)); continue
            rows.append((label, alpha, ss))
    keys = ("small", "zipf", "nocond", "beta", "ratios", "straight", "tailb", "kurt")
    print(f"{'variant':22} {'al':>3} {'yr':>2} | {'dec-3 -2 -1':>12} {'hill':>5} {'top':>5} {'cur':>5} | {'beta':>6} "
          f"{'ratios':>15} {'loc.spr':>7} {'b':>4} {'kurt lo/hi':>10} {'lag1':>5} | pass  failing")
    for label, alpha, ss in sorted(rows, key=lambda r: -np.mean([sum(bool(v) for v in s["checks"].values())
                                                                     for s in r[2]]) if r[2] else 1):
        if ss is None:
            print(f"{label:22} {alpha:>3} | numerical failure"); continue
        m = lambda k: np.nanmean([s.get(k, np.nan) for s in ss])
        rat = np.nanmean([s["ratios"] for s in ss if "ratios" in s], 0) if any("ratios" in s for s in ss) else [np.nan] * 3
        spr = np.nanmean([np.ptp(s["local_beta"]) for s in ss if "local_beta" in s]) if any("local_beta" in s for s in ss) else np.nan
        npass = np.mean([sum(bool(v) for v in s["checks"].values()) for s in ss])
        fail = [k for k in keys if not all(s["checks"].get(k, False) for s in ss)]
        print(f"{label:22} {alpha:>3} {m('year'):2.0f} | {100*m('dec-3'):3.0f}{100*m('dec-2'):4.0f}{100*m('dec-1'):4.0f}% {m('hill1'):5.2f} "
              f"{m('top_share'):5.3f} {m('cur_slope'):5.2f} | {m('beta'):6.3f} {rat[0]:4.2f} {rat[1]:4.2f} {rat[2]:5.2f} "
              f"{spr:7.2f} {m('tail_b'):4.1f} {m('kurt_lo'):4.1f}/{m('kurt_hi'):4.1f} {m('lag1'):5.2f} | {npass:3.1f}  {' '.join(fail)}"
              + (f"   [partners {m('npart'):.0f}, pair corr {m('pair_corr'):.3f}, agg vol {m('agg_vol'):.3f}]"
                 if np.isfinite(m('npart')) else ""))
    print("targets: dec>=5% each, hill .8-1.3, top<.05, beta .12-.30, ratios 2.0 2.6 2.9 (+-.3), "
          "local-beta spread<.25, b 3.5-6, kurt>1 both and ratio<4")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:] == ["summary"]:
        summary(); sys.exit()
    if sys.argv[1:2] == ["auto"]:                 # rescore every cached run with the calibrated year
        import glob
        mine = {name(l, a, s) for l, _, _ in GRID for a in ALPHAS for s in SEEDS}
        Parallel(n_jobs=12)(delayed(rescore_one)(p) for p in sorted(glob.glob(os.path.join(OUT, "*.npz")))
                            if os.path.basename(p)[:-4] in mine)
        MODE = ".auto.json"; summary(); sys.exit()
    pref = sys.argv[1] if len(sys.argv) > 1 else ""
    jobs = [(l, v, e, a, s) for l, v, e in GRID if l.startswith(pref) for a in ALPHAS for s in SEEDS]
    jobs.sort(key=lambda j: "minflow" in j[1] or "scope" in j[1], reverse=True)   # slowest first
    Parallel(n_jobs=12, verbose=5)(delayed(job)(*j) for j in jobs)
    summary()
