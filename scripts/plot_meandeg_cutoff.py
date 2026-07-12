"""Plot completed matched mean-degree cutoff arms."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
paths = sorted(DATA.glob("meandeg_cutoff_N10000*.npz"))
if not paths:
    raise RuntimeError("no N=10000 cutoff arms found")

records = []
for path in paths:
    data = np.load(path)
    minimum = int(data["min_degree"])
    records.append({
        "label": "legacy" if minimum < 0 else str(minimum),
        "order": float("inf") if minimum < 0 else minimum,
        "valid": int(data["seeds"]),
        "attempted": int(data["attempted_seeds"]) if "attempted_seeds" in data.files else 10,
        "beta": data["beta"],
        "survival": data["survival"],
        "ratio": data["ratio"],
        "d1_tail": data["d1_tail"],
        "d3_trend": data["d3_trend"],
    })
records.sort(key=lambda record: record["order"], reverse=True)

position = np.arange(len(records))
labels = [f"{record['label']}\n{record['valid']}/{record['attempted']} valid" for record in records]

def summary(field):
    values = [record[field] for record in records]
    return np.array([
        np.mean(value) if value.size else np.nan for value in values
    ]), np.array([
        np.std(value) if value.size else np.nan for value in values
    ])

beta, beta_error = summary("beta")
survival, survival_error = summary("survival")
d1_tail, d1_error = summary("d1_tail")
d3_trend, d3_error = summary("d3_trend")
ratio = np.array([
    np.mean(record["ratio"], axis=0) if record["ratio"].size else np.full(4, np.nan)
    for record in records
])
ratio_error = np.array([
    np.std(record["ratio"], axis=0) if record["ratio"].size else np.full(4, np.nan)
    for record in records
])

figure, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
panels = [
    (axes[0, 0], beta, beta_error, r"$\beta$", "size-volatility exponent"),
    (axes[0, 1], survival, survival_error, "survival fraction", "late-window survival"),
    (axes[0, 2], ratio[:, 1], ratio_error[:, 1], r"$\zeta_2/\zeta_1$", "multiscaling ratio"),
    (axes[1, 0], ratio[:, 3], ratio_error[:, 3], r"$\zeta_4/\zeta_1$", "multiscaling ratio"),
    (axes[1, 1], d1_tail, d1_error, "D1 P99/median", "rescaled-volatility tail"),
    (axes[1, 2], d3_trend, d3_error, r"D3 $d\kappa/d\log S$", "unmixed-kurtosis trend"),
]
for axis, value, error, ylabel, title in panels:
    axis.errorbar(position, value, yerr=error, fmt="o-", color="#c0504d", capsize=4)
    axis.axhline(value[0], color="#777777", linestyle="--", linewidth=1)
    axis.set(xticks=position, xticklabels=labels, ylabel=ylabel, title=title)
    axis.set_xlabel(r"degree floor $k_{\min}$")

figure.suptitle(r"Shifted power-law cutoff sweep ($N=10{,}000$, $C=N/40$, 10 matched seeds)")
output = DATA / "meandeg_cutoff_N10000.png"
figure.savefig(output, dpi=160)
print(f"saved {output}")
