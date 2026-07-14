"""Compute and plot the heterogeneous-DMFT zero-growth contour.

The theoretical contour is evaluated on one representative graph from the
same N=4000, C=N/40 power-law ensemble as the stored production phase sweep.
The measured contour is obtained by linear interpolation between adjacent
values of the median simulated finite-time growth rate that bracket zero.

Run with:

    .venv/bin/python scripts/plot_hdmft_zero_growth.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from relative_glv.hdmft import (
    bin_degree_distribution,
    heterogeneous_zero_growth,
    heterogeneous_zero_growth_line,
    soft_configuration_kernel,
)
from relative_glv.model import coupling


ROOT = Path(__file__).resolve().parents[1]
PHASE_DATA = ROOT / "data" / "phase_meandeg_fine_24x20_10s.npz"
OUTPUT_DIR = ROOT / "thesis" / "standalone" / "zero_growth_theory"
FIGURE = OUTPUT_DIR / "zero_growth_line.png"
RESULTS = OUTPUT_DIR / "zero_growth_line.npz"
UNSTABLE_LABEL = r"HDMFT continuation ($\rho(\mathcal{L})>1$)"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase = np.load(PHASE_DATA)
    n = int(phase["N"])
    target_degree = n // 40

    adjacency = coupling(
        n,
        mu=0.0,
        sigma=1.0,
        kind="powerlaw",
        seed=0,
        mean_degree=target_degree,
    )
    degrees = np.diff(adjacency.indptr)
    classes = bin_degree_distribution(degrees, n_bins=30)
    kernel = soft_configuration_kernel(
        classes["degree_fractions"],
        classes["weights"],
        classes["connectance"],
    )["kernel"]

    theory_sigmas = np.unique(
        np.concatenate([np.linspace(0.35, 1.25, 25), phase["sigmas"]])
    )
    theory = heterogeneous_zero_growth_line(
        theory_sigmas,
        classes["weights"],
        kernel,
        classes["connectance"],
        lam=0.0,
    )

    def stability_on_contour(sigma: float) -> float:
        point = heterogeneous_zero_growth(
            sigma,
            classes["weights"],
            kernel,
            classes["connectance"],
            lam=0.0,
        )
        return point["fixed_point"]["stability_radius"] - 1.0

    endpoint_sigma = brentq(stability_on_contour, 0.8, 1.2)
    endpoint = heterogeneous_zero_growth(
        endpoint_sigma,
        classes["weights"],
        kernel,
        classes["connectance"],
        lam=0.0,
    )

    median = np.nanmedian(phase["raw"], axis=2)
    growth = median[..., 1]
    observed_mu_c = np.full(phase["sigmas"].shape, np.nan)
    for row_index, row in enumerate(growth):
        brackets = np.where(
            np.isfinite(row[:-1])
            & np.isfinite(row[1:])
            & (row[:-1] * row[1:] <= 0.0)
        )[0]
        if brackets.size:
            j = brackets[0]
            observed_mu_c[row_index] = phase["mus"][j] - row[j] * (
                phase["mus"][j + 1] - phase["mus"][j]
            ) / (row[j + 1] - row[j])
    observed_mask = np.isfinite(observed_mu_c)
    theory_on_grid = heterogeneous_zero_growth_line(
        phase["sigmas"],
        classes["weights"],
        kernel,
        classes["connectance"],
        lam=0.0,
    )
    errors = theory_on_grid["mu_c"][observed_mask] - observed_mu_c[observed_mask]

    np.savez(
        RESULTS,
        sigma=theory["sigma"],
        mu_c=theory["mu_c"],
        mu_thesis=theory["mu_thesis"],
        stability_radius=theory["stability_radius"],
        controlled=theory["controlled"],
        measured_sigma=phase["sigmas"][observed_mask],
        measured_mu_thesis=-observed_mu_c[observed_mask],
        endpoint_sigma=endpoint_sigma,
        endpoint_mu_thesis=endpoint["mu_thesis"],
        realized_mean_degree=degrees.mean(),
        mean_absolute_error=np.mean(np.abs(errors)),
        mean_signed_error=np.mean(errors),
    )

    controlled = theory["controlled"]
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.plot(
        theory["mu_thesis"][controlled],
        theory["sigma"][controlled],
        color="#1f6f8b",
        linewidth=2.4,
        label=r"HDMFT $g^*=0$ (stable fixed point)",
    )
    axis.plot(
        theory["mu_thesis"][~controlled],
        theory["sigma"][~controlled],
        color="#1f6f8b",
        linewidth=2.4,
        linestyle="--",
        label=UNSTABLE_LABEL,
    )
    axis.scatter(
        -observed_mu_c[observed_mask],
        phase["sigmas"][observed_mask],
        s=27,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
        label=r"simulation $g_{\rm eff}=0$",
    )
    axis.scatter(
        [endpoint["mu_thesis"]],
        [endpoint_sigma],
        s=58,
        color="#c1121f",
        zorder=4,
        label=r"$g^*=0$ / stability intersection",
    )
    axis.scatter([-1.76], [1.75], marker="*", s=170, color="black", zorder=4)
    axis.annotate("operating point", (-1.76, 1.75), xytext=(-1.57, 1.81), fontsize=8)
    axis.set(
        xlabel=r"mean interaction $\mu$ (competition: $\mu<0$)",
        ylabel=r"disorder $\sigma$",
        title="Zero-effective-growth contour on the power-law graph",
        xlim=(-2.65, 0.08),
        ylim=(0.3, 2.0),
    )
    axis.grid(alpha=0.18)
    axis.legend(loc="upper right", fontsize=8, framealpha=0.95)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=200)

    print(f"realized mean degree: {degrees.mean():.4f}")
    print(
        "stable endpoint: "
        f"mu={endpoint['mu_thesis']:.6f}, sigma={endpoint_sigma:.6f}"
    )
    print(f"comparison rows: {observed_mask.sum()}")
    print(f"mean absolute error in mu: {np.mean(np.abs(errors)):.6f}")
    print(f"mean signed error (theory-observed, code mu): {np.mean(errors):.6f}")
    print(f"saved {RESULTS}")
    print(f"saved {FIGURE}")


if __name__ == "__main__":
    main()
