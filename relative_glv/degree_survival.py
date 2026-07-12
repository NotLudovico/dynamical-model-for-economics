"""Conditional survival summaries for heterogeneous interaction networks."""
import numpy as np


def survival_by_degree(degree, survived, *, bin_edges, confidence=0.95):
    """Estimate P(survive | degree) in supplied degree bins.

    Bins include their lower bound and exclude their upper bound, except the
    final bin, which includes both bounds. Confidence intervals use the Wilson
    score interval for a binomial proportion.
    """
    degree = np.asarray(degree, dtype=float)
    survived = np.asarray(survived, dtype=bool)
    bin_edges = np.asarray(bin_edges, dtype=float)
    if degree.ndim != 1 or survived.ndim != 1 or degree.shape != survived.shape:
        raise ValueError("degree and survived must be one-dimensional arrays of equal length")
    if bin_edges.ndim != 1 or bin_edges.size < 2 or np.any(np.diff(bin_edges) <= 0):
        raise ValueError("bin_edges must be a strictly increasing one-dimensional array")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between 0 and 1")

    valid = np.isfinite(degree) & (degree >= bin_edges[0]) & (degree <= bin_edges[-1])
    indices = np.digitize(degree[valid], bin_edges[1:-1])
    count = np.bincount(indices, minlength=bin_edges.size - 1)
    survived_count = np.bincount(indices, weights=survived[valid], minlength=bin_edges.size - 1)
    probability = np.divide(
        survived_count,
        count,
        out=np.full(count.size, np.nan, dtype=float),
        where=count > 0,
    )

    z = 1.959963984540054 if confidence == 0.95 else _normal_quantile((1 + confidence) / 2)
    populated = count > 0
    center = np.full(count.size, np.nan, dtype=float)
    half_width = np.full(count.size, np.nan, dtype=float)
    denominator = 1 + z ** 2 / count[populated]
    center[populated] = (probability[populated] + z ** 2 / (2 * count[populated])) / denominator
    half_width[populated] = z * np.sqrt(
        probability[populated] * (1 - probability[populated]) / count[populated]
        + z ** 2 / (4 * count[populated] ** 2)
    ) / denominator

    return {
        "degree": np.sqrt(bin_edges[:-1] * bin_edges[1:]),
        "count": count,
        "survived": survived_count.astype(int),
        "probability": probability,
        "ci_low": center - half_width,
        "ci_high": center + half_width,
    }


def survival_by_degree_by_economy(degree, survived, economy, *, bin_edges, confidence=0.95):
    """Estimate P(survive | degree) with equal weight for every economy.

    Each economy contributes its within-bin survival rate once, irrespective of
    how many firms it places in that bin. Confidence intervals are t intervals
    over the available economy-level rates.
    """
    degree = np.asarray(degree, dtype=float)
    survived = np.asarray(survived, dtype=bool)
    economy = np.asarray(economy)
    if economy.ndim != 1 or economy.shape != degree.shape:
        raise ValueError("economy must be a one-dimensional array matching degree")

    pooled = survival_by_degree(degree, survived, bin_edges=bin_edges, confidence=confidence)
    bin_edges = np.asarray(bin_edges, dtype=float)
    valid = np.isfinite(degree) & (degree >= bin_edges[0]) & (degree <= bin_edges[-1])
    labels = np.unique(economy[valid])
    rates = np.full((labels.size, bin_edges.size - 1), np.nan)
    for row, label in enumerate(labels):
        in_economy = valid & (economy == label)
        indices = np.digitize(degree[in_economy], bin_edges[1:-1])
        count = np.bincount(indices, minlength=bin_edges.size - 1)
        survived_count = np.bincount(
            indices, weights=survived[in_economy], minlength=bin_edges.size - 1
        )
        rates[row] = np.divide(
            survived_count,
            count,
            out=np.full(count.size, np.nan, dtype=float),
            where=count > 0,
        )

    economies = np.isfinite(rates).sum(axis=0)
    probability = np.divide(
        np.nansum(rates, axis=0),
        economies,
        out=np.full(economies.size, np.nan, dtype=float),
        where=economies > 0,
    )
    deviation = np.where(np.isfinite(rates), rates - probability, 0.0)
    variance = np.divide(
        np.square(deviation).sum(axis=0),
        economies - 1,
        out=np.full(economies.size, np.nan, dtype=float),
        where=economies > 1,
    )
    half_width = _t_quantile((1 + confidence) / 2, economies - 1) * np.sqrt(variance / economies)
    return {
        **pooled,
        "probability": probability,
        "ci_low": np.clip(probability - half_width, 0.0, 1.0),
        "ci_high": np.clip(probability + half_width, 0.0, 1.0),
        "economies": economies,
        "economy_probability": rates,
    }


def reentry_masks(active):
    """Classify persistent, exited, and genuinely re-entering trajectories."""
    active = np.asarray(active, dtype=bool)
    if active.ndim != 2 or active.shape[1] < 2:
        raise ValueError("active must have shape (firms, at least two time points)")

    exits = active[:, :-1] & ~active[:, 1:]
    entries = ~active[:, :-1] & active[:, 1:]
    prior_exit = np.cumsum(exits, axis=1) > 0
    return {
        "persistent": active.all(axis=1),
        "exited": exits.any(axis=1),
        "reentered": (entries & prior_exit).any(axis=1),
        "entry_count": entries.sum(axis=1),
    }


def _normal_quantile(probability):
    from scipy.stats import norm

    return float(norm.ppf(probability))


def _t_quantile(probability, degrees_of_freedom):
    from scipy.stats import t

    return t.ppf(probability, degrees_of_freedom)
