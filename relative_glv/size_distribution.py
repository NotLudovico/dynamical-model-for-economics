"""Economy-level estimators for stationary relative firm-size distributions."""
import numpy as np


def economy_ccdf(samples_by_economy, grid):
    """Return empirical CCDFs, weighting each economy equally."""
    grid = np.asarray(grid, dtype=float)
    by_economy = np.array([
        (np.asarray(samples, dtype=float)[:, None] >= grid).mean(axis=0)
        for samples in samples_by_economy
    ])
    return {"by_economy": by_economy, "mean": by_economy.mean(axis=0)}


def hill_tail_index(samples):
    """Estimate the Pareto CCDF exponent from all supplied upper-tail samples."""
    samples = np.sort(np.asarray(samples, dtype=float))
    if samples.size < 2 or samples[0] <= 0:
        raise ValueError("at least two positive tail samples are required")
    return float(1.0 / np.mean(np.log(samples / samples[0])))
