"""The relative (scale-invariant) generalised Lotka-Volterra model.

A disordered replicator in which interactions act on the mean firm, giving a
self-consistently growing economy whose firm-size dynamics reproduce the
Moran-Santos-Bouchaud (MSB) stylized facts: a symmetric fat-tailed growth-rate
distribution and a size-variance exponent beta ~ 0.15-0.20.
"""
from relative_glv.model import coupling, integrate, growth_rate, survivors
from relative_glv.msb import rescale, size_volatility, tent_stats
from relative_glv.dmft import solve_fixed_point, sigma_c, solve_twotime
from relative_glv.hdmft import (
    heterogeneous_sigma_c,
    heterogeneous_zero_growth,
    heterogeneous_zero_growth_line,
    solve_heterogeneous_fixed_point,
)

__all__ = [
    "coupling", "integrate", "growth_rate", "survivors",
    "rescale", "size_volatility", "tent_stats",
    "solve_fixed_point", "sigma_c", "solve_twotime",
    "solve_heterogeneous_fixed_point", "heterogeneous_sigma_c",
    "heterogeneous_zero_growth", "heterogeneous_zero_growth_line",
]
