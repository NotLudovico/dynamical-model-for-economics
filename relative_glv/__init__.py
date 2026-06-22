"""The relative (scale-invariant) generalised Lotka-Volterra model.

A disordered replicator in which interactions act on the mean firm, giving a
self-consistently growing economy whose firm-size dynamics reproduce the
Moran-Santos-Bouchaud (MSB) stylized facts: a symmetric fat-tailed growth-rate
distribution and a size-variance exponent beta ~ 0.15-0.20.
"""
from relative_glv.model import coupling, integrate, growth_rate, survivors
from relative_glv.msb import rescale, size_volatility, tent_stats
from relative_glv.dmft import solve_fixed_point, sigma_c

__all__ = [
    "coupling", "integrate", "growth_rate", "survivors",
    "rescale", "size_volatility", "tent_stats",
    "solve_fixed_point", "sigma_c",
]
