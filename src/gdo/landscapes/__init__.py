"""Loss surface definitions and visualization utilities."""

from gdo.landscapes.surfaces import (
    LossSurface,
    QuadraticSurface,
    Rosenbrock,
    Beale,
    Himmelblau,
)
from gdo.landscapes.plotter import LandscapePlotter

__all__ = [
    "LossSurface",
    "QuadraticSurface",
    "Rosenbrock",
    "Beale",
    "Himmelblau",
    "LandscapePlotter",
]
