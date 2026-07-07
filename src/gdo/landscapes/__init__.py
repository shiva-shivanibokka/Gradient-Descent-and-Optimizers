"""Loss surface definitions and visualization utilities."""

from gdo.landscapes.plotter import LandscapePlotter
from gdo.landscapes.surfaces import (
    Beale,
    Himmelblau,
    LossSurface,
    QuadraticSurface,
    Rosenbrock,
)

__all__ = [
    "LossSurface",
    "QuadraticSurface",
    "Rosenbrock",
    "Beale",
    "Himmelblau",
    "LandscapePlotter",
]
