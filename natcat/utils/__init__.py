"""General-purpose helpers: geographic maths and unit conversions."""

from .geo import bearing, destination_point, haversine_distance
from .units import (
    km2nm,
    kmh2kt,
    kt2kmh,
    kt2ms,
    nm2km,
    saffir_simpson_category,
)

__all__ = [
    "bearing",
    "destination_point",
    "haversine_distance",
    "km2nm",
    "kmh2kt",
    "kt2kmh",
    "kt2ms",
    "nm2km",
    "saffir_simpson_category",
]
