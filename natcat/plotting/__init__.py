"""natcat.plotting: house style, maps, curves, and animations for figures.

Public surface
--------------
- Style: ``PALETTE``, ``SAFFIR_SIMPSON``, ``apply_style``, ``style_context``,
  ``category_colormap``, ``format_currency_axis``, ``add_source_note``, ``save``.
- Maps (cartopy required, imported lazily on first call): ``base_map``,
  ``plot_track``, ``plot_footprint``, ``plot_genesis``, ``plot_catalog``,
  ``plot_catalog_comparison``.
- Curves (no cartopy dependency): ``plot_wind_profile``, ``plot_vulnerability``,
  ``plot_rmw_heuristic``, ``plot_annual_loss_distribution``, ``plot_ep_curve``.
- Animation (cartopy + Pillow required, imported lazily): ``animate_footprint``.

Every function accepts ``ax=None`` and returns ``(fig, ax)``, never calls
``plt.show()``, and never saves a figure unless the caller passes it to
``save()`` (curves/maps) or ``path=`` (``animate_footprint``). Cartopy is
never imported at module import time, so ``import natcat`` and
``import natcat.plotting`` work with only numpy/pandas/matplotlib installed.
"""

from __future__ import annotations

from natcat.plotting.animation import animate_footprint
from natcat.plotting.calibration import (
    plot_calibration,
    plot_hazard_grid,
    plot_value_dependent_curves,
)
from natcat.plotting.curves import (
    plot_annual_loss_distribution,
    plot_ep_curve,
    plot_land_decay,
    plot_rmw_heuristic,
    plot_vulnerability,
    plot_wind_profile,
)
from natcat.plotting.maps import (
    base_map,
    map_figsize,
    plot_catalog,
    plot_catalog_comparison,
    plot_footprint,
    plot_genesis,
    plot_track,
)
from natcat.plotting.style import (
    PALETTE,
    SAFFIR_SIMPSON,
    add_source_note,
    apply_style,
    category_colormap,
    format_currency_axis,
    save,
    style_context,
)

__all__ = [
    "plot_calibration",
    "plot_hazard_grid",
    "plot_value_dependent_curves",
    "PALETTE",
    "SAFFIR_SIMPSON",
    "add_source_note",
    "animate_footprint",
    "apply_style",
    "base_map",
    "map_figsize",
    "category_colormap",
    "format_currency_axis",
    "plot_annual_loss_distribution",
    "plot_catalog",
    "plot_catalog_comparison",
    "plot_ep_curve",
    "plot_footprint",
    "plot_genesis",
    "plot_land_decay",
    "plot_rmw_heuristic",
    "plot_track",
    "plot_vulnerability",
    "plot_wind_profile",
    "save",
    "style_context",
]
