"""Helper functions for plotting operations.

This module provides utility functions used across the plotting module.
"""

from collections.abc import Iterable
from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from ..get import isiterable, obs_categories

# Statistical formatting constants
PVAL_THRESHOLDS = [1e-3, 1e-2, 5e-2]
PVAL_STAR_SYMBOL = "\u204e"
PVAL_NOT_SIGNIFICANT = "ns"

# Default scaling factors
DEFAULT_SIZE_SCALE = 1.0
DEFAULT_SIZE_MULTIPLIER = 10.0


# Figure and layout helpers
def _get_default_figsize() -> tuple[float, float]:
    """Get default figure size from matplotlib rcParams."""
    return plt.rcParams["figure.figsize"]


def _get_default_fontsize() -> float:
    """Get default font size from matplotlib rcParams."""
    return plt.rcParams["font.size"]


def _get_scaled_figsize(
    x_scale: float = 1.0, y_scale: float = 1.0
) -> tuple[float, float]:
    """Calculate scaled figure size based on default figsize.

    Parameters
    ----------
    x_scale
        Scaling factor for width.
    y_scale
        Scaling factor for height.

    Returns
    -------
    Tuple of (width, height) in inches.
    """
    figsize = _get_default_figsize()
    return (figsize[0] * x_scale, figsize[1] * y_scale)


def _get_scaled_marker_size(
    cell_count: int,
    figsize: Optional[tuple[float, float]] = None,
    scale: float = DEFAULT_SIZE_SCALE,
) -> float:
    """Calculate marker size based on cell count and figure dimensions.

    Parameters
    ----------
    cell_count
        Number of cells/points to be plotted.
    figsize
        Figure size as (width, height). If None, uses default from rcParams.
    scale
        Additional scaling factor.

    Returns
    -------
    Calculated marker size.
    """
    figsize = figsize if figsize is not None else _get_default_figsize()
    area = figsize[0] * figsize[1]
    fontsize = _get_default_fontsize()

    # Two scaling approaches: fixed minimum and cell-count dependent
    min_size = fontsize / area
    scaled_size = (fontsize * DEFAULT_SIZE_MULTIPLIER * area) / np.sqrt(cell_count)

    return max(min_size, scaled_size) * scale


# Color and palette helpers
def _get_default_color_cycle() -> list[str]:
    """Get default color cycle from matplotlib rcParams."""
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _extract_colors_from_colormap(
    cmap: mpl.colors.Colormap, n_colors: int
) -> list[str]:
    """Extract colors from a colormap as hex strings.

    Parameters
    ----------
    cmap
        Matplotlib colormap object.
    n_colors
        Number of colors to extract.

    Returns
    -------
    List of hex color strings.
    """
    if isinstance(cmap, mpl.colors.ListedColormap):
        colors = [cmap(i) for i in range(min(cmap.N, n_colors))]
    else:
        # For continuous colormaps, sample evenly across the range
        if n_colors == 1:
            colors = [cmap(0.5)]
        else:
            colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    return [mpl.colors.to_hex(color) for color in colors]


def _create_color_palette(
    palette: Union[str, Iterable[str], mpl.colors.Colormap, None], n_colors: int
) -> list[str]:
    """Create a color palette from various input types.

    Parameters
    ----------
    palette
        Color specification (colormap name, colormap object, or list of colors).
    n_colors
        Number of colors needed.

    Returns
    -------
    List of hex color strings.
    """
    if palette is None:
        return _get_default_color_cycle()[:n_colors]

    if isinstance(palette, str):
        # Assume it's a colormap name
        cmap = mpl.colormaps[palette]
        return _extract_colors_from_colormap(cmap, n_colors)

    if isinstance(palette, mpl.colors.Colormap):
        return _extract_colors_from_colormap(palette, n_colors)

    if isiterable(palette):
        # Direct color specification
        colors = list(palette)
        # Extend with cycling if needed
        while len(colors) < n_colors:
            colors.extend(colors)
        return [mpl.colors.to_hex(color) for color in colors[:n_colors]]

    # Fallback to default
    return _get_default_color_cycle()[:n_colors]


def _get_color_palette(
    adata: sc.AnnData,
    key: str,
    palette: Optional[Union[str, Iterable[str], mpl.colors.Colormap]] = None,
    force_update: bool = True,
    as_dict: bool = False,
) -> Union[dict[str, str], list[str]]:
    """Get or create color palette for categorical data.

    Parameters
    ----------
    adata
        Annotated data object.
    key
        Key for categorical data in adata.obs.
    palette
        Color specification. If None, uses stored colors or defaults.
    force_update
        If True, always create new palette even if colors exist.
    as_dict
        If True, return as dict mapping categories to colors.

    Returns
    -------
    Color palette as dict or list depending on as_dict parameter.
    """
    if key is None:
        categories = [None]
    else:
        categories = obs_categories(adata, key)

    color_key = f"{key}_colors"

    # Check for existing colors
    if color_key in adata.uns and not force_update and palette is None:
        colors = adata.uns[color_key]
    else:
        # Create new palette
        colors = _create_color_palette(palette, len(categories))
        if key is not None:
            adata.uns[color_key] = colors

    # Ensure we have the right number of colors
    if len(colors) < len(categories):
        # Extend with default cycle if needed
        default_colors = _get_default_color_cycle()
        while len(colors) < len(categories):
            colors.extend(default_colors)
        colors = colors[: len(categories)]
        if key is not None:
            adata.uns[color_key] = colors

    return dict(zip(categories, colors)) if as_dict else colors


# Statistical formatting helpers
def _format_pvalue(pval: float, return_stars: bool = True) -> str:
    """Format p-value as stars or scientific notation.

    Parameters
    ----------
    pval
        P-value to format.
    return_stars
        If True, return star symbols. If False, return scientific notation.

    Returns
    -------
    Formatted p-value string.
    """
    if return_stars:
        stars = ""
        for threshold in PVAL_THRESHOLDS:
            if pval <= threshold:
                stars += PVAL_STAR_SYMBOL
        return stars if stars else PVAL_NOT_SIGNIFICANT
    else:
        return f"{pval:.2e}"


# Backwards compatibility aliases
get_figsize = _get_scaled_figsize
format_pval = _format_pvalue
get_palette = _get_color_palette
get_marker_size = _get_scaled_marker_size
