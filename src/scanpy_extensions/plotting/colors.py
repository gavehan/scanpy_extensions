"""Color palettes and custom colormaps for scanpy extensions.

This module provides predefined color palettes and custom colormap definitions.
"""

import warnings
from typing import Optional

import numpy as np

# =============================================================================
# Predefined Color Palettes (following matplotlib's tableau colors convention)
# =============================================================================

# Tableau color palettes - reordered and enhanced versions
TABLEAU_10_COLORS = [
    "#76B7B2",  # Teal
    "#FF9DA7",  # Pink
    "#4E79A7",  # Blue
    "#F28E2B",  # Orange
    "#59A14F",  # Green
    "#E15759",  # Red
    "#B07AA1",  # Purple
    "#EDC948",  # Yellow
    "#BAB0AC",  # Gray
    "#9C755F",  # Brown
]

TABLEAU_2_COLORS = ["#499894", "#FF9DA7"]

TABLEAU_3_COLORS = ["#499894", "#F28E2B", "#FF9DA7"]

TABLEAU_4_COLORS = ["#499894", "#FF9DA7", "#E15759", "#9d3c3e"]

# Alternative 4-color palette
TABLEAU_4_ALT_COLORS = [
    "#FF9DA7",  # Pink
    "#AB414C",  # Dark Red
    "#2880DE",  # Bright Blue
    "#4E79A7",  # Blue
]

# Backward compatibility aliases
tab10 = TABLEAU_10_COLORS
tab2 = TABLEAU_2_COLORS
tab3 = TABLEAU_3_COLORS
tab4 = TABLEAU_4_COLORS
tab4_2 = TABLEAU_4_ALT_COLORS


# =============================================================================
# Custom Colormap Creation Constants
# =============================================================================

# Mathematical constants for colormap functions
_TANH_SCALE_FACTOR = 2.0
_LOG_BASE_TRANSFORM = 20.0
_POWER_EXPONENT = 7.0 / 3.0
_COLOR_NORMALIZATION = 2.1

# Colormap function parameters
_WPK_RED_POWER = 10.0 / 4.0
_WPK_BLUE_TANH_PARAMS = (10.0 / 13.0, 8.0)
_WPK_GREEN_TANH_PARAMS = (8.0 / 13.0, 5.0)

_KOY_RED_TANH_PARAMS = (-6.0 / 13.0, 6.0)
_KOY_BLUE_TANH_PARAMS = (-11.0 / 13.0, 4.0)

_KGY_RED_TANH_PARAMS = (-8.0 / 13.0, 13.0)
_KGY_GREEN_POLYNOMIAL_COEF = 15.0
_KGY_GREEN_MULTIPLIER = 2.0 / 2.1
_KGY_BLUE_LINEAR_COEF = 1.0 / 10.0
_KGY_BLUE_POLYNOMIAL_COEF = 8.0

_MAGMA_RED_POWER = 4.0
_MAGMA_RED_TANH_PARAMS = (-7.0 / 13.0, 8.0)
_MAGMA_GREEN_LOG_MULTIPLIER = 63.0
_MAGMA_GREEN_LOG_DIVISOR = 6.0
_MAGMA_GREEN_POWER_BASE = 320.0
_MAGMA_BLUE_SIN_FRACTION = 1.0 / 3.0
_MAGMA_BLUE_LINEAR_FRACTION = 2.0 / 3.0


# =============================================================================
# Custom Colormap Functions
# =============================================================================


def _create_wpk_colormap():
    """Create the WPK (White-Pink-Black) colormap."""

    def red_function(x):
        return 1 - (x**_WPK_RED_POWER)

    def blue_function(x):
        offset, scale = _WPK_BLUE_TANH_PARAMS
        return (np.tanh((offset - x) * scale) + 1.0) / _TANH_SCALE_FACTOR

    def green_function(x):
        offset, scale = _WPK_GREEN_TANH_PARAMS
        return (np.tanh((offset - x) * scale) + 1.0) / _TANH_SCALE_FACTOR

    return {
        "red": red_function,
        "blue": blue_function,
        "green": green_function,
    }


def _create_koy_colormap():
    """Create the KOY (Black-Orange-Yellow) colormap."""

    def red_function(x):
        offset, scale = _KOY_RED_TANH_PARAMS
        return (np.tanh((offset + x) * scale) + 1.0) / _TANH_SCALE_FACTOR

    def blue_function(x):
        offset, scale = _KOY_BLUE_TANH_PARAMS
        return (np.tanh((offset + x) * scale) + 1.0) / _TANH_SCALE_FACTOR

    def green_function(x):
        return x**_POWER_EXPONENT

    return {
        "red": red_function,
        "blue": blue_function,
        "green": green_function,
    }


def _create_kgy_colormap():
    """Create the KGY (Black-Green-Yellow) colormap."""

    def red_function(x):
        offset, scale = _KGY_RED_TANH_PARAMS
        return (np.tanh((offset + x) * scale) + 1.0) / _COLOR_NORMALIZATION

    def green_function(x):
        polynomial_term = (
            x * (x - 1) * (x - 0.5) * (x - 0.65) * _KGY_GREEN_POLYNOMIAL_COEF
        )
        return (x + polynomial_term) * _KGY_GREEN_MULTIPLIER

    def blue_function(x):
        linear_term = x * _KGY_BLUE_LINEAR_COEF
        polynomial_term = (
            x * (x - 1) * (x - 0.2) * (x - 0.9) * _KGY_BLUE_POLYNOMIAL_COEF
        )
        return linear_term + polynomial_term

    return {
        "red": red_function,
        "green": green_function,
        "blue": blue_function,
    }


def _create_magma_colormap():
    """Create a custom Magma-like colormap."""

    def red_function(x):
        term1 = (1 - (1 - x) ** _MAGMA_RED_POWER) / 3
        offset, scale = _MAGMA_RED_TANH_PARAMS
        term2 = (np.tanh((offset + x) * scale) + 1.0) / 3
        return term1 + term2

    def green_function(x):
        log_term = (
            np.log2(1 + x * _MAGMA_GREEN_LOG_MULTIPLIER) / _MAGMA_GREEN_LOG_DIVISOR
        ) / _LOG_BASE_TRANSFORM
        power_term = (
            ((_MAGMA_GREEN_POWER_BASE**x) - 1)
            / (_MAGMA_GREEN_POWER_BASE - 1)
            / _LOG_BASE_TRANSFORM
            * 19
        )
        return log_term + power_term

    def blue_function(x):
        sin_term = np.sin(x * np.pi * 2) * _MAGMA_BLUE_SIN_FRACTION
        linear_term = x * _MAGMA_BLUE_LINEAR_FRACTION
        return sin_term + linear_term

    return {
        "red": red_function,
        "green": green_function,
        "blue": blue_function,
    }


# =============================================================================
# Custom Colormap Creation
# =============================================================================

# Initialize colormap variables
wpk: Optional[object] = None
koy: Optional[object] = None
kgy: Optional[object] = None
magma: Optional[object] = None

try:
    import cmap as cm

    # Create WPK colormap
    try:
        _wpk_colormap = cm.Colormap(_create_wpk_colormap(), name="wpk")
        wpk = _wpk_colormap.to_mpl()
    except Exception as e:
        warnings.warn(f"Failed to create WPK colormap: {e}", UserWarning)

    # Create KOY colormap
    try:
        _koy_colormap = cm.Colormap(_create_koy_colormap(), name="koy")
        koy = _koy_colormap.to_mpl()
    except Exception as e:
        warnings.warn(f"Failed to create KOY colormap: {e}", UserWarning)

    # Create KGY colormap
    try:
        _kgy_colormap = cm.Colormap(_create_kgy_colormap(), name="kgy")
        kgy = _kgy_colormap.to_mpl()
    except Exception as e:
        warnings.warn(f"Failed to create KGY colormap: {e}", UserWarning)

    # Create Magma colormap
    try:
        _magma_colormap = cm.Colormap(_create_magma_colormap(), name="magma_custom")
        magma = _magma_colormap.to_mpl()
    except Exception as e:
        warnings.warn(f"Failed to create custom Magma colormap: {e}", UserWarning)

except ImportError:
    warnings.warn(
        "cmap package not available. Custom colormaps will not be created.", UserWarning
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Tableau color palettes
    "TABLEAU_10_COLORS",
    "TABLEAU_2_COLORS",
    "TABLEAU_3_COLORS",
    "TABLEAU_4_COLORS",
    "TABLEAU_4_ALT_COLORS",
    # Backward compatibility
    "tab10",
    "tab2",
    "tab3",
    "tab4",
    "tab4_2",
    # Custom colormaps
    "wpk",
    "koy",
    "kgy",
    "magma",
]
