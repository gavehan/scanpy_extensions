"""Plotting configuration parameters for scanpy extensions.

This module provides configuration dictionaries
for matplotlib and seaborn styling.
"""

import matplotlib.pyplot as plt

# Layout padding constants
CONSTRAINED_LAYOUT_PAD_ARTICLE = 1 / 24
CONSTRAINED_LAYOUT_PAD_POSTER = 1 / 48
SUBPLOT_SPACING_TIGHT = 1e-2

# Font and size constants
ARTICLE_BASE_FONTSIZE = 6.0
POSTER_BASE_FONTSIZE = 18.0
DEFAULT_DPI_ARTICLE = 150
DEFAULT_DPI_POSTER = 60
SAVE_DPI = 400

# Color constants
GRID_COLOR = "#ababab"
LEGEND_EDGE_COLOR = "#ababab"

# Font families
SANS_SERIF_FONTS_ARTICLE = ["Helvetica", "Liberation Sans", "DejaVu Sans"]
SANS_SERIF_FONTS_POSTER = ["DejaVu Sans"]

# Figure size constants
ARTICLE_FIGURE_SIZE = (1.75, 1.75)
POSTER_FIGURE_SIZE = (3.84, 3.84)


def get_default_font_family():
    """Get default font family from matplotlib."""
    return plt.rcParams["font.family"]


def get_default_figure_dpi():
    """Get default figure DPI from matplotlib."""
    return plt.rcParams["figure.dpi"]


# Base matplotlib parameters for article style
BASE_MPL_PARAMS = {
    # Lines
    "lines.linewidth": 0.5,
    "lines.markeredgecolor": "none",
    "lines.markeredgewidth": 0.0,
    # Patches
    "patch.linewidth": 0.5,
    # Font settings
    "font.family": "sans-serif",
    "font.sans-serif": SANS_SERIF_FONTS_ARTICLE,
    "font.size": ARTICLE_BASE_FONTSIZE,
    # Math text
    "mathtext.fontset": "dejavuserif",
    "mathtext.default": "regular",
    # Axes
    "axes.linewidth": 0.5,
    "axes.titlesize": 7.5,
    "axes.titleweight": "normal",
    "axes.titlepad": 4.0,
    "axes.labelsize": 7.5,
    "axes.labelpad": 2.0,
    "axes.labelweight": "normal",
    "axes.axisbelow": True,
    "axes.xmargin": 2.5e-2,
    "axes.ymargin": 2.5e-2,
    # Ticks
    "xtick.major.size": 2.0,
    "xtick.major.width": 0.5,
    "xtick.major.pad": 2.0,
    "xtick.labelsize": 6.0,
    "ytick.major.size": 2.0,
    "ytick.major.width": 0.5,
    "ytick.major.pad": 2.0,
    "ytick.labelsize": 6.0,
    # Grid
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.5,
    # Legend
    "legend.edgecolor": LEGEND_EDGE_COLOR,
    "legend.markerscale": 0.5,
    "legend.fontsize": 6.0,
    "legend.title_fontsize": 6.0,
    "legend.borderpad": 1 / 3,
    "legend.handlelength": 1.0,
    "legend.handleheight": 0.5,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.5,
    # Figure
    "figure.titlesize": 9.0,
    "figure.titleweight": "bold",
    "figure.labelsize": 7.5,
    "figure.labelweight": "normal",
    "figure.figsize": ARTICLE_FIGURE_SIZE,
    "figure.subplot.left": 0.05,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.05,
    "figure.subplot.top": 0.95,
    "figure.subplot.wspace": SUBPLOT_SPACING_TIGHT,
    "figure.subplot.hspace": SUBPLOT_SPACING_TIGHT,
    # Constrained layout
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": CONSTRAINED_LAYOUT_PAD_ARTICLE,
    "figure.constrained_layout.w_pad": CONSTRAINED_LAYOUT_PAD_ARTICLE,
    "figure.constrained_layout.hspace": SUBPLOT_SPACING_TIGHT,
    "figure.constrained_layout.wspace": SUBPLOT_SPACING_TIGHT,
    # Scatter plots
    "scatter.edgecolors": "none",
    # Saving
    "savefig.bbox": "tight",
    "savefig.pad_inches": 1 / 18,
    "savefig.transparent": True,
    "ps.fonttype": 42,
    "ps.useafm": False,
    "pdf.fonttype": 42,
    "pdf.use14corefonts": False,
    "pdf.inheritcolor": False,
}

# Article style matplotlib parameters (default)
ARTICLE_MPL_PARAMS = BASE_MPL_PARAMS.copy()

# Poster style matplotlib parameters - extends base with poster-specific values
POSTER_MPL_PARAMS = BASE_MPL_PARAMS.copy()
POSTER_MPL_PARAMS.update(
    {
        # Font
        "font.sans-serif": SANS_SERIF_FONTS_POSTER,
        "font.stretch": "condensed",
        "font.size": POSTER_BASE_FONTSIZE,
        # Axes
        "axes.titlesize": 18.0,
        "axes.titleweight": "regular",
        "axes.titlepad": 7.5,
        "axes.labelsize": 18.0,
        "axes.labelpad": 4.5,
        "axes.linewidth": 2.0,
        # Ticks
        "xtick.labelsize": 16.0,
        "xtick.major.pad": 2.0,
        "xtick.minor.pad": 2.0,
        "xtick.major.size": 4.0,
        "xtick.minor.size": 4.0,
        "xtick.major.width": 1.25,
        "xtick.minor.width": 1.25,
        "ytick.labelsize": 16.0,
        "ytick.major.pad": 2.0,
        "ytick.minor.pad": 2.0,
        "ytick.major.size": 4.0,
        "ytick.minor.size": 4.0,
        "ytick.major.width": 1.25,
        "ytick.minor.width": 1.25,
        # Figure
        "figure.figsize": POSTER_FIGURE_SIZE,
        "figure.titlesize": 24.0,
        "figure.titleweight": "bold",
        "figure.labelsize": 20.0,
        "figure.dpi": DEFAULT_DPI_POSTER,
        # Lines and patches for poster
        "patch.linewidth": 2.0,
        "grid.linewidth": 1.25,
        "lines.linewidth": 2.0,
        # Legend for poster
        "legend.fontsize": 16.0,
        "legend.title_fontsize": 16.0,
        # Constrained layout for poster
        "figure.constrained_layout.h_pad": CONSTRAINED_LAYOUT_PAD_POSTER,
        "figure.constrained_layout.w_pad": CONSTRAINED_LAYOUT_PAD_POSTER,
        # Saving for poster
        "savefig.dpi": SAVE_DPI,
        "savefig.pad_inches": 0.1,
    }
)

# Seaborn style parameters
SEABORN_STYLE_PARAMS = {
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.labelcolor": "black",
    "figure.facecolor": "white",
    "grid.color": GRID_COLOR,
    "grid.linestyle": "-",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.bottom": True,
    "xtick.top": False,
    "ytick.left": True,
    "ytick.right": False,
    "patch.edgecolor": "black",
    "patch.force_edgecolor": False,
    "image.cmap": "viridis",
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
}

# Layout parameters for constrained layout
ARTICLE_LAYOUT_PARAMS = {
    "w_pad": CONSTRAINED_LAYOUT_PAD_ARTICLE,
    "h_pad": CONSTRAINED_LAYOUT_PAD_ARTICLE,
    "wspace": SUBPLOT_SPACING_TIGHT,
    "hspace": SUBPLOT_SPACING_TIGHT,
}

POSTER_LAYOUT_PARAMS = {
    "w_pad": CONSTRAINED_LAYOUT_PAD_POSTER,
    "h_pad": CONSTRAINED_LAYOUT_PAD_POSTER,
    "wspace": SUBPLOT_SPACING_TIGHT,
    "hspace": SUBPLOT_SPACING_TIGHT,
}

# Legacy naming for backwards compatibility
# TODO: Deprecate these in favor of the new names
mpl_params = ARTICLE_MPL_PARAMS
mpl_poster_params = POSTER_MPL_PARAMS
sns_params = SEABORN_STYLE_PARAMS
layout_params = ARTICLE_LAYOUT_PARAMS
poster_layout_params = POSTER_LAYOUT_PARAMS


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ARTICLE_MPL_PARAMS",
    "POSTER_MPL_PARAMS",
    "SEABORN_STYLE_PARAMS",
    "ARTICLE_LAYOUT_PARAMS",
    "POSTER_LAYOUT_PARAMS",
    # Backward compatibility
    "mpl_params",
    "mpl_poster_params",
    "sns_params",
    "layout_params",
    "poster_layout_params",
]
