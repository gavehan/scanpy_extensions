from collections.abc import Iterable
from typing import Any, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc

from ..get import isiterable, obs_categories

PVAL_THRESHOLDS = [1e-3, 1e-2, 5e-2]


def get_figsize(x_scale: float = 1, y_scale: float = 1) -> tuple[float, float]:
    _fs = plt.rcParams["figure.figsize"]
    return (_fs[0] * x_scale, _fs[1] * y_scale)


def format_pval(pval: float, return_star: bool = True) -> str:
    ret_str = ""
    if return_star:
        for t in PVAL_THRESHOLDS:
            if pval <= t:
                ret_str += "\u204e"
        if len(ret_str) < 1:
            ret_str = "ns"
        return ret_str
    else:
        return f"{pval:.2e}"


def get_palette(
    adata: sc.AnnData,
    key: str,
    palette: Optional[Union[str, Iterable[str], mpl.colors.Colormap]] = None,
    force: bool = True,
    return_dict: bool = False,
) -> Union[dict[str, Any], list[Any]]:
    _palette = (
        palette
        if palette is not None
        else plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    cats = [None] if key is None else obs_categories(adata, key)
    colors = None
    if f"{key}_colors" in adata.uns.keys() and (not force or palette is None):
        colors = adata.uns[f"{key}_colors"]
    else:
        _cmap = (
            mpl.colormaps[_palette]
            if isinstance(_palette, str)
            else _palette
            if isinstance(_palette, mpl.colors.Colormap)
            else None
        )
        _colors = _palette if (_cmap is None and isiterable(_palette)) else None
        if _colors is None:
            _colors = (
                [_cmap(i) for i in range(_cmap.N)]
                if isinstance(_cmap, mpl.colors.ListedColormap)
                else [_cmap(i / (len(cats) - 1)) for i in range(len(cats))]
            )
        cmap = mpl.colors.ListedColormap(_colors, len(cats))
        colors = [mpl.colors.to_hex(cmap(i)) for i in range(len(cats))]
        adata.uns[f"{key}_colors"] = colors
    if return_dict:
        return dict(zip(cats, colors))
    else:
        return colors


def get_marker_size(
    cell_count: int, figsize: Optional[tuple[float, float]] = None, scale: float = 1.0
) -> float:
    from numpy import sqrt

    _fs = plt.rcParams["figure.figsize"] if figsize is None else figsize
    space = _fs[0] * _fs[1]
    return (
        max(
            (plt.rcParams["font.size"] / space),
            ((plt.rcParams["font.size"] * 10 * space) / sqrt(cell_count)),
        )
        * scale
    )
