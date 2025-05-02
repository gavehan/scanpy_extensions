from numbers import Real
from typing import Literal, Optional

import matplotlib as mpl
import scanpy as sc
import seaborn as sns

from ._comp_vis import comp_bar, div_comp_bar
from ._emb_vis import annot_emb, emb
from ._feat_aggr import aggr
from ._feat_vis import dis, rel
from ._helper import format_pval, get_figsize, get_palette, get_scatter_size
from ._pb_feat_vis import pb_dis, pb_rel
from ._volcano import volcano

# from ._feat_aggr import aggr
from .colors import tab2, tab3, tab4, tab4_2, tab10
from .config import mpl_params, mpl_poster_params

try:
    import cmap

    from .config import kgy, koy, magma, wpk
except ImportError:
    pass


def configure(
    dpi: Optional[Real] = None, type: Literal["article", "poster"] = "article"
) -> None:
    from .config import mpl_params, mpl_poster_params, sns_params

    _mpl_params = mpl_poster_params if type == "poster" else mpl_params
    _dpi = dpi if dpi is not None else 72 if type == "poster" else 150
    sc.settings.set_figure_params(
        dpi=_dpi,
        dpi_save=300,
        vector_friendly=True,
        fontsize=_mpl_params["font.size"],
        format="pdf",
        facecolor="white",
        transparent=True,
    )
    mpl.rcParams.update(**_mpl_params)
    sns.set_style(style=sns_params, rc=_mpl_params)
    sns.set_context(context=_mpl_params, rc=_mpl_params)


__all__ = [
    "configure",
    "tab2",
    "tab3",
    "tab4",
    "tab4_2",
    "tab10",
    "kgy",
    "koy",
    "magma",
    "wpk",
    "mpl_params",
    "mpl_poster_params",
    "annot_emb",
    "emb",
    "aggr",
    "comp_bar",
    "div_comp_bar",
    "dis",
    "rel",
    "get_scatter_size",
    "format_pval",
    "get_palette",
    "volcano",
    "pb_dis",
    "pb_rel",
    "get_figsize",
]
