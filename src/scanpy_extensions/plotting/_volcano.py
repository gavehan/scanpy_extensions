import textwrap
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .._utilities import update_config
from ._baseplot import MultiPanelFigure
from ._helper import get_scatter_size


def volcano(
    dge_df: pd.DataFrame,
    gene_key: str = "names",
    lfc_key: str = "logfoldchanges",
    pval_key: str = "pvals_adj",
    lfc_thres: float = np.log2(1.5),
    pval_thres: float = 5e-2,
    clip_frac: float = 1e-3,
    class_names: tuple[str, str, str] = ["Down reg.", "Not sig.", "Up reg."],
    legend_length: float = 1.0,
    title: Optional[str] = None,
    add_annot: bool = False,
    annot_sort_key: Optional[str] = None,
    annot_n: int = 20,
    textloc_kwargs: Mapping[str, Any] = MappingProxyType({}),
    fig: Optional[mpl.figure.Figure] = None,
    **kwargs,
) -> Optional[Iterable[mpl.axes.Axes]]:
    _annot_sort_key = lfc_key if annot_sort_key is None else annot_sort_key

    sig_key = "is_sig"
    cls_key = "class"
    neg_log_pval_key = "neg_log_pval"

    df = dge_df.copy()
    df[sig_key] = (np.abs(df[lfc_key]) >= lfc_thres) & (df[pval_key] < pval_thres)
    df[cls_key] = df.apply(
        lambda x: "ns" if not x[sig_key] else "up" if x[lfc_key] > 0.0 else "down",
        axis=1,
    ).astype("category")
    df[cls_key] = df[cls_key].cat.reorder_categories(["ns", "down", "up"], ordered=True)
    df[neg_log_pval_key] = -np.log10(df[pval_key])
    if clip_frac > 0.0:
        llim, ulim = df[lfc_key].quantile([clip_frac, 1.0 - clip_frac])
        df[lfc_key] = df[lfc_key].clip(lower=llim, upper=ulim)
        df[neg_log_pval_key] = df[neg_log_pval_key].clip(
            upper=df[neg_log_pval_key].quantile(1.0 - clip_frac)
        )

    params = dict(kwargs)
    update_config("x_rotation", 0, params)
    update_config("palette", ["tab:blue", "tab:grey", "tab:red"], params)
    mpfig = MultiPanelFigure(**params)
    _fig = plt.figure(figsize=mpfig.figsize) if fig is None else fig
    mpfig.axs = _fig.subplots(2, 1, height_ratios=(1, 9))

    textloc_params = dict(textloc_kwargs)
    update_config("textsize", [plt.rcParams["font.size"]] * annot_n, textloc_params)
    update_config("linewidth", mpfig.edge_linewidth, textloc_params)
    update_config("linecolor", mpfig.edge_color, textloc_params)
    update_config("avoid_label_lines_overlap", True, textloc_params)
    update_config("min_distance", 5e-3, textloc_params)
    update_config("max_distance", 0.5, textloc_params)
    update_config("seed", mpfig.random_state, textloc_params)
    update_config("nbr_candidates", int(5e3), textloc_params)
    update_config(
        "path_effects",
        [mpl.patheffects.withStroke(linewidth=mpfig.edge_linewidth, foreground="w")],
        textloc_params,
    )
    update_config(
        "bbox",
        dict(
            alpha=0.8,
            boxstyle="Square,pad=0.05",
            facecolor="w",
            edgecolor="none",
            lw=0.0,
        ),
        textloc_params,
    )

    cur_ax = mpfig.axs[0]
    cur_ax.set_xlim(0.0, 2.5 + (legend_length * 1.5))
    cur_ax.axis("off")
    if title is not None:
        cur_ax.text(0, 0, title, va="center")
    class_counts = df[cls_key].value_counts()
    for i, c in enumerate(["down", "ns", "up"]):
        cur_ax.text(
            i + (legend_length * 1.5),
            0,
            textwrap.indent(f"{class_names[i]}\n({class_counts[c]:,})", "    "),
            va="center",
        )
        cur_ax.plot(
            i + (legend_length * 1.5),
            0,
            color=mpfig.palette[i],
            marker="s",
        )

    cur_ax = mpfig.axs[1]
    sns.scatterplot(
        data=df.sort_values(cls_key),
        x=lfc_key,
        y=neg_log_pval_key,
        linewidths=0.0,
        edgecolor="none",
        s=get_scatter_size(df.shape[0], mpfig.figsize, 0.5),
        hue=cls_key,
        legend=False,
        palette=[mpfig.palette[1], mpfig.palette[0], mpfig.palette[2]],
        rasterized=True,
        ax=cur_ax,
    )
    cur_ax.set_xlabel(r"$Log_2$" + " FC")
    cur_ax.set_ylabel("-" + r"$Log_{10}$" + " Adj. P-value")
    mpfig.set_xy_lim(cur_ax, clip_zero=False)
    mpfig.set_xy_tickloc(cur_ax)
    mpfig.set_xy_ticks(cur_ax)

    if add_annot:
        import textalloc as ta

        _df = df.sort_values(_annot_sort_key).tail(annot_n)
        _df["gene"] = _df.index if gene_key == "index" else _df[gene_key]
        ta.allocate(
            ax=cur_ax,
            x=_df[lfc_key].to_numpy().ravel(),
            y=_df[neg_log_pval_key].to_numpy().ravel(),
            text_list=_df["gene"].to_numpy().ravel(),
            x_scatter=df[lfc_key].to_numpy().ravel(),
            y_scatter=df[neg_log_pval_key].to_numpy().ravel(),
            **textloc_params,
        )

    del df
    return mpfig.save_or_show("volcano")
