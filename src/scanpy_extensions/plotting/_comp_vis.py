from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Literal, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scanpy import logging as logg
from scanpy._utils import AnyRandom

from .._utilities import update_config
from .._validate import isiterable, validate_groupby
from ..get import get_categories, get_fractions
from ._baseplot import MultiPanelFigure
from ._helper import format_pval, get_palette


def _get_comp_axis_label(groups: Union[str, tuple[str, str]], norm: bool) -> str:
    axis_lab = "avg. " if len(groups) > 1 else ""
    axis_lab += "fractions (%)" if norm else "cell counts"
    return axis_lab.capitalize()


def get_percents(
    adata: sc.AnnData,
    keys: str,
    groupby: Union[str, tuple[str, str]],
    norm: bool = True,
    totals: Union[Iterable[float], Mapping[str, float]] = None,
    dropna: bool = False,
    return_grouped_fractions: bool = False,
    return_melt: bool = False,
) -> pd.DataFrame:
    df = get_fractions(
        adata,
        keys,
        groupby,
        norm=norm,
        totals=totals,
        dropna=dropna,
        return_grouped_fractions=return_grouped_fractions,
    )
    if return_melt:
        df = pd.melt(df.reset_index(), id_vars=groupby[0]).rename(
            {"value": "frac"}, axis=1
        )
    if norm:
        if return_melt or return_grouped_fractions:
            df["frac"] = (df["frac"] * (1e2)).astype(df["frac"].dtype)
        else:
            df = df * int(1e2)
    return df


def _set_figtitle(
    mpfig: MultiPanelFigure,
    groups: Union[str, tuple[str, str]],
    norm: bool,
    figtitle: Optional[Union[bool, str]],
) -> None:
    _figtitle = figtitle if isinstance(figtitle, str) else None
    if isinstance(figtitle, bool):
        _figtitle = ""
        if figtitle is True:
            _figtitle += "avg. " if len(groups) > 1 else ""
            _figtitle += "fractions" if norm else "cell counts"
            if len(groups) > 1:
                _figtitle += f" of\n{groups[1]} grouped by {groups[0]}"
            else:
                _figtitle += f" by {groups[0]}"
    _figtitle = _figtitle if isinstance(_figtitle, str) else _figtitle
    if _figtitle is not None:
        mpfig.fig.suptitle(_figtitle.capitalize())


def _percentile_interval(data: Iterable[float], width: float) -> tuple[float, float]:
    edge = (100 - width) / 2
    percentiles = edge, 100 - edge
    return tuple(np.nanpercentile(data, percentiles))


def _bootstrap(data: Iterable[float], n_boot: int, seed: AnyRandom = 0, ci: float = 95):
    from seaborn.algorithms import bootstrap

    return _percentile_interval(
        bootstrap(data, n_boot=n_boot, seed=seed),
        ci,
    )


def comp_bar(
    adata: sc.AnnData,
    keys: Union[str, Iterable[str]],
    groupby: Union[str, tuple[str, str]],
    norm: bool = True,
    totals: Union[Iterable[float], Mapping[str, float]] = None,
    flavor: Literal["stacked", "grouped"] = "stacked",
    dropna: bool = False,
    figtitle: Optional[Union[bool, str]] = True,
    ax: Optional[mpl.axes.Axes] = None,
    fig: Optional[mpl.figure.Figure] = None,
    plot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs,
) -> Optional[Union[mpl.axes.Axes, Iterable[mpl.axes.Axes]]]:
    gfeats = keys if isiterable(keys) else [keys]
    groups = groupby if isiterable(groupby) else [groupby]
    validate_groupby(adata, groups + gfeats)

    params = dict(kwargs)
    update_config("sharey", (flavor == "stacked"), params)
    mpfig = MultiPanelFigure(**params)
    mpfig.create_fig(gfeats, [None], ax=ax, fig=fig)

    plot_params = dict(plot_kwargs)
    update_config(["edgecolor", "ec"], mpfig.edge_color, plot_params)
    update_config(["linewidth", "lw"], mpfig.edge_linewidth, plot_params)

    for i, gf in enumerate(gfeats):
        cur_ax = mpfig.get_ax(i, 0)
        df = get_percents(adata, gf, groups, norm=norm, totals=totals, dropna=dropna)

        _ = df.plot(
            ax=cur_ax,
            kind="bar",
            stacked=(flavor == "stacked"),
            color=get_palette(adata, gf, palette=mpfig.palette),
            xlabel=groups[0],
            ylabel=_get_comp_axis_label(groups, norm),
            **plot_params,
        )
        if mpfig.sharey and mpfig.y_lim is None:
            mpfig.set_axis_lim(cur_ax, which="y", clip_zero=True)
            mpfig.set_axis_tickloc(cur_ax, which="y")
            mpfig.set_axis_ticks(cur_ax, which="y")
        mpfig.set_axis_lim(
            cur_ax,
            which="x",
            axis_lim=(-0.5, len(get_categories(adata, groups[0])) - 0.5),
        )
        mpfig.redo_xy_lim(cur_ax)
        mpfig.redo_xy_ticks(cur_ax)
        mpfig.redo_legend(cur_ax)

    _set_figtitle(mpfig, groups, norm, figtitle)

    mpfig.cleanup_shared_axis()
    mpfig.cleanup()
    return mpfig.save_or_show("comp_bar")


def div_comp_bar(
    adata: sc.AnnData,
    keys: Union[str, Iterable[str]],
    groupby: Union[str, tuple[str, str]],
    norm: bool = True,
    totals: Optional[Any] = None,
    dropna: bool = False,
    add_strip: bool = True,
    add_errbar: bool = True,
    add_stats: bool = True,
    stat_flavor: Literal["value", "star"] = "star",
    figtitle: Optional[Union[bool, str]] = True,
    ax: Optional[mpl.axes.Axes] = None,
    fig: Optional[mpl.figure.Figure] = None,
    bar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    strip_kwargs: Mapping[str, Any] = MappingProxyType({}),
    errorbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    n_boot: int = int(1e4),
    ci: float = 95,
    **kwargs,
) -> Optional[Union[mpl.axes.Axes, Iterable[mpl.axes.Axes]]]:
    gfeats = keys if isiterable(keys) else [keys]
    groups = groupby if isiterable(groupby) else [groupby]
    validate_groupby(adata, groups + gfeats)
    g_cats = get_categories(adata, groups[0])
    assert len(g_cats) == 2, f"{groups[0]} does not exactly two categories."

    params = dict(kwargs)
    update_config("x_rotation", 0, params)
    mpfig = MultiPanelFigure(**params)
    mpfig.create_fig(gfeats, [None], ax=ax, fig=fig)

    single_groupby = len(groups) == 1
    g_cats = get_categories(adata, groups[0])
    g_pal = get_palette(adata, groups[0], palette=mpfig.palette)
    for i, gf in enumerate(gfeats):
        cur_ax = mpfig.get_ax(i, 0)
        f_cats = get_categories(adata, gf)
        df = get_percents(
            adata, gf, groups, norm=norm, totals=totals, dropna=dropna, return_melt=True
        )
        df["signed_frac"] = (-2 * (df[groups[0]] == g_cats[0]) + 1) * (df["frac"])

        args = dict(
            x="signed_frac",
            y=gf,
            hue=groups[0],
            order=f_cats,
            orient="h",
            dodge=False,
            ax=cur_ax,
        )

        bar_args = args.copy()
        bar_args.update(dict(bar_kwargs))
        update_config(["edgecolor", "ec"], mpfig.edge_color, bar_args)
        update_config(["linewidth", "lw"], mpfig.edge_linewidth, bar_args)
        sns.barplot(data=df, palette=g_pal, zorder=1, **bar_args)

        if not single_groupby:
            g_df = get_percents(
                adata,
                gf,
                groups,
                norm=norm,
                totals=totals,
                dropna=dropna,
                return_grouped_fractions=True,
            )
            g_df["signed_frac"] = (-2 * (g_df[groups[0]] == g_cats[0]) + 1) * (
                g_df["frac"]
            )

            if add_strip:
                strip_args = args.copy()
                strip_args.update(dict(strip_kwargs))
                update_config(["edgecolor", "ec"], mpfig.edge_color, strip_args)
                update_config(["alpha", "a"], 1 / 3, strip_args)
                update_config(["linewidth", "lw"], mpfig.edge_linewidth, strip_args)
                update_config("jitter", 1.0, strip_args)
                sns.stripplot(data=g_df, palette=g_pal, zorder=1.1, **strip_args)

            if add_errbar:
                ebar_args = dict(errorbar_kwargs)
                update_config(
                    "fmt",
                    "o",
                    ebar_args,
                )
                update_config(
                    ["markersize", "ms"],
                    plt.rcParams["font.size"] / 2,
                    ebar_args,
                )
                update_config(
                    ["markerfacecolor", "mfc"],
                    mpfig.edge_color,
                    ebar_args,
                )
                update_config("ecolor", mpfig.edge_color, ebar_args)
                update_config("elinewidth", mpfig.edge_linewidth * 2, ebar_args)
                update_config(
                    ["markeredgewidth", "mew"],
                    plt.rcParams["font.size"],
                    ebar_args,
                )
                update_config("capsize", plt.rcParams["font.size"] / 2, ebar_args)
                update_config("capthick", mpfig.edge_linewidth, ebar_args)
                update_config("barsabove", True, ebar_args)

                x_pos = []
                y_pos = []
                xerr = []
                for f_i, f_c in enumerate(f_cats):
                    for g_c in g_cats:
                        val = g_df.loc[
                            ((g_df[gf] == f_c) & (g_df[groups[0]] == g_c)), "frac"
                        ]
                        agg = val.mean()
                        err = _bootstrap(
                            val, n_boot=n_boot, seed=mpfig.random_state, ci=ci
                        )
                        x_pos.append(agg * (-2 * (g_c == g_cats[0]) + 1))
                        y_pos.append(f_i)
                        xerr.append(err[::-1] if (g_c == g_cats[0]) else err)

                cur_ax.errorbar(
                    x=x_pos,
                    y=y_pos,
                    xerr=np.array(xerr).transpose(),
                    zorder=3.5,
                    **ebar_args,
                )

            if add_stats:
                from scipy.stats import mannwhitneyu

                val_f_cats = [
                    c
                    for c in f_cats
                    if (len(g_df.loc[(g_df[gf] == c), groups[0]].unique()) == 2)
                ]
                pairs = [((c, g_cats[0]), (c, g_cats[1])) for c in val_f_cats]
                pvals = []
                for pair in pairs:
                    c = pair[0][0]
                    _, p = mannwhitneyu(
                        g_df.loc[
                            (g_df[gf] == c) & (g_df[groups[0]] == pair[0][1]), "frac"
                        ],
                        g_df.loc[
                            (g_df[gf] == c) & (g_df[groups[0]] == pair[1][1]), "frac"
                        ],
                    )
                    pvals.append(p)
                if len(f_cats) < 20:
                    logg.warning(
                        f"MHT correction not performed due to low number of comparisons ({len(f_cats)}) for {gf}"
                    )
                else:
                    from statsmodels.stats import multitest

                    pvals = multitest.multipletests(
                        pvals, alpha=5e-2, method="fdr_bh", is_sorted=False
                    )[0]
                stat_map = dict(zip(val_f_cats, pvals))
                secax_y = cur_ax.secondary_yaxis(
                    "right", functions=(lambda x: x, lambda x: x)
                )
                secax_y.set_ylim(cur_ax.get_ylim())
                ytick_lab = [
                    format_pval(
                        stat_map[c],
                        return_star=(stat_flavor == "star"),
                    )
                    if c in val_f_cats
                    else "n/a"
                    for c in f_cats
                ]
                mpfig.set_axis_ticks(
                    secax_y,
                    which="y",
                    ticks=(list(range(len(f_cats))), ytick_lab),
                    text_loc="y_r",
                    fontfamily=plt.rcParams["font.sans-serif"],
                )
                secax_y.set_ylabel("")

        cur_ax.set_xlabel(_get_comp_axis_label(groups, norm))
        mpfig.redo_axis_lim(cur_ax, which="x", clip_zero=True)
        mpfig.set_axis_tickloc(cur_ax, which="x")
        xticks = mpfig.get_axis_ticks(cur_ax, which="x")
        cur_ax.set_xticks(
            xticks[0],
            [lab.replace("-", "").replace("\u2212", "") for lab in xticks[1]],
        )
        mpfig.redo_xy_ticks(cur_ax)
        mpfig.redo_legend(
            cur_ax, n=2, title="", loc="best", bbox_to_anchor=None, frameon=True
        )

    _set_figtitle(mpfig, groups, norm, figtitle)

    mpfig.cleanup()
    return mpfig.save_or_show("div_comp_bar")
