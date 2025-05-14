from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from packaging.version import Version

from .._utilities import update_config
from .._validate import (
    isiterable,
    validate_groupby,
    validate_keys,
    validate_layer_and_raw,
)
from ..get import obs_categories, obs_data
from ._baseplot import MultiPanelFigure
from ._helper import get_palette, get_scatter_size


@dataclass
class FeatFigure(MultiPanelFigure):
    axhline: Optional[Mapping[str, Iterable[float]]] = None
    axvline: Optional[Mapping[str, Iterable[float]]] = None
    axline_params: Optional[Mapping[str, Any]] = None

    groups: Iterable[str] = field(default_factory=list)
    null_groups: bool = False
    two_groups: bool = False
    main_group: Optional[str] = None
    null_main_group: bool = False
    sub_group: Optional[str] = None
    null_sub_group: bool = True
    main_cats: Iterable[str] = field(default_factory=list)
    sub_cats: Iterable[str] = field(default_factory=list)
    layer: Optional[str] = None
    use_raw: Optional[bool] = None

    titles: Optional[Iterable[str]] = None

    sns_ver_13: bool = True

    def _process_feat_inputs(
        self,
        adata: sc.AnnData,
        groupby: Optional[Union[str, tuple[str, str]]] = None,
        layer: Optional[str] = None,
        use_raw: Optional[bool] = None,
    ) -> None:
        self.groups = (
            [None] if groupby is None else groupby if isiterable(groupby) else [groupby]
        )
        if groupby is not None:
            validate_groupby(adata, self.groups)
        else:
            self.null_groups = True
        self.two_groups = len(self.groups) > 1
        self.main_group = self.groups[0]
        self.null_main_group = self.main_group is None
        self.sub_group = self.groups[1] if self.two_groups else None
        self.null_sub_group = self.sub_group is None
        self.main_cats = (
            [None] if self.null_main_group else obs_categories(adata, self.main_group)
        )
        self.sub_cats = (
            [None] if self.null_sub_group else obs_categories(adata, self.sub_group)
        )

        self.layer, self.use_raw = validate_layer_and_raw(adata, layer, use_raw)

        self.sns_ver_13 = Version(sns.__version__) >= Version("0.13.0")

    def prepare_pb_feat_data(
        self,
        adata: sc.AnnData,
        feats: Iterable[str],
        main_group_key: str,
        sub_group_key: str,
    ) -> tuple[pd.DataFrame, str, str]:
        df = obs_data(
            adata, feats, layer=self.layer, use_raw=self.use_raw, as_series=False
        )
        df[main_group_key] = (
            ""
            if self.null_main_group
            else obs_data(adata, self.main_group, as_series=True)
        )
        g1_name = "" if self.null_main_group else self.main_group
        df[sub_group_key] = (
            ""
            if self.null_sub_group
            else obs_data(adata, self.sub_group, as_series=True)
        )
        g2_name = "" if self.null_sub_group else self.sub_group
        return (df, g1_name, g2_name)

    def get_hard_zero_axis_lim(
        self,
        val: Iterable[float],
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
        clip_zero: bool = True,
    ) -> tuple[float, float]:
        llim, ulim = np.min(val), np.max(val)
        _llim, _ulim = FeatFigure._get_data_lim(ax=ax, which=which)
        axis_lim = FeatFigure._calc_axis_lim(
            (min(llim, _llim), max(ulim, _ulim)),
            axis_pad=self.axis_pad,
            clip_zero=clip_zero,
        )
        return (
            (0.0 if (llim == 0.0 and clip_zero) else axis_lim[0]),
            (0.0 if (ulim == 0.0 and clip_zero) else axis_lim[1]),
        )

    def process_plot_titles(
        self, titles: Optional[Union[bool, str, Iterable[str]]] = True
    ) -> None:
        self.titles = (
            titles
            if isiterable(titles)
            else [titles] * (self._npanels)
            if isinstance(titles, str)
            else None
        )

    def cleanup_ax_legend(self, ax: mpl.axes.Axes, idx: int, title: str):
        if self.two_groups:
            if (
                ((idx % self._ncols) < (self._ncols - 1))
                and (idx != (self._npanels - 1))
                and ax.get_legend() is not None
            ):
                ax.get_legend().remove()
            else:
                self.redo_legend(ax, n=len(self.sub_cats), title=title)

    def create_feat_figtitle(
        self,
        feats: Iterable[str],
        figtitle: Optional[Union[bool, str]] = True,
    ) -> None:
        _title = figtitle if isinstance(figtitle, str) else None
        if isinstance(figtitle, bool):
            _title = ""
            if figtitle is True and not self.null_groups:
                from textwrap import wrap

                _wrap_title = ", ".join(
                    ["-".join(f) if isiterable(f) else f for f in feats]
                )
                _title += "\n".join(wrap(_wrap_title, width=self.title_textwrap_length))
                _title += " by\n"
                _title += " and ".join(self.groups)
        if _title is not None:
            self.fig.suptitle(_title)


@dataclass
class DisFigure(FeatFigure):
    feats: Iterable[str] = field(default_factory=list)

    def process_dis_inputs(
        self,
        adata: sc.AnnData,
        keys: Union[str, Iterable[str]],
        groupby: Optional[Union[str, tuple[str, str]]] = None,
        layer: Optional[str] = None,
        use_raw: Optional[bool] = None,
    ) -> None:
        self.feats = keys if isiterable(keys) else [keys]
        validate_keys(adata, self.feats)
        self._process_feat_inputs(
            adata=adata, groupby=groupby, layer=layer, use_raw=use_raw
        )

    def _dis_seaborn_params_helper(
        self,
        flavor: Literal["violin", "box", "bar"],
        plot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> Mapping[str, Any]:
        n_axis_cats = len(self.main_cats)
        n_color_cats = len(self.sub_cats)

        plot_params = dict(plot_kwargs)

        update_config(["linewidth", "lw"], self.edge_linewidth, plot_params)
        update_config("dodge", self.two_groups, plot_params)
        if flavor == "bar":
            update_config(["edgecolor", "ec"], self.edge_color, plot_params)
            update_config(("errorbar" if self.sns_ver_13 else "ci"), None, plot_params)
        else:
            if self.sns_ver_13:
                update_config(["linecolor", "lc"], self.edge_color, plot_params)
            if flavor == "violin":
                update_config("width", 0.9, plot_params)
                update_config("inner", None, plot_params)
                update_config(
                    "bw_method" if self.sns_ver_13 else "bw", "silverman", plot_params
                )
                update_config(
                    "density_norm" if self.sns_ver_13 else "scale", "width", plot_params
                )
                update_config("gridsize", int(1e2), plot_params)
                update_config("cut", 0.5, plot_params)
                if n_color_cats == 2:
                    update_config("split", True, plot_params)
                if not self.two_groups:
                    update_config("legend", False, plot_params)
            elif flavor == "box":
                if self.sns_ver_13:
                    if not self.two_groups:
                        update_config("legend", False, plot_params)
                else:
                    update_config(
                        "boxprops",
                        dict(edgecolor=self.edge_color),
                        plot_params,
                    )
                update_config(
                    "whiskerprops",
                    dict(
                        linewidth=(self.edge_linewidth * 2),
                        color=self.edge_color,
                        solid_capstyle="butt",
                    ),
                    plot_params,
                )
                update_config(
                    "capprops",
                    dict(linewidth=(self.edge_linewidth * 2), color=self.edge_color),
                    plot_params,
                )
                update_config(
                    "capwidths",
                    max(1e-2, min(0.5, 1.0 / np.sqrt(n_axis_cats * n_color_cats))),
                    plot_params,
                )
                update_config(
                    "flierprops",
                    dict(
                        marker=".",
                        markersize=plt.rcParams["font.size"] / 5,
                        markerfacecolor="gray",
                    ),
                    plot_params,
                )
                update_config(
                    "medianprops",
                    dict(
                        linewidth=(self.edge_linewidth * 2),
                        color=self.edge_color,
                        solid_capstyle="butt",
                    ),
                    plot_params,
                )

        return plot_params


@dataclass
class RelFigure(FeatFigure):
    feat_pairs: Iterable[tuple[str, str]] = field(default_factory=list)

    def process_rel_inputs(
        self,
        adata: sc.AnnData,
        keys: Union[str, Iterable[str]],
        groupby: Optional[Union[str, tuple[str, str]]] = None,
        layer: Optional[str] = None,
        use_raw: Optional[bool] = None,
    ) -> None:
        self.feat_pairs = keys if isiterable(keys[0]) else [keys]
        for fp in self.feat_pairs:
            validate_keys(adata, fp)
        self._process_feat_inputs(
            adata=adata, groupby=groupby, layer=layer, use_raw=use_raw
        )

    def _rel_seaborn_params_help(
        self,
        flavor: Literal["hist", "kde", "scatter"],
        adata: sc.AnnData,
        plot_kwargs: Mapping[str, Any] = MappingProxyType({}),
        scatter_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        is_color_group = self.two_groups
        cell_counts = (
            adata.obs[self.sub_group].value_counts().mean()
            if is_color_group
            else adata.shape[0]
        )

        plot_params = dict(plot_kwargs)
        scatter_params = dict(scatter_kwargs)

        if flavor == "scatter":
            scatter_params.update(plot_params)

        if is_color_group:
            update_config("palette", self.palette, scatter_params)
            update_config("palette", self.palette, plot_params)
        else:
            update_config(
                ["facecolor", "facecolors", "fc", "color", "c"], "black", scatter_params
            )
            update_config("legend", False, plot_params)
            update_config("legend", False, scatter_params)

        # scatter plot
        update_config(
            "size", get_scatter_size(cell_counts, self.figsize), scatter_params
        )
        update_config(
            ["linewidth", "linewidths", "lw"], self.edge_linewidth, scatter_params
        )
        update_config(["edgecolor", "edgecolors", "ec"], "gray", scatter_params)

        # hist or kde plots
        if flavor in ["hist", "kde"]:
            update_config(
                "cmap", None if is_color_group else self.color_map, plot_params
            )
            update_config("fill", True, plot_params)
            # update_config("cbar", True, plot_params)
            if flavor == "hist":
                update_config("stat", "count", plot_params)
                update_config("bins", 25, plot_params)
                update_config("thresh", max(3, cell_counts * (1e-3)), plot_params)
            else:
                update_config("bw_method", "silverman", plot_params)
                update_config("gridsize", int(1e2), plot_params)
                update_config("cut", 0.5, plot_params)
                update_config("thresh", 5e-2, plot_params)

        return scatter_params if flavor == "scatter" else plot_params, scatter_params


def dis(
    adata,
    keys: Union[str, Iterable[str]],
    groupby: Optional[Union[str, tuple[str, str]]] = None,
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    flavor: Literal["violin", "box", "bar"] = "violin",
    ax: Optional[mpl.axes.Axes] = None,
    fig: Optional[mpl.figure.Figure] = None,
    swap_axis: bool = False,
    titles: Optional[Union[str, Iterable[str]]] = None,
    figtitle: Optional[Union[bool, str]] = True,
    # add_strip: bool = False,
    # add_errbar: Optional[bool] = None,
    plot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    # strip_kwargs: Mapping[str, Any] = MappingProxyType({}),
    # errorbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    seaborn_violin_fix: bool = True,
    **kwargs,
) -> Optional[Union[mpl.axes.Axes, Iterable[mpl.axes.Axes]]]:
    params = dict(kwargs)
    update_config("x_rotation", 0 if swap_axis else 90, params)
    update_config("axis_pad", 5e-2, params)
    disfig = DisFigure(**params)
    disfig.process_dis_inputs(
        adata, keys=keys, groupby=groupby, layer=layer, use_raw=use_raw
    )
    disfig.create_fig(disfig.feats, [None], ax=ax, fig=fig)
    disfig.process_plot_titles(titles)
    disfig.get_title_textwrap_length()

    plot_selector = {"violin": sns.violinplot, "box": sns.boxplot, "bar": sns.barplot}
    _plot_func = plot_selector[flavor]
    plot_params = disfig._dis_seaborn_params_helper(flavor, plot_kwargs)

    _which = "x" if swap_axis else "y"
    for i, f in enumerate(disfig.feats):
        df, ag_name, cg_name = disfig.prepare_pb_feat_data(
            adata, feats=[f], main_group_key="a_group", sub_group_key="c_group"
        )
        pal = (
            get_palette(adata, cg_name, palette=disfig.palette, return_dict=False)
            if disfig.two_groups
            else None
            if disfig.null_main_group
            else get_palette(adata, ag_name, palette=disfig.palette, return_dict=False)
        )
        cur_ax = disfig.get_ax(i, 0, return_idx=False)
        plot_args = dict(
            data=df,
            x=f if swap_axis else "a_group",
            y="a_group" if swap_axis else f,
            orient="h" if swap_axis else "v",
            order=None if disfig.null_main_group else disfig.main_cats,
            hue="c_group" if disfig.two_groups else "a_group",
            hue_order=(
                disfig.sub_cats
                if disfig.two_groups
                else None
                if disfig.null_main_group
                else disfig.main_cats
            ),
            palette=pal,
            ax=cur_ax,
        )
        _plot_func(**plot_args, **plot_params)
        # if add_strip:
        #     strip_args = plot_args.copy()
        #     strip_args.update(dict(strip_kwargs))
        #     update_config(["edgecolor", "ec"], disfig.edge_color, strip_args)
        #     update_config(["alpha", "a"], 1 / 3, strip_args)
        #     update_config(["linewidth", "lw"], disfig.edge_linewidth, strip_args)
        #     update_config("jitter", 1.0, strip_args)
        #     sns.stripplot(zorder=1.1, **strip_args)
        # if add_errbar:
        #     pass
        if seaborn_violin_fix and flavor == "violin" and len(plot_kwargs) == 0:
            for j in range(len(cur_ax.collections)):
                cur_ax.collections[j].set_edgecolor(disfig.edge_color)

        _set_func = cur_ax.set_ylabel if swap_axis else cur_ax.set_xlabel
        _set_func(ag_name)
        plot_title = disfig.titles[i] if disfig.titles is not None else ""
        cur_ax.set_title(plot_title)
        disfig.set_axis_lim(
            cur_ax,
            which=_which,
            axis_lim=disfig.get_hard_zero_axis_lim(
                df[f], ax=cur_ax, which=_which, clip_zero=True
            ),
            force=True,
        )
        disfig.set_axis_tickloc(cur_ax, which=_which)
        if disfig.null_main_group:
            _set_func = cur_ax.set_yticks if swap_axis else cur_ax.set_xticks
            _set_func([])
        disfig.redo_xy_ticks(cur_ax)
        disfig.cleanup_ax_legend(cur_ax, idx=i, title=cg_name)

    disfig.create_feat_figtitle(disfig.feats, figtitle=figtitle)

    del df
    disfig.cleanup_shared_axis(sharex=(not swap_axis), sharey=swap_axis)
    disfig.cleanup()
    return disfig.save_or_show(flavor)


def rel(
    adata,
    keys: Union[tuple[str, str], Iterable[tuple[str, str]]],
    groupby: Optional[Union[str, tuple[str, str]]] = None,
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    flavor: Literal["hist", "kde", "scatter"] = "hist",
    titles: Optional[Union[bool, str, Iterable[str]]] = True,
    figtitle: Optional[Union[bool, str]] = True,
    ax: Optional[mpl.axes.Axes] = None,
    fig: Optional[mpl.figure.Figure] = None,
    plot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    scatter_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs,
) -> Optional[mpl.figure.Figure]:
    import warnings

    params = dict(kwargs)
    update_config("x_rotation", 0, params)
    update_config("axis_pad", 5e-2, params)
    update_config("color_map", "viridis", params)
    relfig = RelFigure(**params)
    relfig.process_rel_inputs(
        adata, keys=keys, groupby=groupby, layer=layer, use_raw=use_raw
    )
    relfig.create_fig(relfig.feat_pairs, relfig.main_cats, ax=ax, fig=fig)
    relfig.process_plot_titles(titles)
    relfig.get_title_textwrap_length()

    plot_selector = {
        "hist": sns.histplot,
        "kde": sns.kdeplot,
        "scatter": sns.scatterplot,
    }
    _plot_func = plot_selector[flavor]
    plot_params, scatter_params = relfig._rel_seaborn_params_help(
        flavor, adata, plot_kwargs=plot_kwargs, scatter_kwargs=scatter_kwargs
    )

    for i, fp in enumerate(relfig.feat_pairs):
        df, pg_name, cg_name = relfig.prepare_pb_feat_data(
            adata,
            feats=fp,
            main_group_key="p_group",
            sub_group_key="c_group",
        )
        for j, c in enumerate(relfig.main_cats):
            cur_ax, cur_idx = relfig.get_ax(i, j, return_idx=True)
            _df = df if c is None else df.loc[df["p_group"] == c]
            common_params = dict(
                data=_df,
                x=fp[0],
                y=fp[1],
                hue="c_group" if relfig.two_groups else None,
                hue_order=relfig.sub_cats if relfig.two_groups else None,
                ax=cur_ax,
            )
            sns.scatterplot(
                **common_params,
                **scatter_params,
            )
            if flavor != "scatter":
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=".*cmap parameter ignored.*"
                    )
                    _plot_func(**common_params, **plot_params)

            plot_title = relfig.titles[cur_idx] if relfig.titles is not None else None
            if isinstance(titles, bool):
                plot_title = ""
                if titles is True:
                    if c is not None:
                        plot_title += c
                        if relfig.two_groups:
                            plot_title += f" by {cg_name}"
                        if fp is not None:
                            plot_title += " : "
                    if fp is not None:
                        plot_title += " - ".join(fp)
            cur_ax.set_title(plot_title)
            cur_ax.set_xlabel(fp[0])
            cur_ax.set_ylabel(fp[1])
            if j > 0:
                relfig.redo_xy_lim(cur_ax, clip_zero=False)
            else:
                relfig.set_xy_lim(cur_ax, clip_zero=False, force=True)
            relfig.set_xy_tickloc(cur_ax)
            relfig.redo_xy_ticks(cur_ax)
            relfig.cleanup_ax_legend(cur_ax, idx=cur_idx, title=cg_name)

    relfig.create_feat_figtitle(relfig.feat_pairs, figtitle=figtitle)

    del df, _df
    relfig.cleanup()
    return relfig.save_or_show(flavor)
