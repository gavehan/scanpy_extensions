from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Optional, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from .._utilities import update_config
from .._validate import (
    isiterable,
    validate_groupby,
    validate_keys,
)
from ..get import obs_categories, obs_data
from ._feat_vis import DisFigure, FeatFigure, RelFigure
from ._helper import get_palette


@dataclass
class PBFeatFigure(FeatFigure):
    pb_group: Optional[str] = None
    pb_cats: Iterable[str] = field(default_factory=list)

    def _process_pb_feat_inputs(
        self,
        adata: sc.AnnData,
        groupby: Optional[Union[str, tuple[str, str]]] = None,
        pb_group: Optional[str] = None,
        layer: Optional[str] = None,
        use_raw: Optional[bool] = None,
    ) -> None:
        self.pb_group = pb_group
        validate_groupby(adata, self.pb_group)
        self.pb_cats = obs_categories(adata, self.pb_group)

        self._process_feat_inputs(
            adata=adata, groupby=groupby, layer=layer, use_raw=use_raw
        )

    def prepare_pb_feat_data(
        self,
        adata: sc.AnnData,
        feats: Iterable[str],
        main_group_key: str,
        sub_group_key: str,
        undo_log: Optional[Iterable[bool]] = None,
    ) -> tuple[pd.DataFrame, str, str]:
        _undo_log = (
            [f not in adata.obs.columns for f in feats]
            if undo_log is None
            else undo_log
        )
        df = obs_data(
            adata,
            feats + [self.pb_group],
            layer=self.layer,
            use_raw=self.use_raw,
            as_series=False,
        )
        for i, f in enumerate(feats):
            if _undo_log[i]:
                df[f] = np.expm1(df[f])
        df = df.groupby(self.pb_group, observed=False).mean().fillna(0.0)
        df = df.loc[[c for c in self.pb_cats if c in df.index]]
        for i, f in enumerate(feats):
            if _undo_log[i]:
                df[f] = np.log1p(df[f])

        df[main_group_key] = ""
        if not self.null_main_group:
            g_map = dict(zip(adata.obs[self.pb_group], adata.obs[self.main_group]))
            df[main_group_key] = df.index.map(g_map).astype("category")
            df[main_group_key] = df[main_group_key].cat.set_categories(
                self.main_cats, rename=True
            )
        g1_name = "" if self.null_main_group else self.main_group

        df[sub_group_key] = ""
        if not self.null_sub_group:
            g_map = g_map = dict(
                zip(adata.obs[self.pb_group], adata.obs[self.sub_group])
            )
            df[sub_group_key] = df.index.map(g_map).astype("category")
            df[sub_group_key] = df[sub_group_key].cat.set_categories(
                self.sub_cats, rename=True
            )
        g2_name = "" if self.null_sub_group else self.sub_group

        return (df, g1_name, g2_name)

    def create_pb_feat_figtitle(
        self,
        feats: Iterable[str],
        figtitle: Optional[Union[bool, str]] = True,
    ) -> None:
        _title = figtitle if isinstance(figtitle, str) else None
        if isinstance(figtitle, bool):
            _title = ""
            if figtitle is True and not self.null_groups:
                from textwrap import wrap

                _title += f"{self.pb_group} avg. of\n"
                _wrap_title = ", ".join(
                    ["-".join(f) if isiterable(f) else f for f in feats]
                )
                _title += "\n".join(wrap(_wrap_title, width=self.title_text_wrap_width))
                _title += " by\n"
                _title += " and ".join(self.groups)
        if _title is not None:
            self.fig.suptitle(_title)


@dataclass
class PBDisFigure(PBFeatFigure, DisFigure):
    def process_pb_dis_inputs(
        self,
        adata: sc.AnnData,
        keys: Union[str, Iterable[str]],
        groupby: Optional[Union[str, tuple[str, str]]] = None,
        pb_group: str = "sample",
        layer: Optional[str] = None,
        use_raw: Optional[bool] = None,
        undo_log: Optional[Iterable[bool]] = None,
    ) -> None:
        self.feats = keys if isiterable(keys) else [keys]
        if undo_log is not None:
            assert len(undo_log) == len(self.feats), (
                "'undo_log' length does not match 'keys' length"
            )
        validate_keys(adata, self.feats)
        return super()._process_pb_feat_inputs(
            adata, groupby=groupby, pb_group=pb_group, layer=layer, use_raw=use_raw
        )


@dataclass
class PBRelFigure(PBFeatFigure, RelFigure):
    def process_pb_rel_inputs(
        self,
        adata: sc.AnnData,
        keys: Union[str, Iterable[str]],
        groupby: Optional[Union[str, tuple[str, str]]] = None,
        pb_group: str = "sample",
        layer: Optional[str] = None,
        use_raw: Optional[bool] = None,
        undo_log: Optional[Iterable[bool]] = None,
    ) -> None:
        self.feat_pairs = keys if isiterable(keys[0]) else [keys]
        for fp in self.feat_pairs:
            validate_keys(adata, fp)
        if undo_log is not None:
            assert len(undo_log) == len(self.feats), (
                "'undo_log' length does not match 'keys' length"
            )
        return super()._process_pb_feat_inputs(
            adata, groupby=groupby, pb_group=pb_group, layer=layer, use_raw=use_raw
        )


def pb_dis(
    adata,
    keys: Union[str, Iterable[str]],
    groupby: Optional[Union[str, tuple[str, str]]] = None,
    pb_group: str = "sample",
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    undo_log: Optional[Iterable[bool]] = None,
    flavor: Literal["violin", "box", "bar"] = "violin",
    ax: Optional[mpl.axes.Axes] = None,
    fig: Optional[mpl.figure.Figure] = None,
    swap_axis: bool = False,
    titles: Optional[Union[str, Iterable[str]]] = None,
    figtitle: Optional[Union[bool, str]] = True,
    plot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    seaborn_violin_fix: bool = True,
    **kwargs,
) -> Optional[Union[mpl.axes.Axes, Iterable[mpl.axes.Axes]]]:
    params = dict(kwargs)
    update_config("x_rotation", 0 if swap_axis else 90, params)
    update_config("axis_pad", 0.1, params)
    disfig = PBDisFigure(**params)
    disfig.process_pb_dis_inputs(
        adata,
        keys=keys,
        groupby=groupby,
        layer=layer,
        pb_group=pb_group,
        use_raw=use_raw,
        undo_log=undo_log,
    )
    disfig.create_fig(disfig.feats, [None], ax=ax, fig=fig)
    disfig.process_plot_titles(titles)
    disfig.get_title_text_wrap_width()

    plot_selector = {"violin": sns.violinplot, "box": sns.boxplot, "bar": sns.barplot}
    _plot_func = plot_selector[flavor]
    plot_params = disfig._dis_seaborn_params_helper(flavor, plot_kwargs)

    _which = "x" if swap_axis else "y"
    for i, f in enumerate(disfig.feats):
        df, ag_name, cg_name = disfig.prepare_pb_feat_data(
            adata,
            feats=[f],
            main_group_key="a_group",
            sub_group_key="c_group",
            undo_log=None if undo_log is None else undo_log[i],
        )
        pal = (
            get_palette(adata, cg_name, palette=disfig.palette, as_dict=False)
            if disfig.two_groups
            else None
            if disfig.null_main_group
            else get_palette(adata, ag_name, palette=disfig.palette, as_dict=False)
        )
        cur_ax = disfig.get_ax(i, 0, return_idx=False)
        _plot_func(
            data=df,
            x=f if swap_axis else "a_group",
            y="a_group" if swap_axis else f,
            orient="h" if swap_axis else "v",
            order=None if disfig.null_main_group else disfig.main_cats,
            hue="c_group" if disfig.two_groups else None,
            hue_order=disfig.sub_cats if disfig.two_groups else None,
            palette=pal,
            ax=cur_ax,
            **plot_params,
        )
        if seaborn_violin_fix and flavor == "violin" and len(plot_kwargs) == 0:
            for j in range(len(cur_ax.collections)):
                cur_ax.collections[j].set_edgecolor(disfig.edge_color)

        _set_func = cur_ax.set_ylabel if swap_axis else cur_ax.set_xlabel
        _set_func(ag_name)
        plot_title = disfig.titles[i] if disfig.titles is not None else ""
        cur_ax.set_title(plot_title)
        disfig.set_axis_limits(
            cur_ax,
            which=_which,
            axis_lim=disfig.get_hard_zero_axis_limits(
                df[f], ax=cur_ax, which=_which, clip_zero=True
            ),
            force=True,
        )
        disfig.set_axis_tickloc(cur_ax, which=_which)
        if disfig.null_main_group:
            _set_func = cur_ax.set_yticks if swap_axis else cur_ax.set_xticks
            _set_func([])
        disfig.update_xy_ticks(cur_ax)
        disfig.cleanup_ax_legend(cur_ax, idx=i, title=cg_name)

    disfig.create_pb_feat_figtitle(disfig.feats, figtitle=figtitle)

    del df
    disfig.cleanup_shared_axis(sharex=(not swap_axis), sharey=swap_axis)
    disfig.cleanup()
    return disfig.save_or_show(flavor)


def pb_rel(
    adata,
    keys: Union[tuple[str, str], Iterable[tuple[str, str]]],
    groupby: Optional[Union[str, tuple[str, str]]] = None,
    pb_group: str = "sample",
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    undo_log: Optional[Iterable[tuple[bool, bool]]] = None,
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
    update_config("axis_pad", 0.1, params)
    update_config("color_map", "viridis", params)
    relfig = PBRelFigure(**params)
    relfig.process_pb_rel_inputs(
        adata,
        keys=keys,
        groupby=groupby,
        layer=layer,
        pb_group=pb_group,
        use_raw=use_raw,
        undo_log=undo_log,
    )
    relfig.create_fig(relfig.feat_pairs, relfig.main_cats, ax=ax, fig=fig)
    relfig.process_plot_titles(titles)
    relfig.get_title_text_wrap_width()

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
            undo_log=None if undo_log is None else undo_log[i],
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
                relfig.update_xy_limits(cur_ax, clip_zero=False)
            else:
                x_lim = relfig.get_hard_zero_axis_limits(
                    df[fp[0]], ax=cur_ax, which="x", clip_zero=False
                )
                y_lim = relfig.get_hard_zero_axis_limits(
                    df[fp[1]], ax=cur_ax, which="y", clip_zero=False
                )
                relfig.set_xy_limits(
                    cur_ax,
                    xaxis_lim=x_lim,
                    yaxis_lim=y_lim,
                    clip_zero=False,
                    force=True,
                )
            relfig.set_xy_tickloc(cur_ax)
            relfig.update_xy_ticks(cur_ax)
            relfig.cleanup_ax_legend(cur_ax, idx=cur_idx, title=cg_name)

    relfig.create_pb_feat_figtitle(relfig.feat_pairs, figtitle=figtitle)

    del df, _df
    relfig.cleanup()
    return relfig.save_or_show(flavor)
