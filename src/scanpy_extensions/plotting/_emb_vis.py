from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from .._utilities import update_config
from .._validate import (
    isiterable,
    validate_groupby,
    validate_keys,
    validate_layer_and_raw,
)
from ..get import obs_categories, obs_data
from ._baseplot import MultiPanelFigure
from ._helper import get_marker_size, get_palette


@dataclass
class EmbFigure(MultiPanelFigure):
    feats: Iterable[str] = field(default_factory=list)
    null_feats: bool = False
    groupby: Optional[str] = None
    gb_cats: Optional[Iterable[str]] = None
    layer: Optional[str] = None
    use_raw: Optional[bool] = None

    smooth: bool = False
    ng = None

    feat_pal: Optional[Union[str, mpl.colors.Colormap]] = None
    feat_is_num: bool = False
    smooth_undo_log: bool = False

    size: Optional[float] = None
    basis: Optional[str] = None
    others_color: str = "lightgray"
    shuffle_order: bool = True

    def get_emb_size(self, adata: sc.AnnData, scale: float = 1.0) -> None:
        if self.size is None:
            cell_count = (
                adata.shape[0]
                if self.groupby is None
                else adata.obs[self.groupby].value_counts().mean()
            )
            self.size = get_marker_size(cell_count, figsize=self.figsize, scale=scale)

    @staticmethod
    def _process_basis(adata: sc.AnnData, basis: Optional[str] = None) -> str:
        _basis = [basis] if basis is not None else ["umap", "tsne", "pca"]
        for b in _basis:
            if b in adata.obsm.keys():
                return b
            elif f"X_{b}" in adata.obsm.keys():
                return f"X_{b}"
        raise KeyError(f"Key {b} not in .obsm.")

    def process_emb_inputs(
        self,
        adata: sc.AnnData,
        keys: Optional[Union[str, Iterable[str]]] = None,
        groupby: Optional[str] = None,
        basis: Optional[str] = None,
        layer: Optional[str] = None,
        use_raw: Optional[bool] = None,
        smooth: bool = False,
        obsp_key: Optional[str] = None,
    ) -> None:
        self.null_feats = keys is None
        self.feats = [None] if self.null_feats else keys if isiterable(keys) else [keys]
        if not self.null_feats:
            validate_keys(adata, self.feats, check_numeric=False)
        if groupby is not None:
            self.groupby = groupby
            validate_groupby(adata, [self.groupby])
            self.gb_cats = obs_categories(adata, self.groupby)
        else:
            self.gb_cats = [None]
        self.basis = EmbFigure._process_basis(adata, basis)

        self.layer, self.use_raw = validate_layer_and_raw(adata, layer, use_raw)

        self.smooth = smooth
        # _obsp_key = "connectivities" if obsp_key is None else obsp_key
        _obsp_key = f"{self.basis}_connectivities" if obsp_key is None else obsp_key
        if smooth and (_obsp_key not in adata.obsp.keys()):
            from scanpy import logging as logg

            from ..preprocessing._neighbors import sklearn_neighbors

            _nn = max(10, int(np.round(np.log10(adata.shape[0]))))
            logg.warning(
                f"`{self.basis}_connectivities` not present in .obsp and neighbors with n_neighbors={_nn} will be calculated."
            )
            sklearn_neighbors(
                adata,
                n_neighbors=_nn,
                n_pcs=2,
                use_rep=self.basis,
                key_added=self.basis,
            )

        self.ng = adata.obsp[_obsp_key] if smooth else None

    def prepare_emb_data(self, adata: sc.AnnData) -> sc.AnnData:
        from scipy.sparse import csr_matrix

        _adata = sc.AnnData(csr_matrix((adata.shape[0], 1), dtype=np.int8))
        _adata.obs_names = adata.obs_names.copy()
        _adata.obsm[self.basis] = adata.obsm[self.basis].copy()
        for k in self.feats:
            if k is None:
                continue
            _adata.obs[k] = obs_data(
                adata, k, layer=self.layer, use_raw=self.use_raw, as_series=True
            )
        if self.groupby is not None:
            _adata.obs[self.groupby] = obs_data(adata, self.groupby, as_series=True)
        saved_colors = [x for x in adata.uns.keys() if "colors" in x]
        for x in saved_colors:
            _adata.uns[x] = list(adata.uns[x])
        obs_idx = (
            adata.obs.sample(
                frac=1.0, replace=False, random_state=self.random_state
            ).index
            if self.shuffle_order
            else _adata.obs_names
        )
        if self.smooth and self.shuffle_order:
            _int_idx = [adata.obs_names.get_loc(i) for i in obs_idx]
            self.ng = self.ng[_int_idx, :][:, _int_idx].copy()
        return _adata[obs_idx, :].copy() if self.shuffle_order else _adata

    def get_smoothed_feat(
        self,
        adata: sc.AnnData,
        key: str,
        subset_idx: Optional[pd.Index] = None,
        **kwargs,
    ) -> pd.Series:
        from ..tools._smooth import _smooth_over_graph

        if subset_idx is None:
            return pd.Series(
                _smooth_over_graph(
                    adata.obs[key], self.ng, undo_log=self.smooth_undo_log, **kwargs
                ),
                index=adata.obs_names,
            )
        else:
            return pd.Series(
                _smooth_over_graph(
                    adata.obs.loc[subset_idx, key],
                    self.ng[subset_idx, :][:, subset_idx],
                    undo_log=self.smooth_undo_log,
                    **kwargs,
                ),
                index=adata.obs_names[subset_idx],
            )

    def get_smooth_minmax(
        self, adata: sc.AnnData, key: str, **kwargs
    ) -> tuple[float, float]:
        if self.groupby is None:
            val = self.get_smoothed_feat(adata, key=key, **kwargs)
            return (val.min(), val.max())
        else:
            _min, _max = np.inf, -np.inf
            for g in self.gb_cats:
                _subset_idx = adata.obs[self.groupby] == g
                val = self.get_smoothed_feat(
                    adata, key=key, subset_idx=_subset_idx, **kwargs
                )
                _min = min(_min, np.min(val))
                _max = max(_max, np.max(val))
            return (_min, _max)

    def prepare_plot_data(
        self,
        plot_adata: sc.AnnData,
        og_adata: sc.AnnData,
        key: str,
        undo_log: Optional[bool] = None,
        plot_params: Mapping[str, Any] = MappingProxyType({}),
        smooth_params: Mapping[str, Any] = MappingProxyType({}),
    ) -> tuple[pd.Series, Mapping[str, Any]]:
        _plot_params = dict(plot_params)
        _plot_data = None
        if key is None:
            update_config("legend_loc", None, _plot_params)
            if self.palette is None:
                update_config("palette", ["black"], _plot_params)
            _plot_data = pd.Series(
                [""] * plot_adata.shape[0], index=plot_adata.obs_names
            )
        else:
            _plot_data = obs_data(
                plot_adata,
                keys=key,
                layer=self.layer,
                use_raw=self.use_raw,
                as_series=True,
            )
            self.feat_is_num = validate_keys(
                plot_adata,
                keys=key,
                check_numeric=False,
            )[0]
            self.smooth_undo_log = (
                (key not in og_adata.obs.columns) if undo_log is None else undo_log
            )
            if self.feat_is_num:
                _min, _max = (
                    self.get_smooth_minmax(plot_adata, key=key, **smooth_params)
                    if self.smooth
                    else (plot_adata.obs[key].min(), plot_adata.obs[key].max())
                )
                update_config("vmin", _min, _plot_params)
                update_config("vmax", _max, _plot_params)
            else:
                self.feat_pal = get_palette(og_adata, key=key, palette=self.palette)

        return _plot_data, _plot_params

    def plot_emb(
        self,
        adata: sc.AnnData,
        key: str,
        ax: mpl.axes.Axes,
        title: Optional[str] = None,
        **kwargs,
    ) -> mpl.axes.Axes:
        import warnings

        from scanpy.pl import embedding

        params = dict(kwargs)
        update_config("show", False, params)
        update_config("frameon", True, params)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*No data for colormapping provided via.*"
            )
            embedding(
                adata,
                color=key,
                basis=self.basis[2:] if self.basis[:2] == "X_" else self.basis,
                ax=ax,
                title=title,
                **params,
            )
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        _params = dict(
            ms=1.0 + mpl.rcParams["axes.linewidth"] * 6.5,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.plot(1, 0, ">k", **_params)
        ax.plot(0, 1, "^k", **_params)

    def create_emb_title(
        self,
        adata: sc.AnnData,
        title: Optional[Union[bool, str]] = True,
    ) -> str:
        _title = title if isinstance(title, str) else None
        if isinstance(title, bool):
            _title = ""
            if title is True:
                from textwrap import wrap

                _feats = (
                    None
                    if self.null_feats
                    else [k for k in self.feats if k is not None]
                )
                if _feats is not None and len(_feats) > 0 and self.groupby is not None:
                    _title += ", ".join(_feats)
                    _title += " by "
                if self.groupby is not None:
                    _title += self.groupby + "\n"
                    _wrap_title = ", ".join(
                        [
                            f"{x}={i:,}"
                            for (x, i) in adata.obs[self.groupby]
                            .value_counts()[self.gb_cats]
                            .items()
                        ]
                    )
                    _title += "\n".join(
                        wrap(_wrap_title, width=self.title_textwrap_length)
                    )
        return _title


def emb(
    adata: sc.AnnData,
    keys: Optional[Union[str, Iterable[str]]] = None,
    groupby: Optional[str] = None,
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    sort_order: bool = True,
    smooth: bool = False,
    obsp_key: Optional[str] = None,
    undo_log: Optional[bool] = None,
    basis: Optional[str] = None,
    titles: Optional[Union[bool, str, Iterable[str]]] = True,
    ax: Optional[mpl.axes.Axes] = None,
    fig: Optional[mpl.figure.Figure] = None,
    figtitle: Optional[Union[bool, str]] = True,
    vcenter: Optional[float] = None,
    plot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    smooth_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs,
) -> Optional[Union[mpl.axes.Axes, Iterable[mpl.axes.Axes]]]:
    params = dict(kwargs)
    update_config("x_rotation", 0, params)
    update_config("y_rotation", 90, params)
    update_config("axis_pad", 5e-2, params)
    efig = EmbFigure(**params)

    efig.process_emb_inputs(
        adata,
        keys=keys,
        groupby=groupby,
        basis=basis,
        layer=layer,
        use_raw=use_raw,
        smooth=smooth,
        obsp_key=obsp_key,
    )
    efig.create_fig(efig.feats, efig.gb_cats, ax=ax, fig=fig)
    efig.get_emb_size(adata)
    efig.get_title_textwrap_length()
    _titles = (
        titles
        if isiterable(titles)
        else [titles] * (efig._npanels)
        if isinstance(titles, str)
        else None
    )
    _adata = efig.prepare_emb_data(adata)

    plot_params = dict(
        na_color=efig.others_color,
        na_in_legend=False,
        color_map=efig.color_map,
        size=efig.size,
        sort_order=sort_order,
        vcenter=vcenter,
    )
    plot_params.update(dict(plot_kwargs))
    smooth_params = dict(scale=False)
    smooth_params.update(dict(smooth_kwargs))
    fkey = "_feat"
    _fkey = "_color"
    for i, f in enumerate(efig.feats):
        _adata.obs[fkey], _plot_params = efig.prepare_plot_data(
            _adata,
            og_adata=adata,
            key=f,
            undo_log=undo_log,
            plot_params=plot_params.copy(),
            smooth_params=smooth_params,
        )
        for j, c in enumerate(efig.gb_cats):
            cur_ax, cur_idx = efig.get_ax(i, j, return_idx=True)
            if c is None:
                _adata.obs[_fkey] = (
                    efig.get_smoothed_feat(_adata, key=fkey, **smooth_params)
                    if (smooth and efig.feat_is_num)
                    else _adata.obs[fkey]
                )
            else:
                _subset_idx = _adata.obs[groupby] == c
                _adata.obs[_fkey] = pd.Series(
                    [np.nan if efig.feat_is_num else pd.NA] * _adata.shape[0],
                    index=_adata.obs_names,
                )
                _val = _adata.obs.loc[_subset_idx, fkey].copy()
                _adata.obs.loc[_subset_idx, _fkey] = (
                    efig.get_smoothed_feat(
                        _adata, key=fkey, subset_idx=_subset_idx, **smooth_params
                    )
                    if (smooth and efig.feat_is_num)
                    else _val
                )
            if efig.feat_pal is not None:
                _adata.uns[f"{_fkey}_colors"] = efig.feat_pal
            plot_title = _titles[cur_idx] if _titles is not None else None
            if isinstance(titles, bool):
                plot_title = ""
                if titles is True:
                    if c is not None:
                        plot_title += c
                        if f is not None:
                            plot_title += " : "
                    if f is not None:
                        plot_title += f
            efig.plot_emb(
                _adata, key=_fkey, ax=cur_ax, title=plot_title, **_plot_params
            )
            if cur_idx > 0:
                efig.update_xy_limits(cur_ax)
            else:
                efig.set_xy_limits(cur_ax)
            efig.update_xy_ticks(cur_ax)
            efig.update_legend(cur_ax)

    _figtitle = efig.create_emb_title(adata, figtitle)
    if _figtitle is not None:
        efig.fig.suptitle(_figtitle)

    del _adata
    efig.cleanup_shared_axis()
    efig.cleanup()
    return efig.save_or_show(efig.basis)


def annot_emb(
    adata: sc.AnnData,
    groupby: Optional[str] = None,
    basis: Optional[str] = None,
    title: Optional[Union[bool, str, Iterable[str]]] = True,
    do_textloc: bool = True,
    label_kwargs: Mapping[str, Any] = MappingProxyType({}),
    textloc_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ax: Optional[mpl.axes.Axes] = None,
    fig: Optional[mpl.figure.Figure] = None,
    plot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs,
) -> Optional[Union[mpl.axes.Axes, Iterable[mpl.axes.Axes]]]:
    from adjustText import adjust_text

    params = dict(kwargs)
    update_config("x_rotation", 0, params)
    update_config("y_rotation", 90, params)
    update_config("axis_pad", 5e-2, params)
    efig = EmbFigure(**params)
    efig.process_emb_inputs(adata, keys=[None], groupby=groupby, basis=basis)
    efig.create_fig([None], [None], ax=ax, fig=fig)
    efig.get_emb_size(adata, scale=(2 / 3))
    efig.get_title_textwrap_length()
    _adata = efig.prepare_emb_data(adata)

    x_pos = adata.obsm[efig.basis][:, 0].ravel()
    y_pos = adata.obsm[efig.basis][:, 1].ravel()

    _label_args = dict(label_kwargs)
    update_config(["verticalalignment", "va"], "center", _label_args)
    update_config(["horizontalalignment", "ha"], "center", _label_args)
    update_config(["color", "c"], "black", _label_args)
    update_config(["fontweight", "weight"], "bold", _label_args)
    update_config(
        ["path_effects"],
        [
            mpl.patheffects.withStroke(
                linewidth=plt.rcParams["lines.linewidth"] * 2, foreground="white"
            )
        ],
        _label_args,
    )

    _textloc_args = dict(textloc_kwargs)
    update_config(["avoid_self"], True, _textloc_args)
    update_config(
        ["explode_radius"],
        np.linalg.norm([np.ptp(x_pos), np.ptp(y_pos)]) / 8,
        _textloc_args,
    )
    update_config(["time_lim"], 2.0, _textloc_args)

    cur_ax = efig.get_ax(0, 0)
    plot_params = dict(
        size=efig.size,
        legend_loc="none",
    )
    plot_params.update(dict(plot_kwargs))
    plot_title = efig.create_emb_title(_adata, title=title)
    _ = get_palette(adata, groupby, efig.palette)
    efig.plot_emb(_adata, key=groupby, ax=cur_ax, title=plot_title, **plot_params)
    efig.update_xy_limits(cur_ax)
    efig.update_xy_ticks(cur_ax)

    # Add on plot annotations
    all_pos = (
        pd.DataFrame(adata.obsm[efig.basis][:, :2], columns=["x", "y"])
        .groupby(adata.obs[groupby].to_numpy(), observed=False)
        .median()
        .sort_index()
    )
    texts = []
    for t, row in all_pos.iterrows():
        texts.append(cur_ax.text(row["x"], row["y"], t, **_label_args))

    if do_textloc:
        if adata.shape[0] > 1e3:
            from geosketch import gs

            _idx = gs(
                adata.obsm[efig.basis], int(1e3), replace=False, seed=efig.random_state
            )
        else:
            _idx = np.arange(adata.shape[0])
        adjust_text(
            texts,
            x=x_pos[_idx],
            y=y_pos[_idx],
            ax=cur_ax,
            **_textloc_args,
        )

    return efig.save_or_show(f"annot_{efig.basis}")
