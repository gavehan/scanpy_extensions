from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from .._utilities import update_config
from .._validate import (
    isiterable,
    validate_groupby,
    validate_keys,
    validate_layer_and_raw,
)
from ..get import get_categories, get_obs_data
from ._baseplot import TEXT_SEP, MultiPanelFigure


def _scale_avgs(
    avgs_df: pd.DataFrame,
    scale_method: Literal["minmax", "max"] = "max",
) -> pd.Series:
    """Scale average expression values."""
    # Group by variable to get min/max values
    _avg_gby = avgs_df[["variable", "avg"]].groupby(["variable"])
    min_dict = _avg_gby.min().squeeze().to_dict()
    max_dict = _avg_gby.max().squeeze().to_dict()

    # Scale each value according to the selected method
    data = {}
    for i, row in avgs_df.iterrows():
        var, val = row["variable"], row["avg"]
        _min, _max = min_dict[var], max_dict[var]
        _bot = (_max - _min) if scale_method == "minmax" else _max
        _top = (val - _min) if scale_method == "minmax" else val
        data[i] = 0 if (_bot == 0) else (_top / _bot)

    return pd.Series(data)


@dataclass
class AggrFigure(MultiPanelFigure):
    """Configuration class for plot settings."""

    _MAX_LEGEND_HEIGHT_RATIO = 0.95
    _LEGEND_HEIGHT_BASE = 10.0

    # Data configuration
    feats: Iterable[str] = field(default_factory=list)
    _feats: Iterable[str] = field(default_factory=list)
    groups: Iterable[str] = field(default_factory=list)
    keys_is_map: bool = False
    groupby_kind: Literal["single", "conjugate"] = "single"
    gk_is_conj: bool = False
    flavor: Literal["dot", "matrix"] = "dot"
    annotated: bool = False

    # Plot assitstance configuration
    main_cats: Iterable[str] = field(default_factory=list)
    conj_cats: Optional[Iterable[str]] = None
    n_annot: int = 0
    swap_axis: bool = False
    legend_axd: Optional[mpl.figure.Figure] = None
    main_ax: Optional[mpl.figure.Figure] = None
    annot_ax: Optional[mpl.figure.Figure] = None
    cnorm: Optional[mpl.colors.Normalize] = None

    # Dot plot settings
    dot_power: float = 1.5
    dot_size: float = field(default_factory=lambda: plt.rcParams["font.size"] * 8)

    # Rotation settings
    z_rotation: float = 0

    # Legend settings
    legend_color_title: str = "Scaled\navg. exp."
    legend_size_title: str = "Non-zero\nfraction"
    legend_size_include_zero: bool = False
    legend_length: Optional[float] = None
    legend_title_pad: float = 0.75

    # Size settings
    z_length: Optional[float] = None

    def process_aggr_inputs(
        self,
        adata: sc.AnnData,
        keys: Union[str, Iterable[str], Mapping[str, Any]],
        groupby: Union[str, Tuple[str, str]],
        groupby_kind: Literal["single", "conjugate"],
        flavor: Literal["dot", "matrix"],
        swap_axis: bool,
    ) -> None:
        from itertools import chain

        """Process and validate input parameters."""

        # Process grouping input
        gk_is_conj = groupby_kind == "conjugate"
        groups = list(groupby) if isiterable(groupby) else [groupby]
        validate_groupby(adata, groups)

        # Process features input
        keys_is_map = isinstance(keys, Mapping)
        feats = (
            list(chain.from_iterable(keys.values()))
            if keys_is_map
            else list(keys)
            if isiterable(keys)
            else [keys]
        )
        validate_keys(adata, feats)

        # Validate configuration
        if gk_is_conj and keys_is_map:
            raise ValueError(
                "Invalid configuration, mapped key is only compatible with kind 'single'"
            )
        if len(groups) not in [1, 2]:
            raise ValueError("Invalid configuration, groupby must be of length 2 or 1")
        if (len(groups) == 2) and not gk_is_conj:
            raise ValueError(
                "Invalid configuration for kind 'single', groupby must be of length 1"
            )

        self.flavor = flavor
        self.z_rotation = 0 if swap_axis else 90
        self.axis_pad = 0.0 if flavor == "matrix" else 2.5e-2
        self.swap_axis = swap_axis

        self.feats = feats
        self.groups = groups
        self.keys_is_map = keys_is_map
        self.groupby_kind = groupby_kind
        self.gk_is_conj = gk_is_conj
        self.annotated = self.keys_is_map or self.gk_is_conj

    @staticmethod
    def _concat_str_series(
        df: pd.DataFrame,
        x_key: Union[str, Iterable[str]],
        y_key: Union[str, Iterable[str]],
        text_sep: str = TEXT_SEP,
    ) -> pd.Series:
        return df[x_key].astype(str).str.cat(df[y_key].astype(str), sep=text_sep)

    def prepare_aggr_data(
        self,
        adata: sc.AnnData,
        layer: Optional[str] = None,
        use_raw: Optional[bool] = None,
        undo_log: bool = True,
        scale_method: Literal["minmax", "max"] = "max",
    ) -> pd.DataFrame:
        from collections import Counter

        df = get_obs_data(
            adata,
            self.feats + self.groups,
            layer=layer,
            use_raw=use_raw,
            as_series=False,
        )

        counter = Counter(self.feats)
        self._feats = []
        for f in self.feats:
            if counter[f] > 1:
                counter[f] -= 1
                _f = f"{f}{TEXT_SEP}{counter[f]}"
            else:
                _f = f
            self._feats.append(_f)
        df.columns = self._feats + self.groups

        if undo_log:
            df[self._feats] = df[self._feats].transform(np.expm1)
        df[[f"{c}_b" for c in self._feats]] = df[self._feats] > 0.0
        df = df.groupby(self.groups, observed=False, dropna=False).mean().reset_index()
        if undo_log:
            df[self._feats] = df[self._feats].transform(np.log1p)
        df.iloc[:, len(self.groups) :] = df.iloc[:, len(self.groups) :].fillna(0.0)

        avg_df = df.iloc[:, : -len(self._feats)].melt(id_vars=self.groups)
        avg_df.index = AggrFigure._concat_str_series(
            df=avg_df, x_key="variable", y_key=self.groups
        )

        if self.flavor == "dot":
            bin_cols = list(range(len(self.groups))) + [
                (i + len(self.groups) + len(self._feats))
                for i in range(len(self._feats))
            ]
            pct_df = df.iloc[:, bin_cols].melt(id_vars=self.groups)
            pct_df["variable"] = pct_df["variable"].map(lambda x: str(x)[:-2])
            pct_df.index = AggrFigure._concat_str_series(
                df=pct_df, x_key="variable", y_key=self.groups
            )

        data = (
            pd.merge(
                avg_df.rename({"value": "avg"}, axis=1),
                pct_df.rename({"value": "pct"}, axis=1)["pct"],
                left_index=True,
                right_index=True,
            )
            if self.flavor == "dot"
            else avg_df.rename({"value": "avg"}, axis=1)
        )
        data.index.name = None
        data["scaled_avg"] = _scale_avgs(data, scale_method)

        if self.flavor == "dot":
            data["pct_dot"] = (
                (data["pct"] ** self.dot_power) / (data["pct"].max() ** self.dot_power)
            ) * self.dot_size

        return data

    @staticmethod
    def _cleanup_ax(ax: mpl.axes.Axes) -> None:
        ax.patch.set_alpha(0.0)

    @staticmethod
    def _get_max_label_length(lab: Iterable[str]) -> int:
        return max([len(x) for x in lab])

    def create_aggr_fig(
        self,
        adata: sc.AnnData,
        data: pd.DataFrame,
        keys: Union[str, Iterable[str], Mapping[str, Any]],
        fig: Optional[mpl.figure.Figure] = None,
    ) -> None:
        """Prepare plot dimensions and figure."""
        # Set categories for groups
        self.main_cats = list(get_categories(adata, self.groups[0]))
        self.conj_cats = (
            list(get_categories(adata, self.groups[1])) if self.gk_is_conj else None
        )
        x_n = len(self.feats)
        y_n = len(self.main_cats)
        x_lab_len = AggrFigure._get_max_label_length(self.feats)
        y_lab_len = AggrFigure._get_max_label_length(self.main_cats)

        # Handle conjugate mode
        if self.gk_is_conj:
            data["conj_var"] = AggrFigure._concat_str_series(
                df=data, x_key="variable", y_key=self.groups[1]
            )
            x_n = len(self.feats) * len(self.conj_cats)
            x_lab_len = AggrFigure._get_max_label_length(self.conj_cats)

        # Calculate z-dimension size for annotations
        x_annot = 0
        if self.annotated:
            z_lab_len = AggrFigure._get_max_label_length(
                keys if self.keys_is_map else self.feats
            )
            _z_len = (
                (max(0.1, (np.sqrt(z_lab_len) / 20)) / (2 if self.swap_axis else 0.8))
                if self.z_length is None
                else self.z_length
            )
            x_annot = _z_len * y_n

        # Swap dimensions if needed
        if self.swap_axis:
            x_n, y_n = y_n, x_n
            x_lab_len, y_lab_len = y_lab_len, x_lab_len

        # Calculate legend size
        l_base_size = max(2.0, np.log10(1.0 + x_n))
        l_n = (
            l_base_size + (0 if self.swap_axis else 0.5)
            if self.legend_length is None
            else self.legend_length * x_n
        )

        x_main = x_n + (y_lab_len / 4)
        y_main = y_n + (x_lab_len / 4)
        x_legend = l_n

        x_full = (
            x_main + x_legend + (x_annot if (self.annotated and self.swap_axis) else 0)
        )
        y_full = y_main + (x_annot if (self.annotated and not self.swap_axis) else 0)

        # Create figure if not provided
        if fig is None:
            # Use provided figsize or calculate based on content
            if self.fixed_figsize is not None:
                fw, fh = self.fixed_figsize
            else:
                def_w, def_h = self.figsize
                w_div = 9 if self.swap_axis else 12
                h_div = 12 if self.swap_axis else 9
                fw = def_w * x_full / w_div
                fh = def_h * y_full / h_div

            self.fig = plt.figure(figsize=(fw, fh))
        else:
            self.fig = fig

        # Create subfigures for main plot and legend
        sfigs = self.fig.subfigures(
            nrows=1, ncols=2, width_ratios=(x_full - x_legend, x_legend)
        )

        l_top_pad = (
            (x_annot + (z_lab_len / 4)) if self.annotated and not self.swap_axis else 1
        ) / y_full
        l_top_pad = max(
            l_top_pad,
            (1 - (self._LEGEND_HEIGHT_BASE + np.log10(y_n)) / y_n),
        )
        l_mid_pad = 0.1 / y_full
        l_bot_pad = (1 + (x_lab_len / 4)) / y_full
        l_h = 1 - (l_top_pad + l_mid_pad + l_bot_pad)
        self.legend_axd = sfigs[1].subplot_mosaic(
            ".\nS\n.\nC\n.",
            height_ratios=(
                l_top_pad,
                l_h * 0.6,
                l_mid_pad,
                l_h * 0.4,
                l_bot_pad,
            ),
        )

        if self.annotated:
            if self.swap_axis:
                self.annot_ax, self.main_ax = sfigs[0].subplots(
                    nrows=1, ncols=2, width_ratios=(x_annot, x_main), sharey=True
                )
            else:
                self.annot_ax, self.main_ax = sfigs[0].subplots(
                    nrows=2, ncols=1, height_ratios=(x_annot, x_main), sharex=True
                )
        else:
            self.main_ax = sfigs[0].subplots()

    def plot_legend(self, data: pd.DataFrame) -> None:
        """Plot the legend section."""
        # Configure label parameters
        label_params = dict(
            fontsize=plt.rcParams["legend.title_fontsize"],
            loc="left",
            linespacing=self.linespacing,
            labelpad=(self.legend_title_pad * plt.rcParams["font.size"]),
            clip_on=False,
        )
        ytick_params = dict(
            which="y",
            text_loc="y_r",
            text_rotation=0,
            fontsize=plt.rcParams["font.size"],
        )

        # Create color normalization
        cur_ax = self.legend_axd["C"]

        _min, _max = data["scaled_avg"].min(), data["scaled_avg"].max()
        self.cnorm = mpl.colors.Normalize(_min, _max)

        # Add colorbar
        cur_ax.get_figure().colorbar(
            mpl.cm.ScalarMappable(norm=self.cnorm, cmap=self.color_map),
            cax=cur_ax,
            label="",
        )

        # Set colorbar label and ticks
        cur_ax.set_xlabel(self.legend_color_title, **label_params)
        self.set_axis_lim(
            cur_ax, which="y", axis_lim=(_min, _max), clip_zero=False, force=True
        )
        self.set_axis_tickloc(cur_ax, which="y", max_n_ticks=4)
        self.redo_axis_ticks(cur_ax, **ytick_params)

        # Add size legend for dot plots
        cur_ax = self.legend_axd["S"]
        if self.flavor == "dot":
            _min, _max = data["pct"].min(), data["pct"].max()
            # Calculate tick values for size legend
            s_ticks = AggrFigure.create_axis_tickloc(
                n_ticks=self.n_ticks, max_n_ticks=5
            ).tick_values(_min, _max)
            s_ticks = [t for t in s_ticks if (t <= _max)]

            # Remove zero if configured
            if not self.legend_size_include_zero and s_ticks[0] < 1e-6:
                s_ticks = s_ticks[1:].copy()

            # Calculate number of decimal places needed
            decimals = min(sorted([len(str(t).split(".")[-1]) for t in s_ticks])[-2], 3)
            _s_ticks = np.round(s_ticks, decimals)

            # Calculate size values proportionally
            sizes = (
                (_s_ticks**self.dot_power)
                / (_s_ticks.max() ** self.dot_power)
                * self.dot_size
            )

            # Plot size legend
            cur_ax.scatter(
                [0] * len(_s_ticks),
                _s_ticks,
                s=sizes,
                c="lightgray",
                edgecolors=self.edge_color,
                linewidths=self.edge_linewidth,
                clip_on=False,
            )

            # Configure size legend appearance
            cur_ax.set_xlabel(self.legend_size_title, **label_params)
            cur_ax.set(ylabel=None)
            cur_ax.set_xlim(auto=True)
            self.set_axis_lim(cur_ax, which="y", clip_zero=True, force=True)
            cur_ax.set_xticks([])
            self.set_axis_ticks(
                cur_ax, ticks=(_s_ticks, [str(t) for t in _s_ticks]), **ytick_params
            )
            cur_ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
        # Hide size legend for matrix plots
        elif self.flavor == "matrix":
            cur_ax.axis("off")

        # Ensure tight layout for all legend elements
        for a in self.legend_axd.keys():
            AggrFigure._cleanup_ax(self.legend_axd[a])

    def _get_aggr_axis_lim(
        self, ax: mpl.axes.Axes, which: Literal["x", "y"] = "x"
    ) -> Tuple[float, float]:
        axis_lim = AggrFigure._get_data_lim(ax=ax, which=which)
        # axis_lim = ax.get_xlim() if which == "x" else ax.get_ylim()
        if self.flavor == "dot":
            axis_lim = (axis_lim[0] - 0.5, axis_lim[1] + 0.5)
        _axis_lim = AggrFigure._calc_axis_lim(
            axis_lim=axis_lim, axis_pad=self.axis_pad, clip_zero=False
        )
        return _axis_lim

    def _get_aggr_tick_pos(self, tick_labs: Iterable[str]) -> Iterable[float]:
        return [
            (i + 0.5 if self.flavor == "matrix" else i) for i in range(len(tick_labs))
        ]

    def plot_main(self, data: pd.DataFrame) -> None:
        """Plot the main visualization."""
        # Common plot settings
        plot_kwargs = {
            "cmap": self.color_map,
            "linewidths": self.edge_linewidth,
            "norm": self.cnorm,
        }

        # Determine which columns to use for x and y axis
        x_col = "conj_var" if self.gk_is_conj else "variable"
        y_col = self.groups[0]
        if self.swap_axis:
            x_col, y_col = y_col, x_col

        # Create appropriate plot type
        cur_ax = self.main_ax
        if self.flavor == "dot":
            # Create scatter plot (dot plot)
            cur_ax.scatter(
                data=data,
                x=x_col,
                y=y_col,
                c="scaled_avg",
                s="pct_dot",
                edgecolors=self.edge_color,
                **plot_kwargs,
            )
        elif self.flavor == "matrix":
            # Create pivot table with proper ordering
            pivot_df = data.pivot(
                columns=x_col,
                index=y_col,
                values="scaled_avg",
            ).loc[data[y_col].unique(), data[x_col].unique()]

            # Create heatmap
            sns.heatmap(
                pivot_df,
                xticklabels=True,
                yticklabels=True,
                cbar=False,
                ax=cur_ax,
                linecolor=self.edge_color,
                **plot_kwargs,
            )

        # Remove axis labels
        cur_ax.set(xlabel=None, ylabel=None)

        # Prepare tick labels
        x_tick_labels = (
            self.conj_cats * len(self.feats) if self.gk_is_conj else self.feats
        )
        y_tick_labels = self.main_cats
        if self.swap_axis:
            x_tick_labels, y_tick_labels = y_tick_labels, x_tick_labels

        # Calculate tick positions
        cur_ax.set(xlabel=None, ylabel=None)
        self.set_xy_lim(
            cur_ax,
            xaxis_lim=self._get_aggr_axis_lim(cur_ax, which="x"),
            yaxis_lim=self._get_aggr_axis_lim(cur_ax, which="y"),
            clip_zero=False,
            force=True,
        )
        self.set_axis_ticks(
            cur_ax,
            ticks=(self._get_aggr_tick_pos(x_tick_labels), x_tick_labels),
            which="x",
            text_loc="x_b",
        )
        self.set_axis_ticks(
            cur_ax,
            ticks=(self._get_aggr_tick_pos(y_tick_labels), y_tick_labels),
            which="y",
            text_loc="y_r" if (self.annotated and self.swap_axis) else "y_l",
        )
        cur_ax.invert_yaxis()

        AggrFigure._cleanup_ax(cur_ax)

    def _plot_annot_helper(
        self, ax: mpl.axes.Axes, cum_sum: int, count: int, label: str, **kwargs
    ) -> None:
        """Helper function for annotation plotting."""
        lower_offset = 0.3 if self.flavor == "dot" else -0.3
        upper_offset = 0.7 if self.flavor == "dot" else 0.3
        xs = [
            cum_sum - lower_offset,
            cum_sum - lower_offset,
            cum_sum + count - upper_offset,
            cum_sum + count - upper_offset,
        ]
        ys = [0, 1, 1, 0]

        if self.swap_axis:
            xs, ys = ys, xs

        ax.add_artist(
            mpl.lines.Line2D(
                xs,
                ys,
                color=self.edge_color,
                linewidth=self.edge_linewidth,
            )
        )

        next_pos = (cum_sum - lower_offset + cum_sum + count - upper_offset) / 2
        ax.text(
            x=(1.5 if self.swap_axis else next_pos),
            y=(next_pos if self.swap_axis else 2.0),
            s=label,
            **kwargs,
        )

    def plot_annot(
        self,
        keys: Union[str, Iterable[str], Mapping[str, Any]],
    ) -> None:
        cur_ax = self.annot_ax
        cum_sum = 0
        annot_map = (
            keys if self.keys_is_map else {x: self.conj_cats for x in self.feats}
        )

        label_params = self.def_tick_params.copy()
        label_params.update(
            AggrFigure._handle_text_rot(
                rotation=self.z_rotation, text_loc=("y_l" if self.swap_axis else "x_t")
            )
        )
        for annot_label, annot_items in annot_map.items():
            self._plot_annot_helper(
                cur_ax,
                cum_sum=cum_sum,
                count=len(annot_items),
                label=annot_label,
                **label_params,
            )
            cum_sum += len(annot_items)

        if self.swap_axis:
            cur_ax.set_xlim(auto=True)
            cur_ax.invert_xaxis()
            cur_ax.set_xticks([])
            cur_ax.tick_params(
                axis="y", left=False, right=False, labelleft=False, labelright=False
            )
        else:
            cur_ax.set_ylim(auto=True)
            cur_ax.set_yticks([])
            cur_ax.tick_params(
                axis="x", bottom=False, top=False, labelbottom=False, labeltop=False
            )

        cur_ax.grid(False)
        cur_ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
        AggrFigure._cleanup_ax(cur_ax)


def aggr(
    adata: sc.AnnData,
    keys: Union[str, Iterable[str], Mapping[str, Any]],
    groupby: Union[str, Tuple[str, str]],
    groupby_kind: Literal["single", "conjugate"] = "single",
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    undo_log: bool = True,
    scale_method: Literal["minmax", "max"] = "max",
    flavor: Literal["dot", "matrix"] = "dot",
    swap_axis: bool = False,
    fig: Optional[mpl.figure.Figure] = None,
    **kwargs,
) -> Optional[Union[mpl.axes.Axes, Iterable[mpl.axes.Axes]]]:
    params = dict(kwargs)
    update_config("n_ticks", 2, params)
    afig = AggrFigure(**params)
    afig.process_aggr_inputs(
        adata,
        keys=keys,
        groupby=groupby,
        groupby_kind=groupby_kind,
        flavor=flavor,
        swap_axis=swap_axis,
    )
    _layer, _use_raw = validate_layer_and_raw(adata, layer, use_raw)
    data = afig.prepare_aggr_data(
        adata,
        layer=_layer,
        use_raw=_use_raw,
        undo_log=undo_log,
        scale_method=scale_method,
    )
    afig.create_aggr_fig(adata, data, keys, fig=fig)
    afig.plot_legend(data)
    afig.plot_main(data)
    if afig.annotated:
        afig.plot_annot(keys)

    del data
    return afig.save_or_show(flavor)
