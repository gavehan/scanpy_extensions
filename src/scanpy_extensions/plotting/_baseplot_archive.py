from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .._utilities import RandomState, update_config

# Text rotation parameters for different axis positions
TEXT_LOC_PARAMS = {
    "x_b": {
        "bottom": True,
        "top": False,
        "labelbottom": True,
        "labeltop": False,
    },
    "x_t": {
        "bottom": False,
        "top": True,
        "labelbottom": False,
        "labeltop": True,
    },
    "y_l": {
        "left": True,
        "right": False,
        "labelleft": True,
        "labelright": False,
    },
    "y_r": {
        "left": False,
        "right": True,
        "labelleft": False,
        "labelright": True,
    },
}

TEXT_ROT_PARAMS = {
    "x_b": {  # x-axis bottom
        "hz": {"ha": "center", "va": "top"},
        "up": {"ha": "right", "va": "center"},
        "down": {"ha": "left", "va": "center"},
    },
    "x_t": {  # x-axis top
        "hz": {"ha": "center", "va": "bottom"},
        "up": {"ha": "left", "va": "center"},
        "down": {"ha": "right", "va": "center"},
    },
    "y_l": {  # y-axis left
        "hz": {"ha": "right", "va": "center"},
        "up": {"ha": "center", "va": "bottom"},
        "down": {"ha": "center", "va": "top"},
    },
    "y_r": {  # y-axis right
        "hz": {"ha": "left", "va": "center"},
        "up": {"ha": "center", "va": "top"},
        "down": {"ha": "center", "va": "bottom"},
    },
}

TEXT_SEP = "@"

LEGEND_NROWS_MAX = 12


@dataclass
class BaseFigure:
    # Axis settings
    n_ticks: int = 3
    x_rotation: float = 90
    y_rotation: float = 0
    x_lim: Optional[Tuple[float, float]] = None
    y_lim: Optional[Tuple[float, float]] = None
    x_ticks: Optional[Tuple[Iterable[float], Iterable[str]]] = None
    y_ticks: Optional[Tuple[Iterable[float], Iterable[str]]] = None
    axis_pad: float = 2.5e-2
    def_tick_params = dict(
        rotation_mode="anchor",
        bbox=dict(
            alpha=0.0,
            fill=False,
            boxstyle="Square,pad=0.1",
        ),
    )

    # General text formatting
    textwrap_length: int = 30
    linespacing: float = 0.9

    # Visual settings
    color_map: Optional[Union[str, mpl.colors.Colormap]] = None
    palette: Optional[Union[str, mpl.colors.Colormap]] = None
    edge_color: str = "black"
    edge_linewidth: float = field(
        default_factory=lambda: plt.rcParams["patch.linewidth"]
    )

    # Legend settings
    legend_params: Dict[str, Any] = field(
        default_factory=lambda: dict(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=plt.rcParams["legend.fontsize"],
            markerscale=plt.rcParams["legend.markerscale"],
            framealpha=1.0,
            facecolor="inherit",
            edgecolor="inherit",
            title_fontsize=plt.rcParams["legend.fontsize"],
        )
    )

    # Size settings
    figsize: Optional[Tuple[float, float]] = field(
        default_factory=lambda: plt.rcParams["figure.figsize"]
    )
    fixed_figsize: Optional[Tuple[float, float]] = None

    # Random settings
    random_state: RandomState = 0

    def __init__(self, **kwargs):
        params = dict(kwargs)
        if len(params) > 0:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        self.def_tick_params["linespacing"] = self.linespacing

    @staticmethod
    def _calc_axis_lim(
        axis_lim: Tuple[float, float],
        axis_pad: float,
        clip_zero: bool = False,
    ) -> Tuple[float, float]:
        _is_rev = axis_lim[0] > axis_lim[1]
        if axis_pad == 0:
            return axis_lim
        else:
            lpad = (
                0
                if (clip_zero and axis_lim[0] == 0)
                else ((1 if _is_rev else -1) * axis_pad)
            )
            upad = (
                0
                if (clip_zero and axis_lim[1] == 0)
                else ((-1 if _is_rev else 1) * axis_pad)
            )
            return (axis_lim[0] + lpad, axis_lim[1] + upad)

    @staticmethod
    def _get_data_lim(
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
    ) -> Tuple[float, float]:
        data_lim = ax.dataLim.get_points()
        return (
            [data_lim[0][0], data_lim[1][0]]
            if which == "x"
            else [data_lim[0][1], data_lim[1][1]]
        )

    def _create_axis_lim(
        self,
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
        clip_zero: bool = False,
        axis_lim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        _axis_lim = (
            BaseFigure._get_data_lim(ax=ax, which=which)
            if axis_lim is None
            else axis_lim
        )
        _range = np.ptp(_axis_lim)
        return BaseFigure._calc_axis_lim(
            axis_lim=_axis_lim, axis_pad=(_range * self.axis_pad), clip_zero=clip_zero
        )

    def get_axis_lim(
        self,
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
        clip_zero: bool = False,
    ) -> Tuple[float, float]:
        axis_lim = self.x_lim if which == "x" else self.y_lim
        return (
            self._create_axis_lim(ax=ax, which=which, clip_zero=clip_zero)
            if axis_lim is None
            else axis_lim
        )

    def get_xy_lim(
        self,
        ax: mpl.axes.Axes,
        clip_zero: bool = False,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (
            self.get_axis_lim(ax=ax, which="x", clip_zero=clip_zero),
            self.get_axis_lim(ax=ax, which="y", clip_zero=clip_zero),
        )

    def set_axis_lim(
        self,
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
        axis_lim: Optional[Tuple[float, float]] = None,
        clip_zero: bool = False,
        force: bool = False,
    ) -> None:
        _set_func = ax.set_xlim if which == "x" else ax.set_ylim
        _axis_lim = self.x_lim if which == "x" else self.y_lim
        _set_axis_lim = _axis_lim
        if force or _axis_lim is None:
            _set_axis_lim = (
                self._create_axis_lim(ax=ax, which=which, clip_zero=clip_zero)
                if axis_lim is None
                else axis_lim
            )
        if which == "x":
            self.x_lim = _set_axis_lim
        else:
            self.y_lim = _set_axis_lim
        _set_func(_set_axis_lim[0], _set_axis_lim[1])

    def set_xy_lim(
        self,
        ax: mpl.axes.Axes,
        xaxis_lim: Optional[Tuple[float, float]] = None,
        yaxis_lim: Optional[Tuple[float, float]] = None,
        clip_zero: bool = False,
        force: bool = False,
    ) -> None:
        self.set_axis_lim(
            ax=ax, which="x", axis_lim=xaxis_lim, clip_zero=clip_zero, force=force
        )
        self.set_axis_lim(
            ax=ax, which="y", axis_lim=yaxis_lim, clip_zero=clip_zero, force=force
        )

    def redo_axis_lim(
        self,
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
        clip_zero: bool = False,
    ) -> None:
        _set_func = ax.set_xlim if which == "x" else ax.set_ylim
        axis_lim = self.get_axis_lim(ax=ax, which=which, clip_zero=clip_zero)
        _set_func(axis_lim[0], axis_lim[1])

    def redo_xy_lim(
        self,
        ax: mpl.axes.Axes,
        clip_zero: bool = False,
    ) -> None:
        self.redo_axis_lim(ax=ax, which="x", clip_zero=clip_zero)
        self.redo_axis_lim(ax=ax, which="y", clip_zero=clip_zero)

    @staticmethod
    def create_axis_tickloc(
        n_ticks: int,
        max_n_ticks: Optional[int] = None,
        simple_steps: bool = False,
    ) -> mpl.ticker.MaxNLocator:
        _steps = [1, 2, 2.5, 5, 10] if simple_steps else [1, 2, 2.5, 3, 4, 5, 10]
        return mpl.ticker.MaxNLocator(
            nbins="auto" if max_n_ticks is None else max_n_ticks,
            steps=_steps,
            min_n_ticks=n_ticks,
        )

    def set_axis_tickloc(
        self,
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
        n_ticks: Optional[int] = None,
        max_n_ticks: Optional[int] = None,
        simple_steps: bool = False,
        redo_axis_lim: bool = True,
    ) -> None:
        _axis = ax.xaxis if which == "x" else ax.yaxis
        _n_ticks = self.n_ticks if n_ticks is None else n_ticks
        _max_n_ticks = (_n_ticks * 2) if max_n_ticks is None else max_n_ticks
        _axis.set_major_locator(
            BaseFigure.create_axis_tickloc(
                n_ticks=_n_ticks, max_n_ticks=_max_n_ticks, simple_steps=simple_steps
            )
        )
        _axis.reset_ticks()
        if redo_axis_lim:
            self.redo_axis_lim(ax=ax, which=which)
        ax.grid(visible=True, which="major", axis=which)

    def set_xy_tickloc(
        self,
        ax: mpl.axes.Axes,
        n_ticks: Optional[int] = None,
        max_n_ticks: Optional[int] = None,
        simple_steps: bool = False,
        redo_axis_lim: bool = True,
    ) -> None:
        self.set_axis_tickloc(
            ax=ax,
            which="x",
            n_ticks=n_ticks,
            max_n_ticks=max_n_ticks,
            simple_steps=simple_steps,
            redo_axis_lim=redo_axis_lim,
        )
        self.set_axis_tickloc(
            ax=ax,
            which="y",
            n_ticks=n_ticks,
            max_n_ticks=max_n_ticks,
            simple_steps=simple_steps,
            redo_axis_lim=redo_axis_lim,
        )

    @staticmethod
    def _handel_text_loc(
        text_loc: Literal["x_b", "x_t", "y_l", "y_r"],
    ) -> Dict[str, Any]:
        return TEXT_LOC_PARAMS[text_loc]

    @staticmethod
    def _handle_text_rot(
        rotation: float, text_loc: Literal["x_b", "x_t", "y_l", "y_r"]
    ) -> Dict[str, Any]:
        _rot = rotation % 360
        if "x_" in text_loc:
            _rot_type = "down" if _rot >= 180 else "up" if _rot > 0 else "hz"
        else:
            _rot_type = (
                "down"
                if ((_rot / 90) == 3.0)
                else "up"
                if ((_rot / 90) == 1.0)
                else "hz"
            )
        return dict(rotation=rotation, **TEXT_ROT_PARAMS[text_loc][_rot_type])

    @staticmethod
    def _validate_ticks(labels: Iterable[str]) -> bool:
        return sum([(lab is not None and len(lab) > 0) for lab in labels]) > 0

    def _create_axis_ticks(
        self,
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
        text_sep: str = TEXT_SEP,
    ) -> Tuple[Iterable[float], Iterable[str]]:
        from textwrap import wrap

        _get_func = ax.get_xticklabels if which == "x" else ax.get_yticklabels
        ticks = _get_func()
        axis_lim = ax.get_xlim() if which == "x" else ax.get_ylim()
        tick_pos = []
        tick_lab = []
        for i in range(len(ticks)):
            _tick = ticks[i]
            _pos = _tick.get_position()[0 if which == "x" else 1]
            if (_pos < axis_lim[0]) or (_pos > axis_lim[1]):
                continue
            tick_pos.append(_pos)
            tick_lab.append(_tick.get_text())
        wrapped_lab = [
            "\n".join(
                wrap(
                    (lab.split(text_sep)[0] if text_sep in lab else lab),
                    width=self.textwrap_length,
                )
            )
            for lab in tick_lab
        ]
        return (tick_pos, wrapped_lab)

    def get_axis_ticks(
        self,
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
    ) -> Tuple[Iterable[float], Iterable[str]]:
        ticks = self.x_ticks if which == "x" else self.y_ticks
        return self._create_axis_ticks(ax=ax, which=which) if ticks is None else ticks

    def get_xy_ticks(
        self, ax: mpl.axes.Axes
    ) -> Tuple[
        Tuple[Iterable[float], Iterable[str]], Tuple[Iterable[float], Iterable[str]]
    ]:
        return (
            self.get_axis_ticks(ax=ax, which="x"),
            self.get_axis_ticks(ax=ax, which="y"),
        )

    def _set_axis_ticks_helper(
        self,
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
        ticks: Optional[Tuple[Iterable[float], Iterable[str]]] = None,
        text_loc: Optional[Literal["x_b", "x_t", "y_l", "y_r"]] = None,
        text_rotation: Optional[float] = None,
        **kwargs,
    ) -> None:
        tick_params = dict(kwargs)
        for k, v in self.def_tick_params.copy().items():
            update_config(k, v, tick_params)

        _set_func = ax.set_xticks if which == "x" else ax.set_yticks
        _set_func(
            ticks=ticks[0],
            labels=ticks[1],
            minor=False,
            **tick_params,
            **BaseFigure._handle_text_rot(rotation=text_rotation, text_loc=text_loc),
        )
        ax.tick_params(axis=which, **BaseFigure._handel_text_loc(text_loc=text_loc))
        ax.grid(visible=True, which="major", axis=which)

    def set_axis_ticks(
        self,
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
        ticks: Optional[Tuple[Iterable[float], Iterable[str]]] = None,
        text_loc: Optional[Literal["x_b", "x_t", "y_l", "y_r"]] = None,
        text_rotation: Optional[float] = None,
        force: bool = True,
        **kwargs,
    ) -> None:
        _text_loc = ("x_b" if which == "x" else "y_l") if text_loc is None else text_loc
        _text_rot = (
            (self.x_rotation if which == "x" else self.y_rotation)
            if text_rotation is None
            else text_rotation
        )

        _ticks = self.x_ticks if which == "x" else self.y_ticks
        _set_ticks = _ticks
        if force or _ticks is None:
            _set_ticks = (
                self._create_axis_ticks(ax=ax, which=which) if ticks is None else ticks
            )
        if which == "x":
            self.x_ticks = _set_ticks
        else:
            self.y_ticks = _set_ticks

        if BaseFigure._validate_ticks(_set_ticks[1]):
            self._set_axis_ticks_helper(
                ax=ax,
                which=which,
                ticks=_set_ticks,
                text_loc=_text_loc,
                text_rotation=_text_rot,
                **kwargs,
            )

    def set_xy_ticks(
        self,
        ax: mpl.axes.Axes,
        x_ticks: Optional[Tuple[Iterable[float], Iterable[str]]] = None,
        x_text_loc: Optional[Literal["x_b", "x_t", "y_l", "y_r"]] = None,
        x_text_rotation: Optional[float] = None,
        y_ticks: Optional[Tuple[Iterable[float], Iterable[str]]] = None,
        y_text_loc: Optional[Literal["x_b", "x_t", "y_l", "y_r"]] = None,
        y_text_rotation: Optional[float] = None,
        force: bool = True,
        **kwargs,
    ) -> None:
        self.set_axis_ticks(
            ax=ax,
            which="x",
            ticks=x_ticks,
            text_loc=x_text_loc,
            text_rotation=x_text_rotation,
            force=force,
            **kwargs,
        )
        self.set_axis_ticks(
            ax=ax,
            which="y",
            ticks=y_ticks,
            text_loc=y_text_loc,
            text_rotation=y_text_rotation,
            force=force,
            **kwargs,
        )

    def redo_axis_ticks(
        self,
        ax: mpl.axes.Axes,
        which: Literal["x", "y"] = "x",
        text_loc: Optional[Literal["x_b", "x_t", "y_l", "y_r"]] = None,
        text_rotation: Optional[float] = None,
        **kwargs,
    ) -> None:
        _text_loc = ("x_b" if which == "x" else "y_l") if text_loc is None else text_loc
        _text_rot = (
            (self.x_rotation if which == "x" else self.y_rotation)
            if text_rotation is None
            else text_rotation
        )

        ticks = self.get_axis_ticks(ax=ax, which=which)

        if BaseFigure._validate_ticks(ticks[1]):
            self._set_axis_ticks_helper(
                ax=ax,
                which=which,
                ticks=ticks,
                text_loc=_text_loc,
                text_rotation=_text_rot,
                **kwargs,
            )

    def redo_xy_ticks(
        self,
        ax: mpl.axes.Axes,
        x_text_loc: Optional[Literal["x_b", "x_t", "y_l", "y_r"]] = None,
        x_text_rotation: Optional[float] = None,
        y_text_loc: Optional[Literal["x_b", "x_t", "y_l", "y_r"]] = None,
        y_text_rotation: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.set_axis_ticks(
            ax=ax,
            which="x",
            text_loc=x_text_loc,
            text_rotation=x_text_rotation,
            **kwargs,
        )
        self.set_axis_ticks(
            ax=ax,
            which="y",
            text_loc=y_text_loc,
            text_rotation=y_text_rotation,
            **kwargs,
        )

    def redo_legend(
        self,
        ax: mpl.axes.Axes,
        n: Optional[int] = None,
        title: Optional[str] = None,
        **kwargs,
    ) -> None:
        _legend = ax.get_legend()
        if _legend is None:
            return
        _handles, _labels = ax.get_legend_handles_labels()
        if n is not None:
            _handles = _handles[:n]
            _labels = _labels[:n]
        _title = title if title is not None else _legend.get_title().get_text()
        _ncols = min(3, int(np.ceil(len(_handles) / LEGEND_NROWS_MAX)))

        params = dict(kwargs)
        for k, v in self.legend_params.copy().items():
            update_config(k, v, params)

        ax.legend(
            handles=_handles, labels=_labels, ncols=_ncols, title=_title, **params
        )


@dataclass
class MultiPanelFigure(BaseFigure):
    # Title settings
    title_textwrap_length: Optional[int] = None

    # Matplotlib objects
    fig: Optional[mpl.figure.Figure] = None
    axs: Optional[np.ndarray] = None

    # Multi-panel layout
    ncols: Optional[int] = None
    wspace: Optional[float] = 5e-2
    hspace: Optional[float] = 2.5e-2
    sharex: bool = False
    sharey: bool = False
    _nrows = 1
    _ncols = 1
    _nvar1 = 1
    _nvar2 = 1
    _npanels = 1
    _col_by_var1 = True

    # Save settings
    show: Optional[bool] = None
    save: Optional[Union[bool, str]] = None

    # def __post_init__(self):
    #     # super().__post_init__()
    #     self._nrows = 1
    #     self._ncols = 1
    #     self._nvar1 = 1
    #     self._nvar2 = 1
    #     self._npanels = 1
    #     self._col_by_var1 = True

    def create_fig(
        self,
        var1: Iterable[Any],
        var2: Iterable[Any],
        ax: Optional[mpl.axes.Axes] = None,
        fig: Optional[mpl.figure.Figure] = None,
    ) -> None:
        null_var1 = sum([x is not None for x in var1]) == 0
        null_var2 = sum([x is not None for x in var2]) == 0
        self._nvar1 = len(var1)
        self._nvar2 = len(var2)
        self._npanels = self._nvar1 * self._nvar2
        self._col_by_var1 = self._nvar1 >= self._nvar2
        self._ncols = (
            self.ncols
            if self.ncols is not None
            else (self._nvar1 if self._col_by_var1 else self._nvar2)
            if ((not null_var1) and (not null_var2))
            else min(5, max(self._nvar1, self._nvar2))
        )
        self._nrows = int(np.ceil(self._npanels / self._ncols))

        if ax is None or self._npanels > 1:
            ws = self.wspace
            hs = self.hspace
            if self.fixed_figsize is None:
                fs = self.figsize
                fw = ((1 + ws) * self._ncols - ws) * fs[0]
                fh = ((1 + hs) * self._nrows - hs) * fs[1]
            else:
                fw = self.fixed_figsize[0]
                fh = self.fixed_figsize[1]
                fs = (
                    fw / ((1 + ws) * self._ncols - ws),
                    fh / ((1 + hs) * self._nrows - hs),
                )
                self.figsize = fs
            subplot_params = dict(
                nrows=self._nrows,
                ncols=self._ncols,
                figsize=(fw, fh),
                squeeze=False,
                subplot_kw=dict(axisbelow=True),
            )
            if fig is None:
                self.fig, self.axs = plt.subplots(**subplot_params)
            else:
                self.fig = fig
                self.axs = fig.subplots(**subplot_params)
        else:
            self.fig = ax.get_figure()
            self.axs = [[ax]]
            self.figsize = self.fig.get_size_inches()

    def get_ax(
        self, idx_var1: int, idx_var2: int, return_idx: bool = False
    ) -> Union[mpl.axes.Axes, Tuple[mpl.axes.Axes, int]]:
        idx_ax = (
            (idx_var2 * self._nvar1 + idx_var1)
            if self._col_by_var1
            else (idx_var1 * self._nvar2 + idx_var2)
        )
        _ax = self.axs[(idx_ax // self._ncols)][(idx_ax % self._ncols)]
        return (_ax, idx_ax) if return_idx else _ax

    def cleanup_shared_axis(self, sharex: bool = True, sharey: bool = True) -> None:
        idx_ax = 0
        max_n_ax = self._npanels
        while idx_ax < max_n_ax:
            _idx_row = idx_ax // self._ncols
            _idx_col = idx_ax % self._ncols
            cur_ax = self.axs[_idx_row, _idx_col]
            if sharex and (
                _idx_row < (self._nrows - 1)
                and self.axs[(_idx_row + 1)][_idx_col].has_data()
            ):
                cur_ax.set_xlabel("")
            if sharey and (_idx_col > 0):
                cur_ax.set_ylabel("")
            idx_ax += 1
        return

    def cleanup(self) -> None:
        idx_ax = self._npanels
        max_n_ax = self._nrows * self._ncols
        while idx_ax < max_n_ax:
            _idx_ax = idx_ax
            self.axs[(_idx_ax // self._ncols)][(_idx_ax % self._ncols)].remove()
            idx_ax += 1
        return

    def save_or_show(
        self, plot_name: str
    ) -> Optional[Union[mpl.axes.Axes, Iterable[mpl.axes.Axes]]]:
        from scanpy.plotting._utils import savefig_or_show

        _ = savefig_or_show(plot_name, show=self.show, save=self.save)
        if isinstance(self.show, bool) and not self.show:
            return self.axs[0][0] if (self._npanels == 1) else self.axs

    def get_title_textwrap_length(self) -> None:
        if self.title_textwrap_length is None:
            self.title_textwrap_length = max(50, 30 * self._ncols)
