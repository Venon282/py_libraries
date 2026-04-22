from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes as MplAxes
from matplotlib.figure import Figure as MplFigure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# internal
from .Config import AxisConfig
from .main import _unwrap, makeBins, save as saveFig
from .Theme import Theme, THEMES
from ..other.loggingUtils import getLogger

logger = getLogger(__name__)


class PlotAxes:
    """
    Fluent wrapper around matplotlib Axes.

    All plot methods return self to allow chaining:
        ax.plot(data).scatter(pts).style(title='Result', legend=True)

    Style configuration is applied lazily: call render() explicitly,
    or let Figure.show() / Figure.save() trigger it automatically.
    """

    def __init__(self, ax: MplAxes, config: Optional[AxisConfig] = None) -> None:
        self._ax     = ax
        self._config = config if config is not None else AxisConfig()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def style(self, **kwargs) -> PlotAxes:
        """
        Update axis configuration. Accepts any AxisConfig field as a keyword argument.

        Example: ax.style(title='Signal', xlabel='Time (s)', legend=True)
        """
        valid = AxisConfig.fieldNames()
        for key, val in kwargs.items():
            if key not in valid:
                raise ValueError(
                    f"Unknown style option '{key}'. Valid options: {sorted(valid)}"
                )
            setattr(self._config, key, val)
        return self

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot(self, *args,
             cmap: Optional[str] = None,
             cmap_label: str = '',
             cmap_values=None,
             **kwargs) -> PlotAxes:
        """Line plot."""
        colors = None
        if cmap:
            cmap_obj = cm.get_cmap(cmap)
            if cmap_values is not None:
                cmap_arr   = np.asarray(cmap_values, dtype=float)
                cmap_range = cmap_arr.max() - cmap_arr.min()
                normed = (
                    (cmap_arr - cmap_arr.min()) / cmap_range
                    if cmap_range > 0
                    else np.zeros(len(cmap_arr))
                )
                colors = cmap_obj(normed)
            else:
                colors = cmap_obj(np.linspace(0, 1, len(args)))

        for i, arg in enumerate(args):
            x, y, opts = _unwrap(arg)
            if colors is not None:
                opts.setdefault('color', colors[i])
            self._ax.plot(x, y, **{**kwargs, **opts})

        if cmap and colors is not None:
            norm_src = colors if cmap_values is None else np.asarray(cmap_values, dtype=float)
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=norm_src.min(), vmax=norm_src.max()),
            )
            self._ax.figure.colorbar(sm, ax=self._ax, label=cmap_label)

        return self

    def scatter(self, *args,
                cmap_label: Optional[str] = None,
                **kwargs) -> PlotAxes:
        """Scatter plot. Accepts c=array, cmap='...' for color-mapping."""
        for arg in args:
            x, y, opts = _unwrap(arg)
            sc = self._ax.scatter(x, y, **{**kwargs, **opts})
            if cmap_label:
                self._ax.figure.colorbar(sc, ax=self._ax, label=cmap_label)
        return self

    def histogram(self, data,
                  bins: Optional[int] = None,
                  bin_type: str = 'linear',
                  **kwargs) -> PlotAxes:
        """Univariate histogram."""
        if bin_type != 'linear':
            bins = makeBins(data, bins=bins if bins is not None else 100, bin_type=bin_type)
        self._ax.hist(data, bins=bins, **kwargs)
        return self

    def bar(self, *args,
            text: bool = False,
            fmt: str = '{}',
            **kwargs) -> PlotAxes:
        """Bar chart. Set text=True to annotate bar heights; use fmt to control format."""
        for arg in args:
            x, y, opts = _unwrap(arg)
            bars = self._ax.bar(x, y, **{**kwargs, **opts})
            if text:
                for bar in bars:
                    height = bar.get_height()
                    self._ax.annotate(
                        fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom',
                    )
        return self

    def errorbar(self, *args, **kwargs) -> PlotAxes:
        """Error bar plot. Each series is plotted as mean +/- std."""
        for arg in args:
            x, y, opts = _unwrap(arg)
            self._ax.errorbar(
                x, y,
                **{**kwargs, **opts},
            )
        return self

    def scatterDensity(self, *args,
                       density_alpha: bool = False,
                       center: Union[bool, float] = False,
                       rate: Union[bool, float, int] = False,
                       bw_method: Optional[Union[str, float]] = None,
                       cbar_label: Optional[str] = None,
                       **kwargs) -> PlotAxes:
        """
        Scatter plot colored by 2D Gaussian KDE density.

        Parameters
        ----------
        density_alpha : bool
            Scale point opacity by local density.
        center : bool or float
            Restrict the view to the densest region. True = 90%.
            A value in (0, 1] is a fraction; a value > 1 is a percentage.
        rate : float or int
            Discard low-density points before plotting.
            Float: keep the top fraction. Int: keep the top N points.
        bw_method : str or float, optional
            Bandwidth selector forwarded to scipy.stats.gaussian_kde.
        cbar_label : str, optional
            Colorbar label. Pass None to suppress the colorbar.
        """
        legend_handles: List[Line2D] = []
        sc = None

        for arg in args:
            x, y, opts = _unwrap(arg)

            kde = scipy.stats.gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
            z   = kde(np.vstack([x, y]))
            idx = z.argsort()  # ascending so dense points render on top

            if rate:
                if isinstance(rate, float):
                    idx = idx[int(len(idx) * (1 - rate)):]
                elif isinstance(rate, int):
                    idx = idx[max(0, len(idx) - rate):]
                else:
                    logger.warning("rate must be float or int; got %s (%s). Ignoring.", rate, type(rate))

            x, y, z = x[idx], y[idx], z[idx]

            alpha_values = (z - z[0]) / (z[-1] - z[0]) if density_alpha else None

            if center:
                center_rate = (
                    0.9        if center is True
                    else center if 0.0 < center <= 1.0
                    else float(center) / 100
                )
                sub_idx      = int((1.0 - center_rate) * len(x))
                sub_x, sub_y = x[sub_idx:], y[sub_idx:]
                x_c, y_c     = sub_x[-1], sub_y[-1]
                x_half       = max(abs(x_c - sub_x.min()), abs(sub_x.max() - x_c))
                y_half       = max(abs(y_c - sub_y.min()), abs(sub_y.max() - y_c))
                self._config.xlim = (x_c - x_half, x_c + x_half)
                self._config.ylim = (y_c - y_half, y_c + y_half)

            sc = self._ax.scatter(x, y, c=z, alpha=alpha_values, **opts)

            label = opts.get('label', '')
            if label:
                legend_handles.append(Line2D(
                    [0], [0],
                    marker=opts.get('marker', 'o'),
                    color='none',
                    markeredgecolor=opts.get('edgecolor', 'none'),
                    markerfacecolor=sc.get_cmap()(0.5),
                    markersize=sc.get_sizes()[0] ** 0.5,
                    label=label,
                ))

        if cbar_label is not None and sc is not None:
            self._ax.figure.colorbar(sc, ax=self._ax, label=cbar_label)

        if legend_handles:
            self._config.legend_opts.setdefault('handles', legend_handles)

        return self

    def heatmap(self, *args,
                bins: int = 100,
                cmap: str = 'viridis',
                cbar_label: str = '',
                intensity=None,
                **kwargs) -> PlotAxes:
        """2D density heatmap. Pass intensity=array for weighted bin averaging."""
        for arg in args:
            x, y, opts = _unwrap(arg)
            local_intensity = opts.pop('intensity', None) or intensity
            if local_intensity is not None:
                heat_sum,   xe, ye = np.histogram2d(x, y, bins=bins, weights=local_intensity)
                heat_count, _,  _  = np.histogram2d(x, y, bins=[xe, ye])
                heat = np.divide(
                    heat_sum, heat_count,
                    out=np.zeros_like(heat_sum), where=heat_count != 0,
                )
            else:
                heat, xe, ye = np.histogram2d(x, y, bins=bins)

            im = self._ax.imshow(
                heat.T,
                extent=[xe[0], xe[-1], ye[0], ye[-1]],
                origin='lower', aspect='auto',
                cmap=cm.get_cmap(cmap),
                **kwargs, **opts,
            )
            self._ax.figure.colorbar(im, ax=self._ax, label=cbar_label)

        return self

    def image(self, *args,
              colorbar: bool = False,
              cbar_label: str = '',
              **kwargs) -> PlotAxes:
        """Display one or more images."""
        im = None
        for arg in args:
            _, img, opts = _unwrap(arg)
            im = self._ax.imshow(img, **{**kwargs, **opts})
        if colorbar and im is not None:
            self._ax.figure.colorbar(im, ax=self._ax, label=cbar_label)
        return self

    def candles(self, data,
                width: float = 0.6,
                open_color: str = 'green',
                close_color: str = 'red',
                wick_width: float = 1.0,
                show_legend: bool = True,
                **kwargs) -> PlotAxes:
        """
        OHLC candlestick chart.

        Parameters
        ----------
        data : array of shape (N, 4) -- columns: open, high, low, close.
        """
        for i, (o, h, l, c) in enumerate(np.asarray(data)):
            color = open_color if c >= o else close_color
            self._ax.plot([i, i], [l, h], color=color, lw=wick_width)
            body_low, body_high = sorted((o, c))
            self._ax.add_patch(Rectangle(
                (i - width / 2, body_low), width, body_high - body_low,
                facecolor=color, edgecolor=color,
            ))
        if show_legend:
            self._ax.add_patch(Patch(facecolor=open_color,  label='Up'))
            self._ax.add_patch(Patch(facecolor=close_color, label='Down'))
            self._ax.legend()
        return self

    def ecdf(self, x, **kwargs) -> PlotAxes:
        """Empirical cumulative distribution function."""
        self._ax.ecdf(x, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> PlotAxes:
        """Apply AxisConfig to the underlying matplotlib Axes."""
        cfg = self._config
        ax  = self._ax

        if cfg.title:   ax.set_title(cfg.title)
        if cfg.xlabel:  ax.set_xlabel(cfg.xlabel)
        if cfg.ylabel:  ax.set_ylabel(cfg.ylabel)

        ax.minorticks_on()
        if isinstance(cfg.grid, bool):
            ax.grid(cfg.grid)
        elif isinstance(cfg.grid, dict):
            ax.grid(True, **cfg.grid)
        else:
            raise TypeError(f"grid must be bool or dict, got {type(cfg.grid)}")

        if cfg.xlim        is not None: ax.set_xlim(cfg.xlim)
        if cfg.ylim        is not None: ax.set_ylim(cfg.ylim)
        if cfg.xscale      is not None: ax.set_xscale(cfg.xscale)
        if cfg.yscale      is not None: ax.set_yscale(cfg.yscale)
        if cfg.xticks      is not None: ax.set_xticks(cfg.xticks)
        if cfg.yticks      is not None: ax.set_yticks(cfg.yticks)
        if cfg.xticklabels is not None: ax.set_xticklabels(cfg.xticklabels)
        if cfg.yticklabels is not None: ax.set_yticklabels(cfg.yticklabels)
        if cfg.legend:                  ax.legend(**cfg.legend_opts)

        return self

    # ------------------------------------------------------------------
    # Raw access
    # ------------------------------------------------------------------

    @property
    def raw(self) -> MplAxes:
        """Underlying matplotlib Axes, for any operation not covered above."""
        return self._ax


# ----------------------------------------------------------------------


class Figure:
    """
    Layout manager and output controller.

    Usage
    -----
    Single axes:
        fig = Figure(theme='paper')
        ax  = fig.addAxes(title='Signal', xlabel='Time (s)')
        ax.plot((t, y)).scatter((t, pts, {'color': 'red'}))
        fig.show()

    Subplots:
        fig  = Figure(theme='paper', figsize=(12, 8))
        grid = fig.addSubplots(2, 3, sharex=True)
        grid[0][0].histogram(data)
        grid[0][1].scatter((x, y))
        fig.save('results.pdf')

    Twin axes:
        fig = Figure(theme='paper')
        ax1 = fig.addAxes(ylabel='Temperature')
        ax2 = fig.addTwin(ax1, ylabel='Pressure')
        ax1.plot((t, temp))
        ax2.plot((t, pres, {'color': 'red'}))
        fig.show()

    Inset:
        fig = Figure(theme='paper')
        ax  = fig.addAxes()
        ins = fig.addInset(ax, bounds=(0.55, 0.55, 0.4, 0.35))
        ax.plot((x, y))
        ins.plot((x_zoom, y_zoom))
        fig.show()
    """

    def __init__(self,
                 theme: Union[str, Theme] = 'paper',
                 figsize: Tuple[float, float] = (10, 6),
                 **kwargs) -> None:
        if isinstance(theme, str):
            theme = THEMES.get(theme, THEMES['default'])
        theme.apply()
        self._fig:  MplFigure       = plt.figure(figsize=figsize, **kwargs)
        self._axes: List[PlotAxes]  = []

    # ------------------------------------------------------------------
    # Axes factory methods
    # ------------------------------------------------------------------

    def addAxes(self, **style_kwargs) -> PlotAxes:
        """
        Add a single axes filling the figure.
        Any AxisConfig field can be passed as a keyword argument.
        """
        cfg     = AxisConfig.fromKwargs(style_kwargs)
        plot_ax = PlotAxes(self._fig.add_subplot(1, 1, 1), cfg)
        self._axes.append(plot_ax)
        return plot_ax

    def addSubplots(self,
                    nrows: int,
                    ncols: int,
                    sharex: bool = False,
                    sharey: bool = False,
                    hspace: float = 0.35,
                    wspace: float = 0.30) -> List[List[PlotAxes]]:
        """
        Create a (nrows x ncols) grid of axes.
        Returns a 2D list; access panels as grid[row][col].

        Parameters
        ----------
        sharex : bool
            All axes in each column share the same x-axis.
        sharey : bool
            All axes in each row share the same y-axis.
        hspace : float
            Vertical spacing between rows (in fraction of average axis height).
        wspace : float
            Horizontal spacing between columns.
        """
        gs       = gridspec.GridSpec(nrows, ncols, figure=self._fig, hspace=hspace, wspace=wspace)
        grid:     List[List[PlotAxes]] = []
        raw_grid: List[List[MplAxes]]  = []

        for r in range(nrows):
            row_plot: List[PlotAxes] = []
            row_raw:  List[MplAxes]  = []
            for c in range(ncols):
                sx = raw_grid[0][c] if sharex and r > 0 else None
                sy = row_raw[0]     if sharey and c > 0 else None
                raw_ax = self._fig.add_subplot(gs[r, c], sharex=sx, sharey=sy)
                plot_ax = PlotAxes(raw_ax)
                self._axes.append(plot_ax)
                row_plot.append(plot_ax)
                row_raw.append(raw_ax)
            grid.append(row_plot)
            raw_grid.append(row_raw)

        return grid

    def addTwin(self, source: PlotAxes,
                axis: str = 'x',
                **style_kwargs) -> PlotAxes:
        """
        Add an axis that shares the x-axis (axis='x') or y-axis (axis='y') with source.

        The new axis gets its own independent scale on the opposite side.
        Any AxisConfig field (e.g. ylabel='Pressure') can be passed as a keyword argument.
        """
        if axis == 'x':
            raw_ax = source.raw.twinx()
        elif axis == 'y':
            raw_ax = source.raw.twiny()
        else:
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")
        cfg     = AxisConfig.fromKwargs(style_kwargs)
        plot_ax = PlotAxes(raw_ax, cfg)
        self._axes.append(plot_ax)
        return plot_ax

    def addInset(self, source: PlotAxes,
                 bounds: Tuple[float, float, float, float],
                 **style_kwargs) -> PlotAxes:
        """
        Add an inset axes inside source.

        Parameters
        ----------
        bounds : (x0, y0, width, height) in axes-fraction coordinates [0, 1].
        """
        cfg     = AxisConfig.fromKwargs(style_kwargs)
        plot_ax = PlotAxes(source.raw.inset_axes(bounds), cfg)
        self._axes.append(plot_ax)
        return plot_ax

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _renderAll(self) -> None:
        for ax in self._axes:
            ax.render()
        self._fig.tight_layout()

    def show(self) -> None:
        """Render and display the figure interactively."""
        self._renderAll()
        plt.show()
        return self

    def save(self, path: str, dpi: int = 150) -> None:
        """Render and write the figure to disk."""
        self._renderAll()
        saveFig(self._fig, path, dpi=dpi)
        return self

    def close(self) -> None:
        """Close the figure and release its memory."""
        plt.close(self._fig)

    def __enter__(self) -> Figure:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Raw access
    # ------------------------------------------------------------------

    @property
    def raw(self) -> MplFigure:
        """Underlying matplotlib Figure, for any operation not covered above."""
        return self._fig