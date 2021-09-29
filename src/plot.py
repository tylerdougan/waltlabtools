"""Class CalCurve, its methods, and related functions.

Everything in waltlabtools.cal_curve is automatically imported with
waltlabtools, so it can be accessed via, e.g.,

.. code-block:: python

   import waltlabtools as wlt  # waltlabtools main functionality

   my_cal_curve = wlt.CalCurve()  # creates a new empty calibration curve


-----


"""

import matplotlib.pyplot as plt

from .core import flatten, _optional_dependencies
from .cal_curve import CalCurve

if _optional_dependencies["jax"]:
    import jax.numpy as np
else:
    import numpy as np


__all__ = ["plot_cal_curve"]


_pyplot_kwargs = {
    "figure": {"agg_filter", "canvas", "clear", "constrained_layout",
        "constrained_layout_pads", "dpi", "edgecolor", "facecolor",
        "figheight", "figsize", "figure", "FigureClass", "figwidth",
        "frameon", "gid", "label", "layout", "linewsidth", "num", "picker",
        "rasterized", "size_inches", "snap", "subplotpars", "tight_layout",
        "transform", "visible"},
    "axes": {"adjustable", "anchor", "arg", "aspect", "autoscale_on",
        "autoscalex_on", "autoscaley_on", "axes_locator", "axisbelow",
        "box_aspect", "fc", "frame_on", "navigate", "navigate_mode", "polar",
        "position", "projection", "prop_cycle", "rasterization_zorder",
        "sharex", "sharey", "title", "xbound", "xlabel", "xlim", "xmargin",
        "xscale", "xticklabels", "xticks", "ybound", "ylabel", "ylim",
        "ymargin", "yscale", "yticklabels", "yticks", "facecolor", "figure",
        "agg_filter", "gid", "picker", "rasterized", "snap", "transform",
        "visible"},
    "scatter_only": {"antialiaseds", "capstyle", "cmap", "edgecolors",
        "facecolors", "hatch", "joinstyle", "linestyles", "linewidths",
        "norm", "offsets", "plotnonfinite", "s", "transOffset", "urls",
        "vmax", "vmin"},
    "scatter_plot": {"alpha", "antialiaseds", "capstyle", "cmap", "edgecolors",
        "facecolors", "hatch", "joinstyle", "linestyles", "linewidths",
        "marker", "norm", "offsets", "pickradius", "plotnonfinite", "s",
        "transOffset", "urls", "vmax", "vmin"},
    "plot_curve": {"dash_capstyle", "dash_joinstyle", "dashes", "drawstyle",
        "ds", "gid", "linestyle", "linewidth", "ls", "lw", "path_effects",
        "picker", "rasterized", "sketch_params", "snap", "solid_capstyle",
        "solid_joinstyle", "transform", "visible"}
}


def _plot_figure(kwargs):
    if "figure" not in kwargs.keys():
        figure_kwargs = {kwargname: kwargs[kwargname] for kwargname
            in _pyplot_kwargs["figure"].intersection(kwargs.keys())}
        plt.figure(**figure_kwargs)


def _plot_axes(kwargs, self=None):
    axes_kwargs = {kwargname: kwargs[kwargname]
        for kwargname in _pyplot_kwargs["axes"].intersection(kwargs.keys())}
    if "xscale" not in kwargs.keys():
        axes_kwargs["xscale"] = self.model.xscale
    if "yscale" not in kwargs.keys():
        axes_kwargs["yscale"] = self.model.yscale
    ax = plt.axes(**axes_kwargs)
    return ax


def _make_curve(kwargs, self, x=None, start=None, stop=None, num=1000):
    if x is None:
        if "xlim" in kwargs.keys():
            x_start = kwargs["xlim"][0] if start is None else start
            x_stop = kwargs["xlim"][1] if stop is None else stop
        else:
            sorted_x = np.unique(flatten(self.x))
            if self.model.xscale == "log":
                pos_x = sorted_x[sorted_x > 0]
                x_start = pos_x[0]**2 / pos_x[1]
                x_stop = pos_x[-1]**2 / pos_x[-2]
                x_flat = np.geomspace(x_start, x_stop, num)
            else:
                x_start = 2*sorted_x[0] - sorted_x[1]
                x_stop = 2*sorted_x[-1] - sorted_x[-2]
                x_flat = np.linspace(x_start, x_stop, num)
    else:
        x_flat = flatten(x)
    y_flat = self.fun(x_flat)
    return x_flat, y_flat


def _point_kwargs(kwargs, point_color=None)
    point_kwargnames = _pyplot_kwargs["scatter_plot"].intersection(kwargs.keys())
    point_kwargs = {kwargname: kwargs[kwargname]
        for kwargname in point_kwargnames}
    point_kwargs["c"] = point_color
    return point_kwargs


def plot_cal_curve(self, point_color=None, curve_color=None, x=None,
        start=None, stop=None, num: int = 1000, plot_points_with=None,
        show: bool = True, **kwargs):
    """Kwarg options.

    xscale, yscale
    xlabel, ylabel
    title
    figsize
    fmt
    xlim, ylim

    """
    _plot_figure(kwargs)
    ax = _plot_axes(kwargs)
    x_flat, y_flat = _make_curve(kwargs, self, x, start, stop, num)

    curve_kwargs = {kwargname: kwargs[kwargname] for kwargname
        in _pyplot_kwargs["plot_curve"].intersection(kwargs.keys())}
    curve_kwargs["c"] = curve_color
    ax.plot(x_flat, y_flat, label=self.model.name + "Calibration Curve",
        **curve_kwargs)

    point_kwargs = _point_kwargs(kwargs, point_color)
    must_scatter = _pyplot_kwargs["scatter_only"].intersection(kwargs.keys())
    if (plot_points_with == "scatter") or must_scatter:
        ax.scatter(self.x, self.y, **point_kwargs)
    else:
        ax.plot(self.x, self.y, "o", **point_kwargs)

    if show:
        plt.show()
    return ax
