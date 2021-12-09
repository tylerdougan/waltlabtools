"""Methods for plotting calibration curves, etc.

In addition to the dependencies for waltlabtools, waltlabtools.plot
also requires matplotlib.

Everything in waltlabtools.plot is automatically imported with
waltlabtools if matplotlib is installed, so it can be accessed via,
e.g.,

.. code-block:: python

   import waltlabtools as wlt  # waltlabtools main functionality

   my_curve = wlt.CalCurve()  # creates a new empty calibration curve
   wlt.plot_cal_curve(my_curve)  # plot my_curve (in this case, empty)

   my_curve.plot()  # equivalent method



-----


"""

import matplotlib.pyplot as plt

from .core import _optional_dependencies, flatten

if _optional_dependencies["jax"]:
    import jax.numpy as np
else:
    import numpy as np


__all__ = ["plot_cal_curve"]


_UNIQUE_PYPLOT_ARGS = {
    "figure": {
        "clear",
        "constrained_layout",
        "dpi",
        "figsize",
        "FigureClass",
        "num",
        "subplotpars",
        "tight_layout",
    },
    "axes": {
        "adjustable",
        "anchor",
        "arg",
        "aspect",
        "autoscale_on",
        "autoscalex_on",
        "autoscaley_on",
        "ax",
        "axes_class",
        "axes_locator",
        "axisbelow",
        "box_aspect",
        "fc",
        "frame_on",
        "index",
        "navigate",
        "navigate_mode",
        "ncols",
        "nrows",
        "polar",
        "pos",
        "position",
        "projection",
        "prop_cycle",
        "rasterization_zorder",
        "sharex",
        "sharey",
        "xbound",
        "xlabel",
        "xlim",
        "xmargin",
        "xscale",
        "xticklabels",
        "xticks",
        "ybound",
        "ylabel",
        "ylim",
        "ymargin",
        "yscale",
        "yticklabels",
        "yticks",
    },
    "scatter": {
        "capstyle",
        "cmap",
        "edgecolors",
        "facecolors",
        "hatch",
        "joinstyle",
        "linestyles",
        "linewidths",
        "norm",
        "offset_position",
        "offsets",
        "plotnonfinite",
        "s",
        "transOffset",
        "urls",
        "vmax",
        "vmin",
    },
    "line": {
        "ymax",
        "ymin",
    },
    "legend": {
        "bbox_to_anchor",
        "bbox_transform",
        "borderaxespad",
        "borderpad",
        "columnspacing",
        "fancybox",
        "fontsize",
        "framealpha",
        "handlelength",
        "handler_map",
        "handles",
        "handletextpad",
        "labelcolor",
        "labels",
        "labelspacing",
        "loc",
        "markerfirst",
        "markerscale",
        "mode",
        "ncol",
        "numpoints",
        "prop",
        "scatterpoints",
        "scatteryoffset",
        "shadow",
        "title_fontsize",
    },
    "plot": set(),
}


def _find_kwarg(name, specific_key=None, name_in_specific_key=None, **kwargs):
    if name in kwargs:
        return kwargs[name]
    elif (specific_key is not None) and (specific_key in kwargs):
        new_name_in_specific_key = (
            name_in_specific_key if name_in_specific_key is not None else name
        )
        if new_name_in_specific_key in kwargs[specific_key]:
            return kwargs[specific_key][new_name_in_specific_key]
    return None


def _get_kwargs(kwargs: dict, matplotlib_fn=None, specific_key=None):
    if matplotlib_fn is not None:
        new_kwargs = {
            kwargname: kwargs[kwargname]
            for kwargname in _UNIQUE_PYPLOT_ARGS[matplotlib_fn].intersection(
                kwargs.keys()
            )
        }
    else:
        new_kwargs = {}
    if specific_key in kwargs:
        new_kwargs.update(kwargs[specific_key])
    return new_kwargs


def _cal_curve_points(cal_curve, **kwargs) -> tuple:
    x = _find_kwarg("x", specific_key="curve_kw", **kwargs)
    if x is not None:
        x_flat = flatten(x)
    else:
        xscale = _find_kwarg("xscale", "axes_kw", **kwargs)
        use_log = (
            xscale == "log" if xscale is not None else cal_curve.model.xscale == "log"
        )
        xlim = _find_kwarg("xlim", "axes_kw", **kwargs)
        try:
            start, stop = xlim
        except (TypeError, ValueError):
            # Infer limits from calibrator points.
            sorted_x = np.unique(cal_curve.x)
            use_log = cal_curve.model.xscale == "log"
            if use_log:
                sorted_x = sorted_x[sorted_x > 0]
                lowest_x = min(sorted_x[0], cal_curve.lod)
                common_ratio = (sorted_x[-1] / sorted_x[0]) ** (1 / (len(sorted_x) - 1))
                start = lowest_x / common_ratio
                stop = sorted_x[-1] * common_ratio
            else:
                lowest_x = min(sorted_x[0], cal_curve.lod)
                common_difference = (sorted_x[-1] - sorted_x[0]) / (len(sorted_x) - 1)
                start = lowest_x - common_difference
                stop = sorted_x[-1] + common_difference
        x_flat = np.geomspace(start, stop) if use_log else np.linspace(start, stop)
    y_flat = cal_curve.fun(x_flat)
    return x_flat, y_flat


def _ax_cal_curve(
    cal_curve,
    ax: plt.Axes,
    show: bool = True,
    hide=(),
    point_color=None,
    curve_color=None,
    lod_color=None,
    **kwargs
) -> plt.Axes:
    """[summary]

    [extended_summary]

    Parameters
    ----------
    cal_curve : CalCurve
        CalCurve object to plot.
    ax : matplotlib.axes.Axes, optional
        Optionally provide a matplotlib.axes.Axes object where the curve
        and data should be plotted.
    hide : collection of str or str, optional
        By default, the calibration curve, calibrator points, limit of
        detection (LOD), and legend are shown. To hide any of these,
        include "points", "curve", "lod", and/or "legend" in hide.
        For example, to hide the legend, use hide="legend" or
        hide=["legend"].
    point_color, curve_color, lod_color : str or tuple, optional
        Colors for the calibrator points and curve. Can be a color
        name (``"gray"``) or shorthand (``"g"``); an RGB or RGBA
        tuple of float values in the closed interval [0, 1]
        (``(0.5, 0.5, 0.5)``); a hex RGB or RGBA string
        (``"808080"``); or another form accepted by
        matplotlib.colors. For more details, see
        https://matplotlib.org/stable/tutorials/colors/colors.html.
    **kwargs
        Any keyword argument that is unique to one element of the plot
        (figure, axes, curve, points, LOD, legend) can be passed. To
        specify a keyword argument for one of the plot elements, pass
        it via the appropriate argument below.

    Returns
    -------
    plt.Axes
        [description]

    Other Parameters
    ----------------
    ax_kw, fig_kw, point_kw, curve_kw, lod_kw, legend_kw : dict, optional
        Keywords to pass to specific elements of the plot. For example,
        to set the transparency of the calibrator points to 50%, use
        point_kw={"alpha": 0.5}.
    x : array-like, optional
        Data to use for x-values when plotting the calibration curve.
        Defaults to generating 50 x-values based on the calibrator
        points.

    """
    if "points" not in hide:
        new_point_color = point_color if point_color is not None else "k"
        new_point_kw = {"color": new_point_color, "label": "Calibrator points"}
        new_point_kw.update(_get_kwargs(kwargs, "scatter", "point_kw"))
        if _UNIQUE_PYPLOT_ARGS["scatter"].intersection(new_point_kw.keys()):
            ax.scatter(cal_curve.x, cal_curve.y, **new_point_kw)
        else:
            fmt = new_point_kw.pop("fmt", "o")
            ax.plot(cal_curve.x, cal_curve.y, fmt, **new_point_kw)
    if "curve" not in hide:
        x_flat, y_flat = _cal_curve_points(cal_curve, **kwargs)
        new_curve_color = curve_color if curve_color is not None else "C2"
        new_curve_kw = {
            "color": new_curve_color,
            "label": str(cal_curve.model.name) + " calibration curve",
        }
        new_curve_kw.update(_get_kwargs(kwargs, "plot", "curve_kw"))
        ax.plot(x_flat, y_flat, **new_curve_kw)
    if "lod" not in hide:
        new_lod_color = lod_color if lod_color is not None else "C7"
        new_lod_kw = {
            "x": cal_curve.lod,
            "color": new_lod_color,
            "label": "Limit of detection",
            "ls": "dashed"
        }
        new_lod_kw.update(_get_kwargs(kwargs, "line", "lod_kw"))
        ax.axvline(**new_lod_kw)
    if "legend" not in hide:
        new_legend_kw = _get_kwargs(kwargs, "legend", "legend_kw")
        ax.legend(**new_legend_kw)
    if show:
        plt.show()
    else:
        return ax


def plot_cal_curve(
    cal_curve,
    ax=None,
    fig=None,
    show: bool = True,
    hide=(),
    point_color=None,
    curve_color=None,
    lod_color=None,
    **kwargs
):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    cal_curve : CalCurve
        CalCurve object to plot.
    ax : matplotlib.axes.Axes, optional
        Optionally provide a matplotlib.axes.Axes object where the curve
        and data should be plotted.
    fig : matplotlib.figure.Figure, optional
        Optionally provide a matplotlib.figure.Figure object where the
        curve and data should be plotted. Ignored if ax is provided;
        otherwise, a new Axes object will be created within fig.
    show : bool, default True
        Set show=False to return the axes object without showing it.
    hide : collection of str or str, optional
        By default, the calibration curve, calibrator points, limit of
        detection (LOD), and legend are shown. To hide any of these,
        include "points", "curve", "lod", and/or "legend" in hide.
        For example, to hide the legend, use hide="legend" or
        hide=["legend"].
    point_color, curve_color, lod_color : str or tuple, optional
        Colors for the calibrator points and curve. Can be a color
        name (``"gray"``) or shorthand (``"g"``); an RGB or RGBA
        tuple of float values in the closed interval [0, 1]
        (``(0.5, 0.5, 0.5)``); a hex RGB or RGBA string
        (``"808080"``); or another form accepted by
        matplotlib.colors. For more details, see
        https://matplotlib.org/stable/tutorials/colors/colors.html.
    **kwargs
        Any keyword argument that is unique to one element of the plot
        (figure, axes, curve, points, LOD, legend) can be passed. To
        specify a keyword argument for one of the plot elements, pass
        it via the appropriate argument below.

    Returns
    -------
    [type]
        [description]

    Other Parameters
    ----------------
    ax_kw, fig_kw, point_kw, curve_kw, lod_kw, legend_kw : dict, optional
        Keywords to pass to specific elements of the plot. For example,
        to set the transparency of the calibrator points to 50%, use
        point_kw={"alpha": 0.5}.
    x : array-like, optional
        Data to use for x-values when plotting the calibration curve.
        Defaults to generating 50 x-values based on the calibrator
        points.

    """
    if isinstance(ax, plt.Axes):
        new_ax = ax
    elif ax == "current":
        new_ax = plt.gca()
    else:
        new_ax_kw = _get_kwargs(kwargs, "axes", "ax_kw")
        if isinstance(fig, plt.Figure):
            new_ax = fig.add_subplot(**new_ax_kw)
        elif fig == "current":
            new_ax = plt.axes(**new_ax_kw)
        else:
            new_fig_kw = _get_kwargs(kwargs, "figure", "fig_kw")
            new_fig = plt.figure(num=fig, **new_fig_kw)
            new_ax = new_fig.add_subplot(**new_ax_kw)
    return _ax_cal_curve(
        cal_curve,
        ax=new_ax,
        hide=hide,
        show=show,
        point_color=point_color,
        curve_color=curve_color,
        lod_color=lod_color,
        **kwargs
    )
