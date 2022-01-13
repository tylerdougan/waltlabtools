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
    "fig_kw": {
        "clear",
        "constrained_layout",
        "dpi",
        "figsize",
        "FigureClass",
        "num",
        "subplotpars",
        "tight_layout",
    },
    "ax_kw": {
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
    "point_kw": {
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
    "lod_kw": {
        "ymax",
        "ymin",
    },
    "legend_kw": {
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
    "curve_kw": set(),
    None: set(),
}
"""Parameters accepted by one, and only one, matplotlib function.

Each key is a plot element; its value is the set of parameters accepted
by the matplotlib function used for that plot element.

"""


def _get_kw(default_kw=None, plot_element=None, par=None, **kwargs):
    """Gets parameters from kwargs for a given plot element.

    Any keys corresponding to a parameter in a single plot element will
    be returned. If parameters are specified via the plot element key
    (e.g., "ax_kw"), they will also be returned. In the case of
    duplicate values for the same parameter, the argument specified via
    the plot element key will take precedence.

    Parameters
    ----------
    default_kw : dict, optional
        Default parameters.
    plot_element : {"ax_kw", "fig_kw", "point_kw", "curve_kw", "lod_kw",
    "legend_kw"}, optional
        Plot element. This specifies both the key in _UNIQUE_PYPLOT_ARGS
        to search through, and the key within kwargs that gives the dict
        of relevant arguments. If None, then par must be given.
    par : str, optional
        If specified, then only the argument corresponding to the
        parameter par will be returned.

    Returns
    -------
    dict, object, or None
        If par is None, a dict mapping parameters (keys) to arguments
        (values). If par is given, then only the argument (value)
        corresponding to the parameter par will be returned. If par is
        not found, then None is returned.

    """
    new_kw = default_kw if default_kw is not None else {}
    new_kw.update(
        {
            key: kwargs[key]
            for key in _UNIQUE_PYPLOT_ARGS[plot_element].intersection(kwargs.keys())
        }
    )
    if plot_element in kwargs:
        new_kw.update(kwargs[plot_element])
    return new_kw if par is None else new_kw[par] if par in new_kw else None


def _cal_curve_points(cal_curve, **kwargs) -> tuple:
    """Calculates points for a calibration curve.

    If x is specified (either passed directly within kwargs or via
    point_kw), then it will be used as the x-values for drawing the
    calibration curve. If x is not specified, then the x-values will be
    generated as a linear or geometric sequence based on the limit of
    detection and the calibrator points.

    Parameters
    ----------
    cal_curve : CalCurve
        Calibration curve. The following attributes are used:
        x, lod, model.xscale.

    Returns
    -------
    tuple
        Tuple of (x, y) points. Both x and y are 1D numpy arrays.

    Other Parameters
    ----------------
    point_kw : dict, optional
        Keyword arguments for plotting calibration curve points,
        including x.
    ax_kw : dict, optional
        Keyword arguments for the matplotlib Axes, including xscale and
        xlim.
    x : list-like, optional
        The x values to use when generating the curve.
    xscale : {"linear", "log", "symlog", "logit", ...} or ScaleBase,
    optional
        The x-scale scale type to apply. If xscale == "log", then the
        x-values will be generated as a geometric series; otherwise,
        they will be generated as an arithmetic series.
    xlim : list-like of length 2, optional
        Limits for the x-axis. These will be used as the limits of the
        x-values generated.

    """
    x = _get_kw(plot_element="point_kw", par="x", **kwargs)
    if x is not None:
        x_flat = flatten(x)
    else:
        xscale = _get_kw(plot_element="ax_kw", par="xscale", **kwargs)
        if xscale is not None:
            use_log = xscale == "log"
        else:
            use_log = cal_curve.model.xscale == "log"
        xlim = _get_kw(plot_element="ax_kw", par="xlim", **kwargs)
        try:
            start, stop = xlim
        except (TypeError, ValueError):
            # Infer limits from calibrator points.
            sorted_x = np.unique(cal_curve.x)
            if use_log:
                sorted_x = sorted_x[sorted_x > 0]
                lowest_x = min(sorted_x[0], cal_curve.lod)
                common_ratio = (sorted_x[-1] / sorted_x[0]) ** (1 / (len(sorted_x) - 1))
                start = lowest_x / common_ratio
                stop = sorted_x[-1] * common_ratio
                x_flat = np.geomspace(start, stop)
            else:
                lowest_x = min(sorted_x[0], cal_curve.lod)
                common_difference = (sorted_x[-1] - sorted_x[0]) / (len(sorted_x) - 1)
                start = lowest_x - common_difference
                stop = sorted_x[-1] + common_difference
                x_flat = np.linspace(start, stop)
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
    """Given a matplotlib.axes.Axes object, plots a calibration curve.

    [extended_summary]

    Parameters
    ----------
    cal_curve : CalCurve
        CalCurve object to plot.
    ax : plt.Axes
        The matplotlib.axes.Axes object where the curve
        and data should be plotted.
    show : bool, optional
        If True, show the plot (via pyplot.show()) and return None. If
        False, return the matplotlib.axes.Axes object.
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
        [description]
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
        to set the transparency of the calibrator points to 75%, use
        point_kw={"alpha": 0.75}.
    x : array-like, optional
        Data to use for x-values when plotting the calibration curve.
        Defaults to generating 50 x-values based on the calibrator
        points.

    """
    if "points" not in hide:
        new_point_color = point_color if point_color is not None else "k"
        new_point_kw = _get_kw(
            default_kw={
                "color": new_point_color,
                "alpha": 0.6,
                "label": "Calibrator points",
            },
            plot_element="point_kw",
            **kwargs
        )
        if _UNIQUE_PYPLOT_ARGS["point_kw"].intersection(new_point_kw.keys()):
            ax.scatter(cal_curve.x, cal_curve.y, **new_point_kw)
        else:
            fmt = new_point_kw.pop("fmt", "o")
            ax.plot(cal_curve.x, cal_curve.y, fmt, **new_point_kw)
    if "curve" not in hide:
        x_flat, y_flat = _cal_curve_points(cal_curve, **kwargs)
        new_curve_kw = _get_kw(
            default_kw={
                "color": curve_color if curve_color is not None else "C2",
                "alpha": 0.8,
                "label": str(cal_curve.model.name) + " calibration curve",
            },
            plot_element="curve_kw",
            **kwargs
        )
        ax.plot(x_flat, y_flat, **new_curve_kw)
    if "lod" not in hide:
        new_lod_kw = _get_kw(
            default_kw={
                "x": cal_curve.lod,
                "color": lod_color if lod_color is not None else "C7",
                "label": "Limit of detection",
                "alpha": 0.6,
                "ls": "dashed",
            },
            plot_element="lod_kw",
            **kwargs
        )
        ax.axvline(**new_lod_kw)
    if "legend" not in hide:
        new_legend_kw = _get_kw(plot_element="legend_kw", **kwargs)
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
    """Plots a calibration curve.

    Parameters
    ----------
    cal_curve : CalCurve
        [description]
    ax : matplotlib.axes.Axes or "current", optional
        Optionally provide a matplotlib.axes.Axes object where the curve
        and data should be plotted. To plot the figure and data on the
        current Axes, pass ax="current".
    fig : matplotlib.figure.Figure, optional
        Optionally provide a matplotlib.figure.Figure object where the
        curve and data should be plotted. Ignored if ax is provided;
        if ax is not provided, a new Axes object will be created within
        fig.
    show : bool, optional
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
    matplotlib.axes.Axes or None
        If show=False, returns the Axes object; if show=True, returns
        none.

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

    See Also
    --------
    CalCurve.plot() : equivalent method

    """
    if isinstance(ax, plt.Axes):
        new_ax = ax
    elif ax == "current":
        new_ax = plt.gca()
    else:
        new_ax_kw = _get_kw(
            default_kw={
                "xscale": cal_curve.model.xscale,
                "yscale": cal_curve.model.yscale,
            },
            plot_element="ax_kw",
            **kwargs
        )
        if isinstance(fig, plt.Figure):
            new_ax = fig.add_subplot(**new_ax_kw)
        elif fig == "current":
            new_ax = plt.axes(**new_ax_kw)
        else:
            new_fig_kw = _get_kw(plot_element="fig_kw", **kwargs)
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
