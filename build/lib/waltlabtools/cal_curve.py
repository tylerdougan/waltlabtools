import warnings
from collections.abc import Callable, Collection
from functools import partial
from numbers import Integral, Real
from typing import Any, Optional

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.scale import ScaleBase, scale_factory
from scipy.optimize import curve_fit, least_squares
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    _check_sample_weight,  # type: ignore
    check_array,
    check_is_fitted,
    check_non_negative,
    column_or_1d,
)

from ._backend import gmean, np
from ._plot import get_fig_ax, subplots
from .core import (
    _get_value_or_key,
    coerce_array,
    deprecate,
    dropna,
    flatten,
    geothmetic_meandian,
    match_kwargs,
    std,
)
from .model import MODELS, Model


@coerce_array
def _slice_dim(
    a,
    axis: Optional[int | Collection[int]] = None,
    keep: int = 0,
):
    """Slice an array along one or more dimensions.

    Parameters
    ----------
    a : ArrayLike
        The input array to be sliced.
    axis : None or int or tuple of ints, optional
        The dimension(s) along which to slice the array. If None, all
        dimensions except the first will be sliced. If an integer, a
        single dimension will be sliced. If a collection of integers,
        multiple dimensions will be sliced. Default is None.
    keep : int, default 0
        The index to keep along the sliced dimensions. Default is 0.

    Returns
    -------
    ArrayLike
        The sliced array.
    """
    if a.ndim == 0:
        return a
    if a.ndim == 1:
        return a[keep]
    if axis is None:
        axis = tuple(range(1, a.ndim))
    elif not isinstance(axis, Collection):
        axis = (axis,)
    indexer = tuple(keep if dim in axis else slice(None) for dim in range(a.ndim))
    return a[indexer]


_AGGREGATION_STRATEGIES = {
    "median": np.median,
    "mean": np.average,
    "average": np.average,
    "geomean": gmean,
    "gmean": gmean,
    "min": np.min,
    "max": np.max,
    "geothmetic_meandian": geothmetic_meandian,
    "gmnd": geothmetic_meandian,
    "first": partial(_slice_dim, keep=0),
    "last": partial(_slice_dim, keep=-1),
}

_WEIGHTING_SCHEMES = {
    **{key: lambda X, y: np.ones_like(y) for key in ["1", "1.0", "none", "None"]},
    "1/y": lambda X, y: 1 / y,
    "1/y^2": lambda X, y: 1 / y**2,
    "1/X": lambda X, y: 1 / X,
    "1/X^2": lambda X, y: 1 / X**2,
}


class CalCurve(BaseEstimator, RegressorMixin, TransformerMixin):
    """Calibration Curve transformer and regressor.

    Parameters
    ----------
    model : Model or str, default="4PL"
        The model to use for the calibration curve. Can be an instance
        of a Model or a string representing the model name. Current
        available options are:
        - "linear" : Linear function.
        - "power" : Power function.
        - "Hill" : Hill function.
        - "logistic" : Logistic function.
        - "3PL" : Four-parameter logistic (3PL) function.
        - "4PL" : Four-parameter logistic (4PL) function.
        - "5PL" : Five-parameter logistic (5PL) function.

    agg_reps : str Callable, or None, default="median"
        Aggregation method for replicates. Can be a string representing
        an aggregation strategy or a callable function. Current
        available options are:
        - "median"
        - "mean"
        - "average"
        - "geomean", "gmean"
        - "min"
        - "max"
        - "geothmetic_meandian", "gmnd"
        - "first"
        - "last"
        If None, then no aggregation is performed.

    coef_init : array-like, optional
        Initial coefficients for the model.

    warm_start : bool, default=False
        Whether to reuse the solution of the previous call to fit as
        initialization.

    solver : str, default="trf"
        Solver to use for optimization. Options are "trf", "dogbox", or
        "lm".

    lod_sds : float, default=3
        Number of standard deviations for limit of detection
        calculation.

    max_iter : int, optional
        Maximum number of iterations for the solver.

    ensure_2d : bool, default=False
        Whether to ensure the input is 2-dimensional.

    Attributes
    ----------
    coef_ : ndarray
        Coefficients of the fitted model.

    n_iter_ : int
        Number of iterations run by the solver.

    lod_ : float
        Calculated limit of detection.
    """

    _parameter_constraints = {
        # see docs for _param_validation.validate_parameter_constraints
        "model": [Model, StrOptions(set(MODELS))],
        "agg_reps": [callable, StrOptions(set(_AGGREGATION_STRATEGIES))],
        "coef_init": ["array-like", None],
        "warm_start": ["boolean"],
        "solver": [StrOptions({"trf", "dogbox", "lm"})],
        "lod_sds": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left"), None],
        "ensure_2d": ["boolean"],
        "sample_weight": [StrOptions(set(_WEIGHTING_SCHEMES)), None],
    }

    def __init__(
        self,
        model: Model | str = "4PL",
        agg_reps: Optional[str | Callable] = "median",
        coef_init: Optional[np.ndarray] = None,
        warm_start: bool = False,
        solver: str = "trf",
        lod_sds: float = 3,
        max_iter: Optional[int] = None,
        ensure_2d: bool = False,
        sample_weight: Optional[str] = "1/y",
        **kwargs,
    ):
        self.model = model
        self.agg_reps = agg_reps
        self.coef_init = coef_init
        self.warm_start = warm_start
        self.solver = solver
        self.lod_sds = lod_sds
        self.max_iter = max_iter
        self.ensure_2d = ensure_2d
        self.sample_weight = sample_weight

        if kwargs:
            warnings.warn(
                f"The keyword arguments {kwargs.keys()} are not recognized. "
                + "Version 1.0 introduced breaking changes to the CalCurve class. "
                + "Please refer to the documentation for the most recent API.",
                DeprecationWarning,
            )

    def _get_sample_weight(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = self.sample_weight
        if isinstance(sample_weight, str) and sample_weight in _WEIGHTING_SCHEMES:
            sample_weight = _WEIGHTING_SCHEMES[sample_weight](X, y)
        return column_or_1d(sample_weight, warn=True)

    def _aggregate_replicates(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        blank = np.ravel(y[X == 0])

        if self.agg_reps is None:
            return X, y, sample_weight, blank

        unique_X = np.unique(X, axis=0)
        agg_reps = _get_value_or_key(_AGGREGATION_STRATEGIES, self.agg_reps)
        unique_weight = []
        for x in unique_X:
            sw_x = sample_weight[x == X]
            unique_weight.append(agg_reps(sw_x) * len(sw_x))
        unique_weight = np.array(unique_weight)
        unique_X = unique_X[unique_weight > 0]
        unique_weight = unique_weight[unique_weight > 0]
        unique_y = np.array([agg_reps(y[x == X]) for x in unique_X])

        return unique_X, unique_y, unique_weight, blank

    def _preprocess_fit_data(self, X, y, sample_weight=None):
        # check X, y
        X = self._validate_data(
            X=X,
            force_all_finite=False,
            ensure_2d=self.ensure_2d,
            ensure_min_samples=len(self._model.coef_init),
        )
        y = check_array(
            y, force_all_finite=False, ensure_2d=False, estimator=self, input_name="y"
        )
        if np.ndim(X) > 1:
            if self.n_features_in_ > 1 and not self.ensure_2d:  # type: ignore
                warnings.warn(
                    "CalCurve input X should be a 1d array or 2d array with 1 feature; "
                    f"using only the first of {self.n_features_in_} features.",  # type: ignore
                    DataConversionWarning,
                )
            X = X[:, 0]

        # get sample weight
        sample_weight = self._get_sample_weight(X, y, sample_weight)

        # drop NANs and aggregate replicates
        X, y, sample_weight = dropna(X, y, sample_weight)
        self.X_, self.y_ = self._validate_data(X, y, ensure_2d=False)
        if self.ensure_2d:
            check_non_negative(self.X_, whom=self.__class__.__name__)
        sample_weight = _check_sample_weight(sample_weight, self.X_)
        return self._aggregate_replicates(self.X_, self.y_, sample_weight)

    def fit(self, X, y, sample_weight=None):
        """Fit the model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Training data features, e.g., concentrations.

        y : array-like of shape (n_samples,)
            Target signal values, e.g., AEB or fluorescence intensity.

        sample_weight : str or array-like, optional
            Sample weights.

        Returns
        -------
        CalCurve
            Fitted estimator.
        """
        # parameters
        self._validate_params()  # type: ignore
        self._model = _get_value_or_key(MODELS, self.model)

        # data
        X, y, sample_weight, blank = self._preprocess_fit_data(X, y, sample_weight)

        # coef_init
        if self.warm_start and hasattr(self, "coef_"):
            coef_init = self.coef_
        elif self.coef_init is None:
            coef_init = self._model.coef_init
        else:
            coef_init = self.coef_init

        # optimization
        def loss_func(coefs, *, X, sample_weight):
            predictions = self._model.func(X, *coefs)
            residuals = predictions - y
            return residuals * sample_weight

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = least_squares(
                loss_func,
                x0=coef_init,
                jac=self._model.jac,
                max_nfev=self.max_iter,
                method=self.solver,
                kwargs={"X": X, "sample_weight": sample_weight},
            )
        self.coef_ = result.x
        self.n_iter_ = result.nfev

        self.lod_ = limit_of_detection(blank, self, lod_sds=self.lod_sds)
        return self

    def signal(self, X):
        """Predict the signal (e.g., AEB) for given concentrations.

        Parameters
        ----------
        X : array-like
            Input data of concentrations.

        Returns
        -------
        signal : array-like
            Predicted signal values.
        """
        return self._model.func(X, *self.coef_)

    def conc(self, y):
        """Estimate the concentration for given signal values.

        Parameters
        ----------
        y : array-like
            Signal values.

        Returns
        -------
        conc : array-like
            Estimated concentration values.
        """
        return self._model.inverse(y, *self.coef_)

    def predict(self, X):
        """Predict signal using the calibration curve model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input data of concentrations.

        Returns
        -------
        y : ndarray
            Predicted signal values.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X=X, reset=False, force_all_finite=False, ensure_2d=self.ensure_2d
        )
        if np.ndim(X) == 1:
            return self.signal(X)
        else:
            return self.signal(X[:, 0])

    def transform(self, X):
        """Transform concentrations into signal.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input data of concentrations.

        Returns
        -------
        y : ndarray
            Transformed signal values.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X=X, reset=False, force_all_finite=False, ensure_2d=self.ensure_2d
        )
        # if self.ensure_2d:
        #     X = X[:, 0].reshape(-1, 1)
        return self.signal(X)

    def inverse_transform(self, X):
        """Estimate the concentration for given signal values.

        Parameters
        ----------
        y : array-like of shape (n_samples, n_features) or (n_samples,)
            Signal values.

        Returns
        -------
        conc : array-like
            Back-calculated concentration values.
        """
        # (n_samples, n_features) -> (n_samples, n_features)
        check_is_fitted(self)
        X = self._validate_data(
            X=X, reset=False, force_all_finite=False, ensure_2d=self.ensure_2d
        )
        return self.conc(X)

    def _make_curve_points(self, **kwargs):
        """Make the points to interpolate for plotting a calibration curve.

        Parameters
        ----------
        cc : CalCurve
            Fitted calibration curve.

        Returns
        -------
        X_curve, y_curve
            Points for plotting the calibration curve.

        Other Parameters
        ----------------
        xscale : str or matplotlib.scale.ScaleBase, optional
            Override the calibration curve's x-scale.
        num : int, default 50
            Number of samples to generate. Default is 50. Must be
            non-negative.

        """
        check_is_fitted(self)
        xscale = kwargs.get("xscale", self._model.xscale)
        if not isinstance(xscale, ScaleBase):
            xscale = scale_factory(
                scale=xscale,
                axis=None,  # type: ignore
                **match_kwargs(scale_factory, kwargs),  # type: ignore
            )

        trans = xscale.get_transform()
        vmin = np.amin(self.X_)
        vmax = np.amax(self.X_)
        minpos = np.amin(self.X_[self.X_ > 0])
        if np.isfinite(self.lod_):
            vmin = min(vmin, self.lod_)
            minpos = min(minpos, self.lod_)
        vmin, vmax = xscale.limit_range_for_scale(vmin=vmin, vmax=vmax, minpos=minpos)
        trvmin = trans.transform(vmin)
        trvmax = trans.transform(vmax)
        start = 1.1 * trvmin - 0.1 * trvmax
        stop = 1.1 * trvmax - 0.1 * trvmin
        X_curve = trans.inverted().transform(
            np.linspace(start=start, stop=stop, num=kwargs.get("num", 50))
        )
        y_curve = self.signal(X_curve)
        self._curve_points = (X_curve, y_curve)
        calculated = {
            "xscale": xscale,
            "trans": trans,
            "vmin": vmin,
            "vmax": vmax,
            "trvmin": trvmin,
            "trvmax": trvmax,
            "start": start,
            "stop": stop,
            "X_curve": X_curve,
            "y_curve": y_curve,
        }
        return calculated

    def plot(
        self,
        fig: Optional[Figure | SubFigure] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        """Plot the calibration curve and calibrator points.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure object. If None, the current figure will be
            used. If there is no current figure, a new figure will be
            created.
        ax : matplotlib.axes.Axes, optional
            The axes object. If None, the current axes will be used. If
            there is no current axes, a new axes will be created.
        **kwargs
            Additional keyword arguments for customizing the plot, such
            as `xlabel`, `ylabel`, and `title`.

        """
        check_is_fitted(self)
        self._make_curve_points(**kwargs)

        fig, ax = get_fig_ax(fig, ax)
        X_curve, y_curve = self._curve_points

        lod_kwargs = {
            "x": self.lod_,
            "color": "tab:blue",
            "linestyle": "--",
            "alpha": 0.5,
            "label": f"limit of detection: {self.lod_:.2g}",
        } | match_kwargs("lod_", kwargs)
        if np.isfinite(lod_kwargs["x"]):
            ax.axvline(lod_kwargs.pop("x"), **lod_kwargs)

        curve_kwargs = {
            "x": X_curve,
            "y": y_curve,
            "color": "tab:green",
            "label": f"{self._model.name} calibration curve",
        } | match_kwargs("curve_", kwargs)
        ax.plot(curve_kwargs.pop("x"), curve_kwargs.pop("y"), **curve_kwargs)

        point_kwargs = {
            "x": self.X_,
            "y": self.y_,
            "color": "black",
            "linestyle": "",
            "marker": "o",
            "alpha": 0.5,
            "label": "calibrator points",
        } | match_kwargs("point_", kwargs)
        ax.plot(point_kwargs.pop("x"), point_kwargs.pop("y"), **point_kwargs)

        ax_kwargs = (
            {
                "xscale": self._model.xscale,
                "yscale": self._model.yscale,
            }
            | match_kwargs(ax.set, kwargs)
            | match_kwargs("ax_", kwargs)
        )
        ax.set(**ax_kwargs)

        legend_kwargs = match_kwargs("legend_", kwargs)
        ax.legend(**legend_kwargs)

    def _more_tags(self):
        if self.ensure_2d:
            return {"allow_nan": True, "poor_score": True, "requires_positive_X": True}
        else:
            return {
                "allow_nan": True,
                "requires_positive_X": True,
                "X_types": ["2darray", "1darray"],
            }

    # Deprecated methods
    @deprecate()
    def bound_lod(self, x_flat):
        return np.maximum(x_flat, self.lod_)

    @deprecate("CalCurve.signal, CalCurve.predict, or CalCurve.transform")
    def fun(self, x):
        try:
            y = self._model.func(self.bound_lod(x), **self.coef_)
        except TypeError:
            x_flat = np.array(x)
            y = self._model.func(self.bound_lod(x_flat), **self.coef_)
        return y

    @deprecate("CalCurve.conc or CalCurve.inverse_transform")
    def inverse(self, y):
        try:
            x = self.bound_lod(self._model.inverse(y, **self.coef_))
        except TypeError:
            y_flat = np.array(y)
            x = self.bound_lod(self._model.inverse(y_flat, **self.coef_))
        return x

    @deprecate("CalCurve.fit")
    @classmethod
    def from_data(
        cls,
        *,
        x,
        y,
        model,
        lod_sds=3,
        corr="c4",
        force_lod: bool = False,
        weights="1/y^2",
        **kwargs,
    ):
        cc = cls(model=model, lod_sds=lod_sds)
        return cc.fit(x, y, sample_weight=weights)

    @deprecate()
    @classmethod
    def from_function(
        cls,
        fun,
        inverse,
        lod=-np.inf,
        lod_sds=3,
        force_lod: bool = False,
        xscale="linear",
        yscale="linear",
    ):
        pass


@deprecate("CalCurve.fit")
def regress(*, x, y, model, weights="1/y^2", **kwargs):
    named_model = model if isinstance(model, Model) else MODELS[model]
    sigma = None
    if isinstance(weights, str):
        if weights == "1/y^2":
            sigma = y
        if weights == "1/y":
            sigma = np.sqrt(np.array(y))
        elif weights == "1":
            sigma = np.ones(len(y))
    if sigma is None:
        sigma = flatten(weights) ** -2
    calibration_function = named_model.func
    xdata, ydata, sigma = dropna([x, y, sigma])
    return curve_fit(
        f=calibration_function, xdata=xdata, ydata=ydata, sigma=sigma, **kwargs
    )[0]


def limit_of_detection(
    blank: Any,
    inverse: CalCurve | Model | Callable | str,
    lod_sds: float = 3,
    corr: str = "c4",
    coef=None,
    nan_policy="omit",
    **kwargs,
):
    """Computes the limit of detection (LOD).

    Parameters
    ----------
    blank : array-like
        Signal (e.g., average number of enzymes per bead, AEB) of the
        zero calibrator. Must have at least two elements.
    inverse_fun : ``function`` or ``CalCurve``
        The functional form used for the calibration curve. If a
        function, it should accept the measurement reading (`y`, e.g.,
        fluorescence) as its only argument and return the value (`x`,
        e.g., concentration). If **inverse_fun** is a ``CalCurve``
        object, the LOD will be calculated from its ``inverse`` method.
    sds : numeric, optional
        How many standard deviations above the mean should the
        background should the limit of detection be calculated at?
        Common values include 2.5 (Quanterix), 3 (Walt Lab), and 10
        (lower limit of quantification, LLOQ).
    corr : {"n", "n-1", "n-1.5", "c4"} or numeric, default "c4"
        The sample standard deviation under-estimates the population
        standard deviation for a normally distributed variable.
        Specifies how this should be addressed. Options:

            - "n" : Divide by the number of samples to yield the
              uncorrected sample standard deviation.

            - "n-1" : Divide by the number of samples minus one to
              yield the square root of the unbiased sample variance.

            - "n-1.5" : Divide by the number of samples minus 1.5 to
              yield the approximate unbiased sample standard deviation.

            - "c4" : Divide by the correction factor to yield the
              exact unbiased sample standard deviation.

            - If numeric, gives the delta degrees of freedom.

    Returns
    -------
    numeric
        The limit of detection, in units of x (e.g., concentration).

    See Also
    --------
    c4 : factor `c4` for unbiased estimation of the standard deviation

    std : unbiased estimate of the population standard deviation

    numpy.std : standard deviation

    """
    blank = np.ravel(np.asarray(blank))

    if nan_policy == "omit":
        blank = blank[~np.isnan(blank)]

    if len(blank) < 2:  # noqa: PLR2004
        # warnings.warn(f"limit_of_detection undefined for {len(blank)} replicates")
        return np.nan

    lod_signal = lod_sds * std(blank, corr=corr) + np.mean(blank)

    if isinstance(inverse, CalCurve):
        f = inverse.conc
    elif isinstance(inverse, Model):
        f = inverse.inverse
    elif callable(inverse):
        f = inverse
    else:
        f = _get_value_or_key(MODELS, inverse, (CalCurve, Model, Callable)).inverse

    if coef is None:
        return f(lod_signal, **kwargs)  # kwargs can be provided or not
    elif not kwargs:
        return f(lod_signal, *coef)  # only if coef is provided
    else:
        raise ValueError("Coefficients can be given in coef or kwargs, but not both.")


class _CalCurveSeries(pd.Series):
    """Subclass of pandas.Series for when entries are CalCurves."""

    @property
    def _constructor(self):
        return _CalCurveSeries

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return self.__class__(result) if isinstance(result, pd.Series) else result

    # def subplots(
    #     self, nrows: int = 1, ncols: int = 1, **kwargs
    # ) -> tuple[Figure, np.ndarray]:
    #     """Create a figure and axes grid.

    #     Parameters
    #     ----------
    #     nrows, ncols : int, default 1
    #         Number of rows/columns of the subplot grid.

    #     **kwargs
    #         Additional keyword arguments passed to `plt.subplots()`.

    #     Returns
    #     -------
    #     fig : matplotlib.figure.Figure
    #         The created figure object.
    #     axs : matplotlib.axes.Axes or array of Axes
    #         ax can be either a single Axes object, or an array of Axes
    #         objects if more than one subplot was created.

    #     """
    #     figsize = (
    #         ncols * rcParams["figure.figsize"][0],
    #         nrows * rcParams["figure.figsize"][1],
    #     )
    #     return plt.subplots(
    #         nrows=nrows,
    #         ncols=ncols,
    #         figsize=figsize,
    #         squeeze=False,
    #         **match_kwargs(plt.subplots, kwargs),
    #     )

    def plot(  # type: ignore
        self,
        *args,
        **kwargs,
    ) -> Axes | np.ndarray:
        """Plot the calibration curve and calibrator points.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure object. If None, the current figure will be
            used. If there is no current figure, a new figure will be
            created.
        ax : matplotlib.axes.Axes, optional
            The axes object. If None, the current axes will be used. If
            there is no current axes, a new axes will be created.
        **kwargs
            Additional keyword arguments for customizing the plot, such
            as `xlabel`, `ylabel`, and `title`.

        """
        fig = kwargs.get("fig")
        ax = kwargs.get("ax")
        max_cols = kwargs.get("max_cols", 5)

        levels = [
            i
            for i in range(self.index.nlevels)
            if self.index.get_level_values(i).nunique(dropna=False) > 1
        ]

        if len(levels) > 1:
            cols = self.index.get_level_values(levels[0]).unique()
            ncols = self.index.get_level_values(levels[0]).nunique()
            nrows = self.index.get_level_values(0).value_counts().max()
            fig, axs = subplots(nrows=nrows, ncols=ncols, fig=fig, **kwargs)

            ax_s = pd.Series(index=self.index)
            for c, col in enumerate(cols):
                # idx = sorted(self.index[self.index.get_level_values(levels[0]) == col])
                # ax_s[col] = dict(zip(idx, axs[:, c]))
                ax_s[col] = axs[: len(ax_s[col]), c]
        else:
            nrows = int(np.ceil(len(self) / max_cols))
            ncols = int(np.ceil(len(self) / nrows))
            fig, axs = subplots(nrows=nrows, ncols=ncols, fig=fig, **kwargs)
            ax_s = pd.Series(dict(zip(self.index, np.ravel(axs))))

        for i, ax in ax_s.items():
            title = "\n".join(str(a) for a in i) if isinstance(i, tuple) else i
            self[i].plot(ax=ax, title=title, **match_kwargs(self[i].plot, kwargs))

        return ax_s.item() if len(ax_s) == 1 else axs  # type: ignore
        # self_notna = self[self.notna()]

        # if len(self_notna) == 0:
        #     # empty series
        #     ax_map = {}

        # elif len(self_notna) == 1:
        #     # only one CalCurve in series
        #     if ax is not None and isinstance(ax, Axes):
        #         if fig is None:
        #             fig = ax.figure
        #     elif fig is not None:
        #         ax = fig.subplots(**match_kwargs(plt.subplots, kwargs))
        #     else:
        #         fig, ax = self.subplots(**kwargs)
        #     ax_map = {self.index[0]: ax}

        # elif self.index.nlevels == 1:
        #     # only one assay or only one plex
        #     if len(self) <= max_cols:
        #         fig, axs = self.subplots(ncols=len(self_notna), **kwargs)
        #         ax_map = {self_notna.index[i]: axs[i] for i in range(len(self_notna))}
        #     else:
        #         nrows = int(np.ceil(len(self) / max_cols))
        #         fig, axs = self.subplots(nrows=nrows, ncols=max_cols, **kwargs)
        #         ax_map = {
        #             self_notna.index[i]: axs[divmod(i, max_cols)]
        #             for i in range(len(self_notna))
        #         }

        # else:
        #     # try to lay out cal curves according to assay and plex
        #     ncols = self.index.get_level_values(0).nunique()
        #     nrows = self.index.get_level_values(0).value_counts().max()
        #     fig, axs = self.subplots(nrows=nrows, ncols=ncols, **kwargs)
        #     ax_map = {}
        #     for a0, assay_level_0 in enumerate(
        #         sorted(self_notna.index.get_level_values(0).unique())
        #     ):
        #         for a1, assay_level_1 in enumerate(
        #             sorted(self_notna[assay_level_0].index)
        #         ):
        #             ax_map[(assay_level_0, assay_level_1)] = axs[a1, a0]

        # for assay, ax in ax_map.items():
        #     if hasattr(assay, "__iter__") and not isinstance(assay, str):
        #         title = "\n".join([str(a) for a in assay])
        #     else:
        #         title = assay
        #     self[assay].plot(
        #         ax=ax, title=title, **match_kwargs(self[assay].plot, kwargs)
        #     )  # type: ignore
        # return

        # # if tight_layout:
        # #     plt.tight_layout()
