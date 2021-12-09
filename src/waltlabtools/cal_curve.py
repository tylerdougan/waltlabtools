"""Class CalCurve, its methods, and related functions.

Everything in waltlabtools.cal_curve is automatically imported with
waltlabtools, so it can be accessed via, e.g.,

.. code-block:: python

   import waltlabtools as wlt  # waltlabtools main functionality

   my_curve = wlt.CalCurve()  # creates a new empty calibration curve


-----


"""

import scipy.optimize as opt

from .core import _optional_dependencies, dropna, flatten, std
from .model import Model, models
from .plot import plot_cal_curve

if _optional_dependencies["jax"]:
    import jax.numpy as np
else:
    import numpy as np


__all__ = ["regress", "limit_of_detection", "CalCurve"]


def _len_for_message(collection) -> str:
    try:
        return str(len(collection))
    except TypeError:
        return "an unknown number of"


def _match_coefs(params, coefs) -> dict:
    if isinstance(coefs, dict):
        if set(coefs.keys()) == set(params):
            return coefs
    elif hasattr(coefs, "__iter__"):
        if len(coefs) == len(params):
            return {params[i]: coefs[i] for i in range(len(coefs))}
    elif len(params) == 1:
        return {params[0]: coefs}
    elif (not params) and (not coefs):
        return {}
    raise ValueError(
        "Wrong number of coefficients. The function requires "
        + _len_for_message(params)
        + " coefficients: "
        + str(params)
        + ". You provided a "
        + str(type(coefs))
        + " with "
        + _len_for_message(coefs)
        + " coefficients: "
        + str(coefs)
        + "."
    )


def regress(model: Model, x, y, weights="1/y^2", **kwargs):
    """Performs a (nonlinear) regression and returns coefficients.

    Parameters
    ----------
    model : waltlabtools.Model or str
        The functional model to use. Should be a valid
        waltlabtools.Model object or a string referring to a built-in
        Model.
    x : array-like
        The independent variable, e.g., concentration.
    y : array-like
        The dependent variable, e.g., fluorescence.
    weights : ``str`` or array-like, default "1/y^2"
        Weights to be used. If array-like, should be the same size as
        **x** and **y**. Otherwise, can be one of the following:

            - "1/y^2" : Inverse-squared (1/y^2) weighting.

            - "1/y" : Inverse (1/y) weighting.

            - "1" : Equal weighting for all data points.

        Other strings raise a ``NotImplementedError``.
    kwargs
        Keyword arguments passed to scipy.optimize.curve_fit.

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the
        squared residuals is minimized.

    See Also
    --------
    scipy.optimize.curve_fit : backend function used by ``regress``

    """
    named_model = model if isinstance(model, Model) else models[model]
    sigma = None
    if isinstance(weights, str):
        if weights == "1/y^2":
            sigma = y
        if weights == "1/y":
            sigma = np.sqrt(np.array(y))
        elif weights == "1":
            sigma = np.ones(len(y))
    if sigma is None:
        sigma = flatten(weights, "error") ** -2
    calibration_function = named_model.fun
    xdata, ydata, sigma = dropna([x, y, sigma])
    return opt.curve_fit(
        f=calibration_function, xdata=xdata, ydata=ydata, sigma=sigma, **kwargs
    )[0]


def limit_of_detection(blank_signal, inverse_fun=None, sds=3, corr="c4"):
    """
    Compute the limit of detection (LOD).

    Parameters
    ----------
    blank_signal : array-like
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
    lod_x : numeric
        The limit of detection, in units of x (e.g., concentration).

    See Also
    --------
    c4 : factor `c4` for unbiased estimation of the standard deviation

    std : unbiased estimate of the population standard deviation

    numpy.std : standard deviation

    """
    blank_array = flatten(blank_signal)
    mean = np.mean(blank_array)
    stdev = std(blank_array, corr=corr)
    lod_y = mean + sds * stdev
    if isinstance(inverse_fun, CalCurve):
        lod_x = inverse_fun.inverse(lod_y)
    else:
        lod_x = inverse_fun(lod_y)
    return lod_x


# CLASSES


class CalCurve:
    """Calibration curve.

    A calibration curve is the result of regressing the calibrator data
    with a specific functional form.

    Parameters
    ----------
    model : ``Model`` or ``str``
        The functional model to use. Should be a valid ``Model`` object
        or a string referring to a built-in ``Model``.
    coefs : list-like
        Numerical values of the parameters specified by **model**.
    lod : numeric, optional
        Lower limit of detection (LOD).
    lod_sds : numeric, default 3
        Number of standard deviations above blank at which the lower
        limit of detection is calculated. Common values include 2.5
        (Quanterix), 3 (Walt Lab), and 10 (lower limit of
        quantification, LLOQ).
    force_lod : ``bool``, default False
        Should readings below the LOD be set to the LOD?

    """

    def __init__(
        self,
        model=None,
        x=None,
        y=None,
        coefs=(),
        lod=-np.inf,
        lod_sds=3,
        force_lod: bool = False
    ):
        """Initializes a new, empty CalCurve object."""
        self.x = flatten(x)
        self.y = flatten(y)
        self.model = model if isinstance(model, Model) else models[model]
        self.coefs = _match_coefs(self.model.params, coefs)
        self.lod = lod
        self.lod_sds = lod_sds
        self.force_lod = force_lod

    def __repr__(self) -> str:
        """Print/represent method for CalCurve objects."""
        coefs_named = "".join(
            param + " = " + str(coef) + ", " for param, coef in self.coefs.items()
        )

        return (
            self.model.name
            + " calibration curve with parameters "
            + coefs_named
            + "and LOD = "
            + str(self.lod)
        )

    def bound_lod(self, x_flat):
        """Sets values below the limit of detection (LOD) to the LOD.

        If ``force_lod`` is True, returns a version of the given data
        with all values below the LOD set to be the LOD. Otherwise,
        returns the original data.

        Parameters
        ----------
        x_flat : array
            Data to be bounded. Must be an array, such as the output of
            ``flatten``.

        Returns
        -------
        array
            A copy of **x_flat**, with all its values above the LOD if
            **force_lod** is True, or the original array otherwise.

        """
        if self.force_lod:
            return np.maximum(x_flat, self.lod)
        else:
            return x_flat

    def fun(self, x):
        """Forward function, mapping values to measurement readings.

        Use ``fun`` to convert values (e.g., concentration) to the
        measurement readings (e.g., fluorescence) that they should
        yield.

        Parameters
        ----------
        x : numeric or array-like
            Values, such as concentration.

        Returns
        -------
        y : same as input or array
            Measurement readings, such as fluorescence, calculated from
            the values **x** using the calibration curve. If possible,
            **y** will be the same type, size, and shape as **x**; if
            not, **y** will be a 1D array of the same size as **x**.

        """
        try:
            y = self.model.fun(self.bound_lod(x), **self.coefs)
        except TypeError:
            x_flat = flatten(x)
            y = self.model.fun(self.bound_lod(x_flat), **self.coefs)
        return y

    def inverse(self, y):
        """Inverse function, mapping measurement readings to values.

        Use ``inverse`` to convert measurement readings (e.g.,
        fluorescence) to values (e.g., concentration) of the sample.

        Parameters
        ----------
        y : numeric or array-like
            Measurement readings, such as fluorescence.

        Returns
        -------
        x : same as input or array
            Values, such as concentration, calculated from
            the values **y** using the calibration curve. If possible,
            **x** will be the same type, size, and shape as **y**; if
            not, **x** will be a 1D array of the same size as **y**.

        """
        try:
            x = self.bound_lod(self.model.inverse(y, **self.coefs))
        except TypeError:
            y_flat = flatten(y)
            x = self.bound_lod(self.model.inverse(y_flat, **self.coefs))
        return x

    def plot(
        self,
        ax=None,
        fig=None,
        show: bool = True,
        hide=(),
        point_color=None,
        curve_color=None,
        lod_color=None,
        **kwargs
    ):
        """Plots a CalCurve object.

        See documentation at waltlabtools.plot.plot_cal_curve().

        """
        return plot_cal_curve(
            self,
            ax=ax,
            fig=fig,
            show=show,
            hide=hide,
            point_color=point_color,
            curve_color=curve_color,
            lod_color=lod_color,
            **kwargs
        )

    def __iter__(self):
        return zip(self.x, self.y)

    @classmethod
    def from_data(
        cls,
        model,
        x,
        y,
        lod_sds=3,
        corr="c4",
        force_lod: bool = False,
        weights="1/y^2",
        **kwargs
    ):
        """Constructs a calibration curve from data.

        Parameters
        ----------
        x : array-like
            The independent variable, e.g., concentration.
        y : array-like
            The dependent variable, e.g., fluorescence.
        model : ``Model``
            Mathematical model used.
        lod : numeric, optinal
            Lower limit of detection (LOD).
        lod_sds : numeric, default 3
            Number of standard deviations above blank at which the lower
            limit of detection is calculated. CCommon values include 2.5
            (Quanterix), 3 (Walt Lab), and 10 (lower limit of
            quantification, LLOQ).
        corr : {"n", "n-1", "n-1.5", "c4"} or numeric, default "c4"
            The sample standard deviation under-estimates the population
            standard deviation for a normally distributed variable.
            Specifies how this should be addressed. Options:

                - "n" : Divide by the number of samples to yield the
                  uncorrected sample standard deviation.

                - "n-1" : Divide by the number of samples minus one to
                  yield the square root of the unbiased sample variance.

                - "n-1.5" : Divide by the number of samples minus 1.5
                  to yield the approximate unbiased sample standard
                  deviation.

                - "c4" : Divide by the correction factor to yield the
                  exact unbiased sample standard deviation.

                - If numeric, gives the delta degrees of freedom.
        force_lod : ``bool``, default False
            Should readings below the LOD be set to the LOD?
        weights : ``str`` or array-like, default "1/y^2"
            Weights to be used. If array-like, should be the same size
            as **x** and **y**. Otherwise, can be one of the following:

                - "1/y^2" : Inverse-squared (1/y^2) weighting.

                - "1" : Equal weighting for all data points.

            Other strings raise a ``NotImplementedError``.
        **kwargs
            Keyword arguments passed to scipy.optimize.curve_fit.

        Returns
        -------
        ``CalCurve``

        """
        x_flat, y_flat = dropna([x, y])
        coefs = regress(model=model, x=x_flat, y=y_flat, weights=weights, **kwargs)
        cal_curve = cls(
            x=x_flat,
            y=y_flat,
            model=model,
            coefs=coefs,
            lod_sds=lod_sds,
            force_lod=force_lod,
        )
        cal_curve.lod = limit_of_detection(
            y_flat[x_flat == 0], cal_curve.inverse, sds=lod_sds, corr=corr
        )
        return cal_curve

    @classmethod
    def from_function(
        cls,
        fun,
        inverse,
        lod: float = -np.inf,
        lod_sds=3,
        force_lod: bool = False,
        xscale="linear",
        yscale="linear",
    ):
        """Constructs a calibration curve from a function.

        Parameters
        ----------
        fun : ``function``
            Forward function, mapping values to measurement readings.
        inverse : ``function``
            Inverse function, mapping measurement readings to values.
        lod : numeric, optinal
            Lower limit of detection (LOD).
        lod_sds : numeric, default 3
            Number of standard deviations above blank at which the lower
            limit of detection is calculated. Common values include 2.5
            (Quanterix), 3 (Walt Lab), and 10 (lower limit of
            quantification, LLOQ).
        force_lod : ``bool``, default False
            Should readings below the LOD be set to the LOD?
        xscale, yscale : {"linear", "log", "symlog", "logit"} or
        matplotlib.ScaleBase, default "linear"
            The natural scaling transformations for `x` and `y`. For
            example, "log" means that the data may be distributed
            log-normally and are best visualized on a log scale.

        Returns
        -------
        ``CalCurve``

        """
        model = Model(fun=fun, inverse=inverse, xscale=xscale, yscale=yscale)
        return cls(model=model, lod=lod, lod_sds=lod_sds, force_lod=force_lod)
