"""Class CalCurve, its methods, and related functions.

Everything in waltlabtools.cal_curve is automatically imported with
waltlabtools, so it can be accessed via, e.g.,

.. code-block:: python

   import waltlabtools as wlt  # waltlabtools main functionality

   my_cal_curve = wlt.CalCurve()  # creates a new empty calibration curve


-----


"""

import scipy.optimize as opt

from .core import dropna, _optional_dependencies, std
from .model import Model, model_dict

if _optional_dependencies["jax"]:
    import jax.numpy as np
else:
    import numpy as np

if _optional_dependencies["matplotlib"]:
    from .plot import plot_cal_curve
else:
    def plot_cal_curve(*args, **kwargs):
        raise ModuleNotFoundError(
            "Plotting requires matplotlib to be installed.")


__all__ = ["regress", "lod", "CalCurve"]


def _len_for_message(collection) -> str:
    try:
        return str(len(collection))
    except Exception:
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
    raise ValueError(" ".join(["Wrong number of coefficients.",
            "The function requires", _len_for_message(params), "coefficients:",
            str(params), ". You provided a", str(type(coefs)), "with",
            _len_for_message(coefs), "coefficients:", str(coefs)]))


def _match_model(model_name) -> Model:
    """Returns a Model object from a string matching its name.

    Parameters
    ----------
    model : str or waltlabtools.Model
        Model name or waltlabtools.Model object. Ideally a member of
        model_dict.keys(), but can also be one with some characters
        off or different capitalization.

    Returns
    -------
    named_model : waltlabtools.Model
        Fixed version of model which is a built-in
        waltlabtools.Model.

    """
    if isinstance(model_name, Model):
        named_model = model_name
    elif model_name in model_dict.keys():
        named_model = model_dict[model_name]
    else:
        m = str(model_name)
        if m in model_dict.keys():
            named_model = model_dict[m]
        else:
            m_lower = m.casefold()
            matches = []
            for key in model_dict.keys():
                if (m_lower in key.casefold()) or (key.casefold() in m_lower):
                    matches.append(key)
            if len(matches) == 1:
                named_model = model_dict[matches[0]]
            else:
                error_text = " ".join(["Model", model_name, "not found."])
                if len(matches) > 1:
                    error_text = " ".join([error_text, "Did you mean one of",
                        str(matches), "?"])
                raise KeyError(error_text)
    return named_model


def regress(model, x, y, use_inverse: bool = False, weights="1/y^2", **kwargs):
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
    use_inverse : ``bool``, default False
        Should ``x`` be regressed as a function of ``y`` instead?
    weights : ``str`` or array-like, default "1/y^2"
        Weights to be used. If array-like, should be the same size as
        **x** and **y**. Otherwise, can be one of the following:

            - "1/y^2" : Inverse-squared (1/y^2) weighting.

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
    named_model = _match_model(model)
    if weights == "1/y^2":
        sigma = y
    elif (weights == "1"):
        sigma = np.ones(len(y))
    else:
        try:
            sigma = flatten(weights)**-2
        except Exception:
            raise NotImplementedError(str(weights)
                + " is not a valid weighting scheme. Please try another one.")
    if use_inverse:
        calibration_function = named_model.inverse
        xdata, ydata = dropna([y, x, sigma])
    else:
        calibration_function = named_model.fun
        xdata, ydata = dropna([x, y, sigma])
    kwargs = dict()
    for kwarg_name, kwarg in zip(
            ["p0", "sigma", "bounds", "method"],
            [p0, sigma, bounds, method]):
        if kwarg is not None:
            kwargs[kwarg_name] = kwarg
    popt, pcov = opt.curve_fit(f=calibration_function, xdata=xdata,
        ydata=ydata, sigma=sigma, **kwargs)
    return popt


def lod(blank_signal, inverse_fun=None, sds=3, corr="c4"):
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
    c4 : unbiased estimation of the population standard deviation

    numpy.std : standard deviation

    """
    blank_array = flatten(blank_signal)
    mean = np.mean(blank_array)
    stdev = std(blank_array)
    lod_y = mean + sds*stdev
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

    def __init__(self, model=None, x=None, y=None, coefs=(), lod=-np.inf,
            lod_sds=3, force_lod=False):
        """Initializes a new, empty CalCurve object."""
        self.x = flatten(x)
        self.y = flatten(y)
        self.model = _match_model(model)
        self.coefs = _match_coefs(self.model.params, coefs)
        self.lod = lod
        self.lod_sds = lod_sds
        self.force_lod = force_lod

    def __repr__(self):
        """Print/represent method for CalCurve objects."""
        coefs_named = "".join(
            ["".join([self.model.params[i], " = ", str(self.coefs[i]), ", "])
                for i in range(len(self.coefs))])
        return "".join([self.model.name, " calibration curve with parameters ",
            coefs_named, "and LOD = ", str(self.lod)])

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
        except Exception:
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
        except Exception:
            y_flat = flatten(y)
            x = self.bound_lod(self.model.inverse(y_flat, **self.coefs))
        return x

    def plot(self, point_color=None, curve_color=None, x=None, start=None,
            stop=None, num: int = 1000, plot_points_with=None,
            show: bool = True, **kwargs):
        return plot_cal_curve(self, point_color curve_color, x, start, stop,
            num, plot_points_with, show, **kwargs)

    def __iter__(self):
        return zip(self.x, self.y)

    @classmethod
    def from_data(cls, model, x, y, lod_sds=3, corr="c4",
            force_lod: bool = False, use_inverse: bool = False,
            weights="1/y^2", corr="c4"):
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
        use_inverse : ``bool``, default False
            Should **x** be regressed as a function of **y** instead?
        weights : ``str`` or array-like, default "1/y^2"
            Weights to be used. If array-like, should be the same size
            as **x** and **y**. Otherwise, can be one of the following:

                - "1/y^2" : Inverse-squared (1/y^2) weighting.

                - "1" : Equal weighting for all data points.

            Other strings raise a ``NotImplementedError``.
        kwargs
            Keyword arguments passed to scipy.optimize.curve_fit.

        Returns
        -------
        ``CalCurve``

        """
        x_flat = flatten(x)
        y_flat = flatten(y)
        coefs = regress(model=model, x=x_flat, y=y_flat,
            use_inverse=use_inverse, weights=weights, **kwargs)
        cal_curve = cls(x=x_flat, y=y_flat, model=model, coefs=coefs,
            lod_sds=lod_sds, force_lod=force_lod)
        cal_curve.lod = lod(y_flat[x_flat == 0], cal_curve.inverse,
            sds=lod_sds, corr=corr)
        return cal_curve

    @classmethod
    def from_function(cls, fun, inverse, lod: float = -np.inf, lod_sds=3,
            force_lod=False, xscale="linear", yscale="linear"):
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
        xscale, yscale : {"linear", "log", "symlog", "logit"}, default "linear"
            The natural scaling transformations for `x` and `y`. For
            example, "log" means that the data may be distributed
            log-normally and are best visualized on a log scale.

        Returns
        -------
        ``CalCurve``

        """
        model = Model(fun=fun, inverse=inverse, xscale=xscale, yscale=yscale)
        return cls(model=model, lod=lod, lod_sds=lod_sds, force_lod=force_lod)
