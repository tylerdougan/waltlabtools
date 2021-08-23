"""
:noindex:

"""

# IMPORT MODULES

import warnings
import sys

if "jax" in sys.modules.keys():
    import jax.numpy as np
    from jax import jit
else:
    import numpy as np
    def jit(fun):
        return fun

import scipy.optimize as opt
import scipy.special as spec

from .nonnumeric import _match_coefs, _error_text


TF_CPP_MIN_LOG_LEVEL = 0


def flatten(data, on_bad_data="warn"):
    """Flattens most data structures.
   
    Parameters
    ----------
    data : any
        The data structure to be flattened. Can also be a primitive.
    on_bad_data : {"error", "ignore", "warn"}, default "warn"
        Specifies what to do when the data cannot be coerced to an
        ndarray. Options are as follows:

            - "error" : Raises TypeError.

            - "ignore" : Returns a list or, failing that, the original
              object.

            - "warn" : Returns as in ``"ignore"``, but raises a warning.
    
    Returns
    -------
    flattened_data : array, list, or primitive
        Flattened version of **data**. If ``on_bad_data="error"``,
        always an array.
    
    """
    try:
        return np.ravel(np.asarray(data))
    except Exception:
        if hasattr(data, "__iter__"):
            flattened_data = []
            for datum in data:
                if hasattr(datum, "__iter__"):
                    flattened_data.extend(flatten(datum))
                else:
                    flattened_data.append(datum)
        else:
            flattened_data = data
        try:
            return np.asarray(flattened_data)
        except Exception:
            if on_bad_data != "ignore":
                error_text = _error_text(
                    [type(data), type(flattened_data)], "coercion")
                if on_bad_data == "error":
                    raise TypeError(error_text)
                else:
                    warnings.warn(error_text, Warning)
            return flattened_data


def aeb(fon):
    """The average number of enzymes per bead.

    Converts the fraction of on-beads (fon) to the average number of
    enzymes per bead (AEB) using Poisson statistics. The formula used
    is `aeb` = -log(1 - `fon`).
   
    Parameters
    ----------
    fon : numeric or array-like
        A scalar or array of fractions of beads which are "on."
   
    Returns
    -------
    aeb : same as input, or array
        The average number of enzymes per bead.

    See Also
    --------
    fon : inverse of ``aeb``
   
    """
    try:
        return -np.log(1 - fon)
    except TypeError:
        return -np.log(1 - flatten(fon))


def fon(aeb):
    """The fraction of beads which are on.

    Converts the average enzymes per bead (AEB) to the fraction of
    on-beads (fon) using Poisson statistics. The formula used is
    `fon` = 1 - exp(-`aeb`).
   
    Parameters
    ----------
    aeb : numeric or array-like
        A scalar or array of the average number of enzymes per bead.
   
    Returns
    -------
    fon : same as input, or array
        The fractions of beads which are "on."

    See Also
    --------
    aeb : inverse of ``fon``
   
    """
    try:
        return 1 - np.exp(-aeb)
    except TypeError:
        return 1 - np.exp(-flatten(aeb))


def c4(n):
    """Factor `c4` for unbiased estimation of the standard deviation.

    For a finite sample, the sample standard deviation tends to
    underestimate the population standard deviation. See, e.g.,
    https://www.spcpress.com/pdf/DJW353.pdf for details. Dividing the
    sample standard deviation by the correction factor `c4` gives an
    unbiased estimator of the population standard deviation.
   
    Parameters
    ----------
    n : numeric or array
        The number of samples.
   
    Returns
    -------
    numeric or array
        The correction factor, usually written `c4` or `b(n)`.

    See Also
    --------
    numpy.std : standard deviation
    
    lod : limit of detection
   
    """
    return np.sqrt(2/(n-1)) * spec.gamma(n/2) / spec.gamma((n-1)/2)


class Model:
    """Mathematical model for calibration curve fitting.

    A **Model** is an object with a function and its inverse, with one
    or more free parameters that can be fit to calibration curve data.
   
    Parameters
    ----------
    fun : ``function``
        Forward functional form. Should be a function which takes in `x`
        and other parameters and returns `y`. The first parameter of
        **fun** should be `x`, and the remaining parameters should be
        the coefficients which are fit to the data (typically floats).
    inverse : ``function``
        Inverse functional form. Should be a function which takes in `y`
        and other parameters and returns `x`. The first parameter of
        **inverse** should be `y`, and the remaining parameters should
        be the same coefficients as in **fun**.
    name : ``str``
        The name of the function. For example, "4PL" or "linear".
    params : list-like of ``str``
        The names of the parameters for the function. This should be
        the same length as the number of arguments which **fun** and
        **inverse** take after their inputs `x` and `y`, respectively.
    xscale, yscale : {"linear", "log", "symlog", "logit"}, default \
    "linear"
        The natural scaling transformations for `x` and `y`. For
        example, "log" means that the data may be distributed
        log-normally and are best visualized on a log scale.
   
    """
    def __init__(self, fun=None, inverse=None, name="", params=(),
            xscale="linear", yscale="linear"):
        self.fun = fun
        self.inverse = inverse
        self.name = name
        self.params = params
        self.xscale = xscale
        self.yscale = yscale

    def __iter__(self):
        return self.params

# CONSTANTS

# linear function
def _f_linear(x, a, b):
    return a*x + b
def _i_linear(y, a, b):
    return (y - b) / a
m_linear = Model(_f_linear, _i_linear, "linear",
    ("a", "b"), "linear", "linear")

# power function
def _f_power(x, a, b):
    return a * x**b
def _i_power(y, a, b):
    return (y / a)**(1/b)
m_power = Model(_f_power, _i_power, "power",
    ("a", "b"), "log", "log")

# Hill function
def _f_hill(x, a, b, c):
    return (a * x**b) / (c**b + x**b)
def _i_hill(y, a, b, c):
    return c * (a/y - 1)**(-1/b)
m_hill = Model(_f_hill, _i_hill, "Hill",
    ("a", "b", "c"), "log", "log")

# logistic function
def _f_logistic(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(-b*(x - c)))
def _i_logistic(y, a, b, c, d):
    return c - np.log((a - d)/(y - d) - 1) / b
m_logistic = Model(_f_linear, _i_logistic, "logistic",
    ("a", "b", "c", "d"), "linear", "linear")

# 4-parameter logistic
def _f_4PL(x, a, b, c, d):
    return d + (a - d)/(1 + (x/c)**b)
def _i_4PL(y, a, b, c, d):
    return c*((a-d)/(y-d) - 1)**(1/b)
m_4PL = Model(_f_4PL, _i_4PL, "4PL",
    ("a", "b", "c", "d"), "log", "log")

# 5-parameter logistic
def _f_5PL(x, a, b, c, d, g):
    return d + (a - d)/(1 + (x/c)**b)**g
def _i_5PL(y, a, b, c, d, g):
    return c*(((a-d)/(y-d))**(1/g) - 1)**(1/b)
m_5PL = Model(_f_5PL, _i_5PL, "5PL",
    ("a", "b", "c", "d", "g"), "log", "log")

model_list = [m_linear, m_power, m_hill, m_logistic, m_4PL, m_5PL]
model_dict = {model.name: model for model in model_list}


# CURVE FITTING

def _match_model(model_name):
    """Returns a ``Model`` object from a string matching its name.
   
    Parameters
    ----------
    model : ``str`` or ``Model``
        Model name or ``Model`` object. Ideally a member of 
        ``model_dict.keys()``, but can also be one with some characters 
        off or different capitalization.
   
    Returns
    -------
    named_model : ``Model``
        Fixed version of **model** which is a built-in ``Model``.
        
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
            elif len(matches) > 1:
                raise KeyError(_error_text([m, str(matches)], "_match_model"))
            else:
                raise KeyError(_error_text([m], "_match_model"))
    return named_model


def regress(model, x, y, use_inverse=False, weights="1/y^2",
            p0=None, bounds=None, method=None):
    """Performs a (nonlinear) regression and return coefficients.

    Parameters
    ----------
    model : ``Model`` or ``str``
        The functional model to use. Should be a valid ``Model`` object
        or a string referring to a built-in ``Model``.
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
    p0 : array-like, optional
        Initial guess for the parameters. If provided, must have the
        same length as the number of parameters. If None, then the
        initial values will all be 1 (if the number of parameters for
        the function can be determined using introspection,
        otherwise a ``ValueError`` is raised).
    bounds : 2-tuple of array-like, optional
        Lower and upper bounds on parameters. Defaults to no bounds.
        Each element of the tuple must be either an array with the
        length equal to the number of parameters, or a scalar (in
        which case the bound is taken to be the same for all
        parameters). Use np.inf with an appropriate sign to disable
        bounds on all or some parameters.
    method : {"lm", "trf", "dogbox"}, optional
        Method to use for optimization. See
        scipy.optimize.least_squares for more details. Default is
        "lm" for unconstrained problems and "trf" if bounds are
        provided. The method "lm" won’t work when the number of
        observations is less than the number of variables; use
        "trf" or "dogbox" in this case.
   
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
    if use_inverse:
        calibration_function = named_model.inverse
        xdata = flatten(y)
        ydata = flatten(x)
    else:
        calibration_function = named_model.fun
        xdata = flatten(x)
        ydata = flatten(y)
    if hasattr(weights, "__iter__"):
        sigma = flatten(weights)**-2
    elif weights == "1/y^2":
        sigma = ydata if not use_inverse else xdata
    elif weights == "1":
        sigma = None
    else:
        raise NotImplementedError(_error_text(
            ["weighting scheme", weights], "implementation"))
    kwargs = dict()
    for kwarg_name, kwarg in zip(
            ["p0", "sigma", "bounds", "method"],
            [p0, sigma, bounds, method]):
        if kwarg is not None:
            kwargs[kwarg_name] = kwarg
    popt, pcov = opt.curve_fit(f=calibration_function,
                               xdata=xdata,
                               ydata=ydata,
                               **kwargs)
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
    try:
        ddof = {"n": 0, "n-1": 1, "n-1.5": 1.5, "c4": 1}[corr]
    except KeyError:
        try:
            ddof = float(corr)
        except ValueError:
            raise ValueError(
                _error_text([corr, "correction factor"], "implementation"))
    corr_factor = c4(len(blank_array)) if (corr == "c4") else 1
    mean = np.mean(blank_array)
    stdev = np.std(blank_array, ddof=ddof)/corr_factor
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
    def __init__(self, model=None, coefs=(), lod=-np.inf, lod_sds=3, 
            force_lod=False):
        self.model = _match_model(model)
        self.coefs = _match_coefs(model.params, coefs)
        self.lod = lod
        self.lod_sds = lod_sds
        self.force_lod = force_lod
        
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

    def __iter__(self):
        return self.coefs

    @classmethod
    def from_data(cls, x, y, model, lod_sds=3, force_lod=False,
            use_inverse=False, weights="1/y^2", p0=None, bounds=None,
            method=None, corr="c4"):
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
        p0 : array-like, optional
            Initial guess for the parameters. If provided, must have the
            same length as the number of parameters. If None, then the
            initial values will all be 1 (if the number of parameters
            for the function can be determined using introspection,
            otherwise a ``ValueError`` is raised).
        bounds : 2-tuple of array-like, optional
            Lower and upper bounds on parameters. Defaults to no bounds.
            Each element of the tuple must be either an array with the
            length equal to the number of parameters, or a scalar (in
            which case the bound is taken to be the same for all
            parameters). Use np.inf with an appropriate sign to disable
            bounds on all or some parameters.
        method : {"lm", "trf", "dogbox"}, optional
            Method to use for optimization. See
            ``scipy.optimize.least_squares`` for more details. Default
            is "lm" for unconstrained problems and "trf" if bounds are
            provided. The method "lm" won’t work when the number of
            observations is less than the number of variables; use
            "trf" or "dogbox" in this case.
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
                
        
        Returns
        -------
        ``CalCurve``

        """
        x_flat = flatten(x, on_bad_data="error")
        y_flat = flatten(y, on_bad_data="error")
        coefs = regress(model=model, x=x_flat, y=y_flat, 
            use_inverse=use_inverse, weights=weights, p0=p0, bounds=bounds,
            method=method)
        cal_curve = cls(model=model, coefs=coefs, lod_sds=lod_sds, 
            force_lod=force_lod)
        cal_curve.lod = lod(y_flat[x_flat == 0], cal_curve.inverse, 
            sds=lod_sds, corr=corr)
        return cal_curve

    @classmethod
    def from_function(cls, fun, inverse, lod=-np.inf, lod_sds=3,
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
        xscale, yscale : {"linear", "log", "symlog", "logit"}, default \
        "linear"
            The natural scaling transformations for `x` and `y`. For
            example, "log" means that the data may be distributed
            log-normally and are best visualized on a log scale.

        Returns
        -------
        ``CalCurve``

        """
        model = Model(fun=fun, inverse=inverse, 
            xscale=xscale, yscale=yscale)
        return cls(model=model, lod=lod, lod_sds=lod_sds, force_lod=force_lod)


def gmnd(data):
    """Geometric meandian.

    For details, see https://xkcd.com/2435/. This function compares
    the three most common measures of central tendency for a given
    dataset: the arithmetic mean, the geometric mean, and the median.

    Parameters
    ----------
    data : array-like
        The data for which to take the measure of central tendency.
   
    Returns
    -------
    central_tendencies : ``dict`` of ``str`` -> numeric
        The measures of central tendency, ordered by their distance
        from the geometric meandian. Its keys are:

            - "gmnd" : geometric meandian (always first)

            - "arithmetic" : arithmetic mean

            - "geometric" : geometric mean

            - "median" : median

    """
    flat_data = flatten(data)
    data_amin = np.amin(flat_data)
    if not data_amin > 0:
        warnings.warn(_error_text([data_amin], "nonpositive"))
    mean_ = np.nanmean(flat_data)
    geomean_ = np.exp(np.nanmean(np.log(flat_data)))
    median_ = np.nanmedian(flat_data)
    data_i = np.asarray((mean_, geomean_, median_))
    converged = False
    while not converged:
        data_i, converged = _gmnd_f(data_i)
    gmnd_ = data_i[0]
    avgs = np.asarray([gmnd_, mean_, geomean_, median_])
    errors = abs(np.repeat(gmnd_, 4) - avgs) - np.asarray([1, 0, 0, 0])
    named_errors = sorted(zip(
        errors, ["gmnd", "arithmetic", "geometric", "median"], avgs))
    central_tendencies = {ne[1]: ne[2] for ne in named_errors}
    return central_tendencies


@jit
def _gmnd_f(data_i):
    """Backend for geometric meandian.

    Parameters
    ----------
    data_i : array of length 3
        The current iteration's arithmetic mean, geometric mean, and
        median.

    Returns
    -------
    data_iplus1 : array of length 3
        The next iteration's arithmetic mean, geometric mean, and
        median.
   
    """
    mean_ = np.nanmean(data_i)
    geomean_ = np.exp(np.mean(np.log(data_i)))
    median_ = np.nanmedian(data_i)
    data_iplus1 = np.asarray((mean_, geomean_, median_))
    converged = np.all(data_iplus1 == data_i)
    return data_iplus1, converged
