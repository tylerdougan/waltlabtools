# IMPORT MODULES

import warnings

try:
    import jax.numpy as np
    print("Loaded waltlabtools using jax.")
except Exception:
    import numpy as np
    print("Loaded waltlabtools using numpy.")

import scipy.optimize as opt
import scipy.special as spec

from backend import isiterable, _match_coefs, _error_text


def flatten(data, on_bad_data="warn"):
    """
    Flattens most data structures. Will flatten to a 1D
    ndarray if possible, or otherwise to a list of primitives.
   
    Parameters:
    ----------
    `data` : any
        The data structure to be flattened. Can also be a primitive.
    `on_bad_data` : `{"error", "ignore", "warn"}`, optional
        Specifies what to do when the data cannot be coerced to an
        ndarray. Options are as follows:
            - `"error"` : Raises a TypeError if data cannot be coerced
            to an ndarray.
            - `"ignore"` : Returns a list or, failing that, the original
            object if data cannot be coerced to an ndarray.
            - `"warn"` : Returns as in `"ignore"`, but raises a
            warning if data were not coerced to an ndarray.
        Defaults to `"warn"`.
   
    Returns
    -------
    `flattened_array` : 1D ndarray, list, or primitive
        Flattened version of `data`. If `on_bad_data="error"`, always
        an ndarray.
   
    """
    try:
        return np.ravel(np.asarray(data))
    except Exception:
        if isiterable(data):
            flattened_data = []
            if hasattr(data, "iteritems"):
                iterator = data.iteritems()
            else:
                iterator = enumerate(data)
            for __, datum in iterator:
                if isiterable(datum):
                    flattened_data.extend(flatten(datum))
                else:
                    flattened_data.append(datum)
        else:
            flattened_data = data
        try:
            return np.asarray(flattened_data)
        except Exception:
            if on_bad_data == "error":
                raise TypeError(_error_text(
                    [type(data), type(flattened_data)], "coercion"))
            elif on_bad_data == "ignore":
                return flattened_data
            else:
                warnings.warn(_error_text(
                    [type(data), type(flattened_data)], "coercion"), Warning)
                return flattened_data


def aeb(fon):
    """
    Converts the fraction of beads which are "on" (F_on) to the average
    number of enzymes per bead (AEB) using Poisson statistics. Inverse
    of `fon`.
   
    Parameters:
    ----------
    `fon` : numeric or array-like
        A scalar or array of fractions of beads which are "on."
   
    Returns
    -------
    `aeb` : same as input
        The average number of enzymes per bead.
   
    """
    return -np.log(1 - fon)


def fon(aeb):
    """
    Converts the average number of enzymes per bead (AEB) to the the
    fraction of beads which are "on" (F_on) using Poisson statistics.
    Inverse of `aeb`.
   
    Parameters:
    ----------
    `aeb` : numeric or array-like
        A scalar or array of the average number of enzymes per bead.
   
    Returns
    -------
    `aeb` : same as input
        The fractions of beads which are "on."
   
    """
    return 1 - np.exp(-aeb)


def c4(n):
    """
    Returns the factor c4 which corrects for the bias in estimating the
    standard deviation of a population. Without this correction, it
    may erroneously appear that adding more samples increases the
    standard deviation.
   
    Parameters:
    ----------
    `n` : numeric or array-like
        The number of samples.
   
    Returns
    -------
    `corr_factor` : same as input
        The correction factor, usually written c4 or b(n).
   
    """
    try:
        corr_factor = np.sqrt(2/(n-1)) * spec.gamma(n/2) / spec.gamma((n-1)/2)
    except NameError:
        warnings.warn("Gamma function not available. Using n-1.5 instead.")
        corr_factor = np.sqrt((n-1.5) / (n-1))
    return corr_factor


class Model:
    """
    Mathematical model for calibration curve fitting.
   
    Parameters:
    ----------
    `fun` : function
        Forward functional form. Should be a function which takes in
        `x` and other parameters and returns `y`.
    `inverse` : function
        Inverse functional form. Should be a function which takes in
        `y` and other parameters and returns `x`.
    `name` : str
        The name of the function. For example, "4PL" or "linear".
    `params` : list-like of str
        The names of the parameters for the function. This should be
        the same length as the number of arguments which `fun` and
        `inverse` take after their inputs `x` and `y`, respectively.
    `xscale`, `yscale` : {"linear", "log", "symlog", "logit"}
        The natural scaling transformations for x and y. For example,
        "log" means that the data may be distributed log-normally and
        be best visualized on a log scale. Defaults to "linear".
   
    """
    def __init__(self, fun=None, inverse=None, name="", params=(),
            xscale="linear", yscale="linear"):
        self.fun = fun
        self.inverse = inverse
        self.name = name
        self.params = params
        self.xscale = xscale
        self.yscale = yscale
#        self.grad = grad(self.fun,
#            argnums=tuple(range(len(self.params)+1)))
#        self.invgrad = grad(self.inverse,
#            argnums=tuple(range(len(self.params)+1)))

    def __iter__(self):
        return self.params

# CONSTANTS

# linear function
def fLINEAR(x, a, b):
    return a*x + b
def iLINEAR(y, a, b):
    return (y - b) / a
mLINEAR = Model(fLINEAR, iLINEAR, "linear",
    ("a", "b"), "linear", "linear")

# power function
def fPOWER(x, a, b):
    return a * x**b
def iPOWER(y, a, b):
    return (y / a)**(1/b)
mPOWER = Model(fPOWER, iPOWER, "power",
    ("a", "b"), "log", "log")

# Hill function
def fHILL(x, a, b, c):
    return (a * x**b) / (c**b + x**b)
def iHILL(y, a, b, c):
    return c * (a/y - 1)**(-1/b)
mHILL = Model(fHILL, iHILL, "Hill",
    ("a", "b", "c"), "log", "log")

# logistic function
def fLOGISTIC(x, a, b, c, d):
    return d + (a - d) / (1 + np.exp(-b*(x - c)))
def iLOGISTIC(y, a, b, c, d):
    return c - np.log((a - d)/(y - d) - 1) / b
mLOGISTIC = Model(fLOGISTIC, iLOGISTIC, "logistic",
    ("a", "b", "c", "d"), "linear", "linear")

# 4-parameter logistic
def f4PL(x, a, b, c, d):
    return d + (a - d)/(1 + (x/c)**b)
def i4PL(y, a, b, c, d):
    return c*((a-d)/(y-d) - 1)**(1/b)
m4PL = Model(f4PL, i4PL, "4PL",
    ("a", "b", "c", "d"), "log", "log")

# 5-parameter logistic
def f5PL(x, a, b, c, d, g):
    return d + (a - d)/(1 + (x/c)**b)**g
def i5PL(y, a, b, c, d, g):
    return c*(((a-d)/(y-d))**(1/g) - 1)**(1/b)
m5PL = Model(f5PL, i5PL, "5PL",
    ("a", "b", "c", "d", "g"), "log", "log")

model_list = [mLINEAR, mHILL, mLOGISTIC, m4PL, m5PL]
model_dict = {model.name: model for model in model_list}


# CURVE FITTING

def _match_model(model_name):
    """
    Starting with a string, match the model name which is a key in
    `model_dict`.
   
    Parameters:
    ----------
    `model` : str or waltlabtools.Model
        Model name or waltlabtools.Model. Ideally a member of 
        `model_dict.keys()`, but can also be one with some characters 
        off or different capitalization.
   
    Returns
    -------
    `named_model` : waltlabtools.Model
        Fixed version of `model` which is a built-in 
        `waltlabtools.Model`.
        
    """
    if isinstance(model_name, Model):
        named_model = model_name        
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
                raise KeyError("Model " + m
                    + " not found. Did you mean one of " + str(matches) + "?")
            else:
                raise KeyError("Model " + m + " not found.")
    return named_model


def regress(model, x, y, inverse=False, weights="1/y^2",
            p0=None, bounds=None, method=None):
    """
    Base function for performing regressions. Basically a wrapper for
    `scipy.optimize.curve_fit`.
   
    Parameters:
    ----------
    `model` : waltlabtools.Model or str
        The functional model to use. Should be a valid
        `waltlabtools.Model` object or a string referring to a model in
        `model_names`.
    `x` : array-like
        The independent variable, e.g., concentration.
    `y` : array-like
        The dependent variable, e.g., signal.
    `inverse` : bool, optional
        Should `x` be regressed as a function of `y` instead? Defaults
        to `False`.
    `weights` : str or array-like, optional
        Weights to be used. If array-like, should be the same size as
        `x` and `y`. Otherwise, can be one of the following:
            - `"1/y^2"` : Inverse-squared (1/y^2) weighting.
            - `"1"` : Equal weighting for all data points.
        Other strings raise a NotImplementedError. Defaults to
        `"1/y^2"`.
    `p0` : array-like, optional
        Initial guess for the parameters. If provided, must have the
        same length as the number of parameters. If None, then the
        initial values will all be 1 (if the number of parameters for
        the function can be determined using introspection,
        otherwise a ValueError is raised).
    `bounds` : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to no bounds.
        Each element of the tuple must be either an array with the
        length equal to the number of parameters, or a scalar (in
        which case the bound is taken to be the same for all
        parameters). Use np.inf with an appropriate sign to disable
        bounds on all or some parameters.
    `method` : `{"lm", "trf", "dogbox"}`, optional
        Method to use for optimization. See
        scipy.optimize.least_squares for more details. Default is
        `"lm"` for unconstrained problems and `"trf"` if bounds are
        provided. The method `"lm"` won’t work when the number of
        observations is less than the number of variables; use
        `"trf"` or `"dogbox"` in this case.
   
    Returns
    -------
    `popt` : array
        Optimal values for the parameters so that the sum of the
        squared residuals of f(xdata, *popt) - ydata is minimized.

    """
    if inverse == "both":
        return [regress(model, x, y, inverse=True, weights=weights, p0=p0,
                bounds=bounds, method=method),
            regress(model, x, y, inverse=True, weights=weights, p0=p0,
                bounds=bounds, method=method)]
    else:
        valid_model = _match_model(model)
        if inverse:
            calibration_function = valid_model.inverse
            xdata = flatten(y)
            ydata = flatten(x)
        else:
            calibration_function = valid_model.fun
            xdata = flatten(x)
            ydata = flatten(y)
    if isiterable(weights):
        sigma = flatten(weights)**-2
    elif weights == "1/y^2":
        sigma = ydata
    elif weights == "1":
        sigma = None
    else:
        raise NotImplementedError(_error_text(
            ["weighting scheme", weights], "implementation"))
    kwarg_names = ["p0", "sigma", "bounds", "method"]
    kwargs = dict()
    for k, kwarg in enumerate([p0, sigma, bounds, method]):
        if kwarg is not None:
            kwargs[kwarg_names[k]] = kwarg
    popt, pcov = opt.curve_fit(f=calibration_function,
                               xdata=xdata,
                               ydata=ydata,
                               **kwargs)
    return popt


def lod(blank_signal, inverse_fun, coefs=None, sds=3, corr="c4"):
    """
    Computes the limit of detection (LOD) given a calibration curve and
    blank signal.
    
    Parameters:
    ----------
    `blank_signal` : array_like
        Signal (e.g., average number of enzymes per bead, AEB) of the
        zero calibrator. Must have at least two elements.
    `inverse_fun` : function or waltlabtools.CalCurve
        The functional form used for the calibration curve. If a
        function, it should accept the signal (y) as its only argument
        and return the concentration (x). That is, `inverse_fun` should
        be the inverse function for the calibration curve. If
        `inverse_fun` is a string, it should refer to one of the models
        in model_names, and the coefficients should also be provided.
    `coefs` : dict or list-like, optional
        Coefficients of the model.
            - If `coefs` is a dict, its keys should be the coefficient
            names, and its values should be the coefficient values. If
            there are keys which are not coefficients in the model, a
            RuntimeWarning is issued. If there are parameters of the
            model which are not in `coefs`, a KeyError is raised.
            - If `coefs` is list-like, it should give the values of the
            coefficients in alphabetical order, i.e., in the order in
            which they appear as arguments for the calibration curve
            function. Raises a ValueError if the list is not the same
            length as the number of coefficients in the model.
        Ignored unless `inverse_fun` is a string. Defaults to None.
    `sds` : numeric, optional
        How many standard deviations above the mean should the
        background should the limit of detection be calculated at?
        Standard values include 2.5 (Quanterix), 3 (Walt Lab), and 10
        (LLOQ).
    `corr` : `{"n", "n-1", "n-1.5", "c4"}` or numeric, optional
        The sample standard deviation under-estimates the population
        standard deviation for a normally distributed variable.
        Specifies how this should be addressed. Options:
            - `"n"` : Divide by the number of samples to yield the
            uncorrected sample standard deviation.
            - `"n-1"` : Divide by the number of samples minus one to
            yield the square root of the unbiased sample variance.
            - `"n-1.5"` : Divide by the number of samples minus 1.5 to
            yield the approximate unbiased sample standard deviation.
            - `"c4"` : Divide by the correction factor to yield the
            exact unbiased sample standard deviation. Requires the
            gamma function from either `math` or `scipy.special`; if
            neither of these is loaded, then issues a warning and
            calculates it using `corr="n-1.5"` instead.
            - If numeric, gives the delta degrees of freedom. For
            example, `corr="n"`, `corr="n-1"`, and `corr="n-1.5"` are
            equivalent to `corr=0`, `corr=1`, and `corr=1.5`,
            respectively.
        Defaults to `"c4"`.
       
    Returns
    -------
    `lod_x` : numeric
        The limit of detection, in units of x (typically concentration).
   
    """
    blank_array = flatten(blank_signal)
    n = len(blank_array)
    try:
        ddof = {"n": 0, "n-1": 1, "n-1.5": 1.5, "c4": 1}[corr]
    except KeyError:
        try:
            ddof = float(corr)
        except Exception:
            raise ValueError("Not able to convert corr=" + str(corr)
                             + " into degrees of freedom for standard "
                             + "deviation.")
    corr_factor = c4(n) if (corr == "c4") else 1
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
    """
    Calibration curve.
   
    Parameters:
    ----------
    `model` : waltlabtools.Model
        Mathematical model used.
    `coefs` : list-like
        Numerical values of the parameters specified by `model`.
    `lod` : numeric
        Lower limit of detection (LOD). Defaults to negative infinity.
    `lod_sds` : numeric
        Number of standard deviations above blank at which the lower
        limit of detection is calculated. Common values are 2.5 and 3.
        Defaults to 3.
    `force_lod` : bool
        Should readings below the LOD be set to the LOD?

    Methods
    -------
    `fun(x)`
        Forward function, mapping calibrator readings (e.g., 
        fluorescence) to calibrator values (e.g., concentration).
    `inverse(y)`
        Inverse function, mapping calibrator values (e.g., 
        concentration) to calibrator readings (e.g., fluorescence).

    """
    def __init__(self, model=None, coefs=(), lod=-np.inf, lod_sds=3, 
            force_lod=False):
        self.model = model
        self.coefs = _match_coefs(self.model.params, coefs)
        self.lod = lod
        self.lod_sds = lod_sds
        self.force_lod = force_lod
        
    def bound_lod(self, x_flat):
        if self.force_lod:
            x_above_lod = np.maximum(x_flat, self.lod)
            return x_above_lod
        else:
            return x_flat

    def fun(self, x):
        x_flat = self.bound_lod(flatten(x))
        return self.model.fun(x_flat, **self.coefs)

    def inverse(self, y):
        y_flat = flatten(y)
        x_flat = self.model.inverse(y_flat, **self.coefs)
        return self.bound_lod(x_flat)

    def __iter__(self):
        return self.coefs

    @classmethod
    def from_data(cls, x, y, model, lod_sds=3, force_lod=False, inverse=False,
            weights="1/y^2", p0=None, bounds=None, method=None, corr="c4"):
        """
        Create calibration curve from data.

        """
        x_flat = flatten(x, on_bad_data="error")
        y_flat = flatten(y, on_bad_data="error")
        coefs = regress(model=model, x=x_flat, y=y_flat, inverse=inverse,
            weights=weights, p0=p0, bounds=bounds, method=method)
        cal_curve = cls(model=model, coefs=coefs, lod_sds=lod_sds, 
            force_lod=force_lod)
        cal_curve.lod = lod(y_flat[x_flat == 0], cal_curve.inverse, 
            coefs=coefs, sds=lod_sds, corr=corr)
        return cal_curve

    @classmethod
    def from_function(cls, fun=None, inverse=None, lod=-np.inf,
            ulod=np.inf, lod_sds=3, force_lod=False, 
            xscale="linear", yscale="linear"):
        model = Model(fun=fun, inverse=inverse, 
            xscale=xscale, yscale=yscale)
        return cls(model=model, lod=lod, lod_sds=lod_sds, force_lod=force_lod)
