# IMPORT MODULES

import warnings

import scipy.optimize as opt
import scipy.special as spec

try:
    import jax.numpy as np
    from jax import grad
except Exception:
    import numpy as np
    warnings.warn("Module jax.numpy could not be loaded; " 
                  + "using numpy instead. " 
                  + "Performance may be slightly slower.", Warning)
    grad = None


# CONSTANTS

# a = maximum signal (upper asymptote)
# b = [Hill] slope
# c = inflection point
# d = minimum/blank signal (lower asymptote)
# g = asymmetry

# models is a dict
    # keys are the names of functions you can fit
    # values are a list of [parameters, forward function, inverse function]
models = {"linear": [["a", "b"], 
                lambda x, a, b: a*x + b, 
                lambda y, a, b: (y - b)/a], 
          "hill": [["a", "b", "c"], 
                lambda x, a, b, c: (a * x**b) / (c**b + x**b), 
                lambda y, a, b, c: c * (a/y - 1)**(-1/b)],                      
          "logistic": [["a", "b", "c", "d"], 
                lambda x, a, b, c, d: d + (a - d) / (1 + np.exp(-b*(x - c))), 
                lambda y, a, b, c, d: c - np.log((a - d)/(y - d) - 1) / b], 
          "4PL": [["a", "b", "c", "d"], 
                lambda x, a, b, c, d: d + (a - d)/(1 + (x/c)**b), 
                lambda y, a, b, c, d: c*((a-d)/(y-d) - 1)**(1/b)], 
          "5PL": [["a", "b", "c", "d", "g"], 
                lambda x, a, b, c, d, g: d + (a - d)/(1 + (x/c)**b)**g, 
                lambda y, a, b, c, d, g: c*(((a-d)/(y-d))**(1/g) - 1)**(1/b)], 
         }
model_names = set(models.keys())


### SAMPLE FUNCTION DEFINITION
def my_function(required_input, optional_input_="default value"):
    """
    Definition of function in plain English.
    
    Parameters:
    ----------
    `required_input` : type
        Definition of `required_input`.
    `optional_input` : type, optional
        Definition of `optional_input`. Defaults to "default value".
    
    Returns
    -------
    `output` : type
        Definition of `output`.
         
    Notes
    -----
    First note.
    
    Second note.
    
    """
    output = None
    return output


# GENERAL FUNCTIONS

def I(x):
    """
    The identity function.
    
    Parameters:
    ----------
    `x` : any
    
    Returns
    -------
    `x` : same as `x`
        The same `x`, unchanged.
        
    """
    return x

# function to determine if an object has multiple elements
def isiterable(data):
    """
    Determines whether an object is iterable, as determined by it
    having a finite length.
    
    Parameters:
    ----------
    `data` : any
    
    Returns
    -------
    `iterable` : bool
        Returns `True` iff `data` is not a string, has `len`, and 
        `len(data) > 0`.
        
    """
    try:
        return ((len(data) > 0) and not isinstance(data, str))
    except Exception:
        return False

def flatten(data, on_bad_data="warn"):
    """
    Function to flatten most data structures. Will flatten to a 1D 
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
                the_iterator = data.iteritems()
            else:
                the_iterator = enumerate(data)
            for name, datum in the_iterator:
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
                raise TypeError("Input data were coerced from type " 
                                + str(type(data)) + " to type " 
                                + str(type(flattened_data)) 
                                + ", but could not be coerced to an ndarray.")
            elif on_bad_data == "ignore":
                return flattened_data
            else:
                # This includes the default case, where on_bad_data="warn".
                warnings.warn("Input data were coerced from type " 
                              + str(type(data)) + " to type " 
                              + str(type(flattened_data)) 
                              + ", but could not be coerced to an ndarray.", 
                              Warning)
                return flattened_data
            


# CURVE FITTING

def match_model_name(model):
    """
    Starting with a string, match the model name which is a key in 
    `models`.
    
    Parameters:
    ----------
    `model` : str
        Model name. Ideally a member of `models.keys()`, but can also 
        be one with some characters off or different capitalization.
    
    Returns
    -------
    `m` : str
        Fixed version of `model` which is a member of `models.keys()`.
         
    """
    m = str(model)
    if m not in models.keys():
        model_name_matches = [name for name in model_names 
                              if (m.casefold() in key.casefold() 
                                  or key.casefold() in m.casefold())]
        if len(model_name_matches) == 1:
            m = model_name_matches[0]
        elif len(model_name_matches) > 1:
            error_text = ("Model " + m + " not found. Did you mean " 
                          + model_name_matches[0])
            for i in range(1,len(model_name_matches)-1):
                error_text = error_text + ", " + model_name_matches[i]
            error_text = error_text + " or " + model_name_matches[-1] + "?"
            raise KeyError(error_text)
        else:
            raise KeyError("Model " + m + " not found.")
    return m

def cal_curve(model="4PL", inverse=False, coefs=None):
    """
    Base function for calibration curves. Returns a function for the 
    chosen calibration curve form. Fills out coefficients if given.
    
    Parameters:
    ----------
    `model` : str, optional
        Which calibration curve model to use. Must be an element of 
        `model_names`. Defaults to `"4PL"`, a four-parameter logistic 
        regression model, suitable for most assays measuring 
        concentrations.
    `inverse` : bool, optional
        Whether to return the forward calibration function 
        (concentration -> signal) or the inverse calibration function 
        (signal -> concentration). Defaults to False.
    `coefs` : dict, list-like, or None, optional
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
            - If `coefs=None`, then `calibration_function` will accept 
            the model's coefficients as arguments.
        Defaults to None.
    
    Returns
    -------
    `calibration_function` : function
        A function (`lambda`) mapping concentration to signal (or
        signal to concentration, if `inverse=True`). The first argument 
        of `calibration_function` is the concentration (or signal, if 
        `inverse=True`), and the remaining arguments, if any, are the 
        coefficients which were not defined by `coefs` above. 
        In general, `calibration_function` will use vectorized 
        operations, so it can be applied to numeric primitives, 
        ndarrays, series, etc. For other inputs, use `flatten` first. 
        The coefficients given as arguments for `calibration_function` 
        should be primitives.
    
    """
    m = match_model_name(model)
    if isinstance(coefs, dict):
        coefs_keys = set(coefs.keys)
        model_keys = set(models[m][0])
        if coefs_keys == model_keys:
            calibration_function = (lambda x: 
                                        models[m][1+inverse](x, **coefs))
        elif coefs_keys > model_keys:
            warnings.warn("Coefficients " + str(coefs_keys - model_keys) 
                          + " were provided but not used for model " 
                          + m + ".", RuntimeWarning)
            model_coefs = {key : coefs[key] for key in model_keys}
            calibration_function = (lambda x: 
                                        models[m][1+inverse](x, **model_coefs))
        else:
            raise KeyError("The coefficients " + str(model_keys - coefs_keys) 
                           + " were not found in coefs." + "The model " + m 
                           + " requires the coefficients " 
                           + str(model_keys) + ".")
    elif isiterable(coefs):
        if len(coefs) == len(models[m][0]):
            calibration_function = (lambda x: 
                                        models[m][1+inverse](x, *list(coefs)))
        else:
            raise ValueError("Wrong number of coefficients. The model " + m 
                             + " requires " + str(len(models[m][0])) 
                             + " parameters: " + str(models[m][0]) 
                             + ". You provided " + str(len(coefs)) 
                             + " parameters: " + str(coefs) + ".")
    else:
        calibration_function = models[m][1+inverse]
    return calibration_function
       
def regress(fun, x, y, inverse=False, weights="1/y^2", 
            initial_guess=None, bounds=None, method=None, jac=None):
    """
    Base function for performing regressions. Basically a wrapper for 
    `scipy.optimize.curve_fit`.
    
    Parameters:
    ----------
    `fun` : function or str
        The function to be used. If a function, it should take `x` or 
        `y` as its first argument, and its coefficients as the others. 
        If a string, it should refer to a model in `model_names`.
    `x` : array-like
        The independent variable, e.g., concentration.
    `y` : array-like
        The dependent variable, e.g., signal.
    `inverse` : bool, optional
        Should `x` be regressed as a function of `y` instead? 
        Implemented only when `fun` is a string. Defaults to `False`.
    `weights` : str or array-like, optional
        Weights to be used. If array-like, should be the same size as 
        `x` and `y`. Otherwise, can be one of the following:
            - `"1/y^2"` : Inverse-squared (1/y^2) weighting.
            - `"1"` : Equal weighting for all data points.
        Other strings raise a NotImplementedError. Defaults to 
        `"1/y^2"`.
    `initial_guess` : array-like, optional
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
    `jac` : function or string, optional
        Function with signature jac(x, ...) which computes the 
        Jacobian matrix of the model function with respect to 
        parameters as a dense array_like structure. If None, 
        the Jacobian will be estimated numerically. String keywords 
        for 'trf' and 'dogbox' methods can be used to select a finite 
        difference scheme; see `scipy.optimize.least_squares`. 
        Defaults to `jax.grad` if jax is installed, or None if not.
    
    Returns
    -------
    `popt` : array
        Optimal values for the parameters so that the sum of the 
        squared residuals of f(xdata, *popt) - ydata is minimized.

    """
    if inverse=="both":
        return [regress(x, y, fun, inverse=True), 
                regress(x, y, fun, inverse=False)]
    elif isinstance(fun, str):
        m = match_model_name(fun)
        calibration_function = models[m][1+inverse]
    else:
        calibration_function = fun
    xdata = flatten(x)
    ydata = flatten(y)
    if isiterable(weights):
        sigma = flatten(weights)**-2
    elif weights == "1/y^2":
        sigma = ydata
    elif weights == "1":
        sigma = None
    else:
        raise NotImplementedError("The weighting scheme " + str(weights) 
                                  + " does not exist. Please try another one.")
    kwarg_names = ["p0", "sigma", "bounds", "method", "jac"]
    kwargs = dict()
    if jac == "grad":
         jac2 = lambda fun: jax.grad(fun, 
                                     argnums=list(range(
                                        fun.__code__.co_argcount)))
    for k, kwarg in enumerate([initial_guess, sigma, bounds, method, jac]):
        if kwarg is not None:
            kwargs[kwarg_names[k]] = kwarg
    popt, pcov = opt.curve_fit(f=calibration_function, 
                               xdata=xdata, 
                               ydata=ydata, 
                               **kwargs)
    return popt

def aeb(fon):
    return -np.log(1 - fon)
    
def fon(aeb):
    return 1 - np.exp(-aeb)

def c4(n):
    try:
        return np.sqrt(2 / (n-1)) * spec.gamma(n/2) / spec.gamma((n-1)/2)
    except Exception:
        warnings.warn("Gamma function not available. Using n-1.5 instead.")
        return np.sqrt((n-1.5) / (n-1))    

def lod(blank_signal, inverse_fun, coefs=None, sds=3, corr="c4"):
    """
    Computes the limit of detection (LOD) given a calibration curve and 
    blank signal.
    
    Parameters:
    ----------
    `blank_signal` : array_like
        Signal (e.g., average number of enzymes per bead, AEB) of the 
        zero calibrator. Must have at least two elements.
    `inverse_fun` : function or str
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
        ddof = {"n":0, "n-1":1, "n-1.5":1.5, "c4":1}[corr]
    except KeyError:
        try:
            ddof = float(corr)
        except Exception:
            raise ValueError("Not able to convert corr=" + str(corr) 
                             + " into degrees of freedom for standard " 
                             + "deviation.")
    corr_factor = c4(n) if (corr=="c4") else 1
    mean = np.mean(blank_array)
    stdev = np.std(blank_array, ddof=ddof)/corr_factor
    lod_y = mean + sds*stdev
    if isinstance(inverse_fun, str):
        lod_x = cal_curve(match_model_name(inverse_fun), inverse=True, 
                          coefs=coefs)(lod_y)
    else:
        lod_x = inverse_function(lod_y)
    return lod_x