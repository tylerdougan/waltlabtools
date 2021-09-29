"""Class Model, its methods, and related functions.

Everything in waltlabtools.model is automatically imported with
waltlabtools, so it can be accessed via, e.g.,

.. code-block:: python

   import waltlabtools as wlt  # waltlabtools main functionality

   my_model = wlt.Model()  # creates a new empty Model object


-----


"""

from .core import _optional_dependencies

if _optional_dependencies["jax"]:
    import jax.numpy as np
else:
    import numpy as np

__all__ = ["Model", "model_dict"]


class Model:
    """Mathematical model for calibration curve fitting.

    A Model is an object with a function and its inverse, with one
    or more free parameters that can be fit to calibration curve data.

    Parameters
    ----------
    fun : function
        Forward functional form. Should be a function which takes in `x`
        and other parameters and returns `y`. The first parameter of
        fun should be `x`, and the remaining parameters should be
        the coefficients which are fit to the data (typically floats).
    inverse : function
        Inverse functional form. Should be a function which takes in `y`
        and other parameters and returns `x`. The first parameter of
        **inverse** should be `y`, and the remaining parameters should
        be the same coefficients as in fun.
    name : str
        The name of the function. For example, "4PL" or "linear".
    params : list-like of str
        The names of the parameters for the function. This should be
        the same length as the number of arguments which fun and
        inverse take after their inputs `x` and `y`, respectively.
    xscale, yscale : {"linear", "log", "symlog", "logit"}, default "linear"
        The natural scaling transformations for `x` and `y`. For
        example, "log" means that the data may be distributed
        log-normally and are best visualized on a log scale.

    """

    def __init__(self, fun=None, inverse=None, name: str = "", params=(),
            xscale="linear", yscale="linear"):
        self.fun = fun
        self.inverse = inverse
        self.name = name
        self.params = params
        self.xscale = xscale
        self.yscale = yscale

    def __iter__(self):
        return self.params

    def __repr__(self):
        return " ".join([self.name, "Model with parameters", str(self.params)])


# CONSTANTS

def _f_linear(x, a, b):
    """Linear function."""
    return a*x + b
def _i_linear(y, a, b):
    """Linear inverse."""
    return (y - b) / a
m_linear = Model(_f_linear, _i_linear, "linear",
    ("a", "b"), "linear", "linear")


def _f_power(x, a, b):
    """Power function."""
    return a * x**b
def _i_power(y, a, b):
    """Power inverse."""
    return (y / a)**(1/b)
m_power = Model(_f_power, _i_power, "power",
    ("a", "b"), "log", "log")


def _f_hill(x, a, b, c):
    """Hill function."""
    return (a * x**b) / (c**b + x**b)
def _i_hill(y, a, b, c):
    """Hill inverse."""
    return c * (a/y - 1)**(-1/b)
m_hill = Model(_f_hill, _i_hill, "Hill",
    ("a", "b", "c"), "log", "log")


def _f_logistic(x, a, b, c, d):
    """Logistic function."""
    return d + (a - d) / (1 + np.exp(-b*(x - c)))
def _i_logistic(y, a, b, c, d):
    """Logistic inverse."""
    return c - np.log((a - d)/(y - d) - 1) / b
m_logistic = Model(_f_linear, _i_logistic, "logistic",
    ("a", "b", "c", "d"), "linear", "linear")


def _f_4PL(x, a, b, c, d):
    """Four-parameter logistic (4PL) function."""
    return d + (a - d)/(1 + (x/c)**b)
def _i_4PL(y, a, b, c, d):
    """Four-parameter logistic (4PL) inverse."""
    return c*((a-d)/(y-d) - 1)**(1/b)
m_4PL = Model(_f_4PL, _i_4PL, "4PL",
    ("a", "b", "c", "d"), "log", "log")


def _f_5PL(x, a, b, c, d, g):
    """Five-parameter logistic (5PL) function."""
    return d + (a - d)/(1 + (x/c)**b)**g
def _i_5PL(y, a, b, c, d, g):
    """Five-parameter logistic (5PL) inverse."""
    return c*(((a-d)/(y-d))**(1/g) - 1)**(1/b)
m_5PL = Model(_f_5PL, _i_5PL, "5PL",
    ("a", "b", "c", "d", "g"), "log", "log")


model_list = [m_linear, m_power, m_hill, m_logistic, m_4PL, m_5PL]
model_dict = {model.name: model for model in model_list}
    """Built-in regression models.

    Keys of model_dict are strings giving model names; values are
    waltlabtools.Model objects.

    Models
    ------
    "linear" : Linear function.

    "power" : Power function.

    "Hill" : Hill function.

    "logistic" : Logistic function.

    "4PL" : Four-parameter logistic (4PL) function.

    "5PL" : Five-parameter logistic (5PL) function.

    """
