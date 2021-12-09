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

__all__ = ["Model", "models"]


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
    xscale, yscale : {"linear", "log", "symlog", "logit"} or
    matplotlib.ScaleBase, default "linear"
        The natural scaling transformations for `x` and `y`. For
        example, "log" means that the data may be distributed
        log-normally and are best visualized on a log scale.

    """

    def __init__(
        self,
        fun=None,
        inverse=None,
        name: str = "",
        params=(),
        xscale="linear",
        yscale="linear",
        plaintext_formula: str = "",
    ):
        self.fun = fun
        self.inverse = inverse
        self.name = name
        self.params = params
        self.xscale = xscale
        self.yscale = yscale
        self.plaintext_formula = plaintext_formula

    def __iter__(self):
        return self.params

    def __repr__(self) -> str:
        name_prefix = self.name + " Model" if self.name else "Model"
        if self.plaintext_formula:
            return name_prefix + ": " + self.plaintext_formula
        else:
            return name_prefix + " with parameters " + str(self.params)


# Models

m_linear = Model(
    lambda x, a, b: a * x + b,
    lambda y, a, b: (y - b) / a,
    "linear",
    ("a", "b"),
    "linear",
    "linear",
    "y = a x + b",
)


m_power = Model(
    lambda x, a, b: a * x ** b,
    lambda y, a, b: (y / a) ** (1 / b),
    "power",
    ("a", "b"),
    "log",
    "log",
    "y = a x^b",
)


m_hill = Model(
    lambda x, a, b, c: (a * x ** b) / (c ** b + x ** b),
    lambda y, a, b, c: c * (a / y - 1) ** (-1 / b),
    "Hill",
    ("a", "b", "c"),
    "log",
    "log",
    "y = a x^b / (c^b + x^b)",
)

m_logistic = Model(
    lambda x, a, b, c, d: d + (a - d) / (1 + np.exp(-b * (x - c))),
    lambda y, a, b, c, d: c - np.log((a - d) / (y - d) - 1) / b,
    "logistic",
    ("a", "b", "c", "d"),
    "linear",
    "linear",
    "y = d + (a - d) / {1 + exp[-b (x - c)]}",
)


m_4pl = Model(
    lambda x, a, b, c, d: d + (a - d) / (1 + (x / c) ** b),
    lambda y, a, b, c, d: c * ((a - d) / (y - d) - 1) ** (1 / b),
    "4PL",
    ("a", "b", "c", "d"),
    "log",
    "log",
    "d + (a - d) / [1 + (x/c)^b]",
)

m_5pl = Model(
    lambda x, a, b, c, d, g: d + (a - d) / (1 + (x / c) ** b) ** g,
    lambda y, a, b, c, d, g: c * (((a - d) / (y - d)) ** (1 / g) - 1) ** (1 / b),
    "5PL",
    ("a", "b", "c", "d", "g"),
    "log",
    "log",
    "d + (a - d) / [1 + (x/c)^b]^g",
)


_model_list = [m_linear, m_power, m_hill, m_logistic, m_4pl, m_5pl]
models = {model.name: model for model in _model_list}
"""Built-in regression models.

Keys of model_dict are strings giving model names; values are
waltlabtools.Model objects.

Models
------
"linear" : Linear function.

"power" : Power function.

"hill" : Hill function.

"logistic" : Logistic function.

"4pl" : Four-parameter logistic (4PL) function.

"5pl" : Five-parameter logistic (5PL) function.

"""


#def match_model(model_name) -> Model:
#    """Returns a Model object from a string matching its name.
#
#    Parameters
#    ----------
#    model : str or waltlabtools.Model
#        Model name or waltlabtools.Model object. Ideally a member of
#        model_dict.keys(), but can also be one with some characters
#        off or different capitalization.
#
#    Returns
#    -------
#    named_model : waltlabtools.Model
#        Fixed version of model which is a built-in
#        waltlabtools.Model.
#
#    """
#    if isinstance(model_name, Model):
#        return model_name
#
#    m = str(model_name).casefold()
#    if m in _model_dict:
#        return _model_dict[m]
#
#    matches = [key for key in _model_dict.keys() if (m in key) or (key in m)]
#    if len(matches) == 1:
#        return _model_dict[matches[0]]
#
#    error_text = "Model " + str(model_name) + " not found."
#    if len(matches) > 1:
#        error_text = error_text + " Did you mean " + " or ".join(matches) + "?"
#    raise KeyError(error_text)
