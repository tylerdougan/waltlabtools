from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from matplotlib.scale import ScaleBase

from .core import coerce_array

EPS = np.finfo(float).resolution


class Model:
    """Mathematical model for calibration curve fitting.

    A Model is an object with a function and its inverse, with one
    or more free parameters that can be fit to calibration curve data.

    Parameters
    ----------
    func : function
        Forward functional form, mapping levels (e.g., concentrations)
        to signal values (e.g., AEB). Should be a function which takes
        in `X` and other parameters and returns `y`. The first
        parameter of func should be `X`, and the remaining parameters
        should be the coefficients which are fit to the data (typically
        floats).
    inverse : function
        Inverse functional form, mapping signal values (e.g., AEB) to
        levels (e.g., concentrations). Should be a function which takes
        in `y` and other parameters and returns `X`. The first
        parameter of inverse should be `y`, and the remaining
        parameters should be the same coefficients as in fun.
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
        func: Callable,
        inverse: Callable,
        coef_init: Any,
        name: str = "",
        plaintext_formula: str = "",
        xscale: str | ScaleBase = "linear",
        yscale: str | ScaleBase = "linear",
        jac: Optional[Callable] = None,
    ):
        self.func = func
        self.inverse = inverse
        self.coef_init = coef_init
        self.name = name
        self.plaintext_formula = plaintext_formula
        self.xscale = xscale
        self.yscale = yscale
        self.coef_init = coef_init
        self.jac = jac


# LINEAR
# @coerce_array
def linear(X, a: float = 1, b: float = 0):
    return a * X + b


# @coerce_array
def linear_inverse(y, a: float = 1, b: float = 0):
    return (y - b) / a


def jac_linear(coef, X, y, sample_weight):
    J_a = X * sample_weight
    J_b = sample_weight
    return np.column_stack((J_a, J_b))


linear_model = Model(
    func=linear,
    inverse=linear_inverse,
    coef_init=np.array([1, 0]),
    name="linear",
    plaintext_formula="y = a x + b",
    xscale="linear",
    yscale="linear",
    jac=jac_linear,
)


# POWER
@coerce_array
def power(X, a: float = 1, b: float = 1):
    return b * X**a


@coerce_array
def power_inverse(y, a: float = 1, b: float = 1):
    return (y / b) ** (1 / a)


def jac_power(coef, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):
    a, b = coef  # Unpack the coefficients
    # Derivative with respect to a
    J_a = b * X**a * np.log(X + EPS) * sample_weight
    # Derivative with respect to b
    J_b = X**a * sample_weight
    return np.column_stack((J_a, J_b))  # Shape (n_samples, 2)


power_model = Model(
    func=power,
    inverse=power_inverse,
    coef_init=np.array([1, 1]),
    name="power",
    plaintext_formula="y = a x^b",
    xscale="log",
    yscale="log",
    jac=jac_power,
)


# HILL
def hill(X, a: float = 1, b: float = 1, c: float = 1):
    return (a * X**b) / (c**b + X**b)


def hill_inverse(y, a: float = 1, b: float = 1, c: float = 1):
    return c * (a / y - 1) ** (-1 / b)


def jac_hill(coef, X, y, sample_weight):
    a, b, c = coef  # Unpack the coefficients
    denominator = c**b + X**b
    # Derivative with respect to a
    J_a = (X**b / (denominator)) * sample_weight
    # Derivative with respect to b
    J_b = a * (c * X) ** b * np.log((X + EPS) / c) / (denominator**2) * sample_weight
    # Derivative with respect to c
    J_c = (-a * X**b * b * c ** (b - 1) / (denominator**2)) * sample_weight
    return np.column_stack((J_a, J_b, J_c))  # Shape (n_samples, 3)


hill_model = Model(
    func=hill,
    inverse=hill_inverse,
    coef_init=np.array([1, 1, 1]),
    name="Hill",
    plaintext_formula="y = a x^b / (c^b + x^b)",
    xscale="log",
    yscale="log",
    jac=jac_hill,
)


# LOGISTIC
@coerce_array
def logistic(X, a: float = 1, b: float = 1, c: float = 0, d: float = 0):
    return d + (a - d) / (1 + np.exp(-b * (X - c)))


@coerce_array
def logistic_inverse(y, a: float = 1, b: float = 1, c: float = 0, d: float = 0):
    return c - np.log((a - d) / (y - d) - 1) / b


logistic_model = Model(
    func=logistic,
    inverse=logistic_inverse,
    coef_init=np.array([1, 1, 0, 0]),
    name="logistic",
    plaintext_formula="y = d + (a - d) / {1 + exp[-b (x - c)]}",
    xscale="linear",
    yscale="linear",
)


# 3PL
@coerce_array
def three_param_logistic(X, a: float = 0, c: float = 1, d: float = 30):
    return d + (a - d) / (1 + X / c)


@coerce_array
def three_param_logistic_inverse(y, a: float = 0, c: float = 1, d: float = 30):
    return c * ((a - d) / (y - d) - 1)


def jac_three_param_logistic(coef, X, y, sample_weight):
    a, c, d = coef
    one_plus_X_over_c = 1 + X / c

    # Derivative with respect to a
    J_a = (1 / one_plus_X_over_c) * sample_weight

    # Derivative with respect to c
    J_c = ((a - d) * X) / (c**2 * one_plus_X_over_c**2) * sample_weight

    # Derivative with respect to d
    J_d = (1 - 1 / one_plus_X_over_c) * sample_weight

    return np.column_stack((J_a, J_c, J_d))


three_param_logistic_model = Model(
    func=three_param_logistic,
    inverse=three_param_logistic_inverse,
    coef_init=np.array([0, 1, 30]),
    name="3PL",
    plaintext_formula="y = d + (a - d) / (1 + x/c)",
    xscale="log",
    yscale="log",
    jac=jac_three_param_logistic,
)


# 4PL
@coerce_array
def four_param_logistic(X, a: float = 0, b: float = 1, c: float = 1, d: float = 30):
    return d + (a - d) / (1 + (X / c) ** b)


@coerce_array
def four_param_logistic_inverse(
    y, a: float = 0, b: float = 1, c: float = 1, d: float = 30
):
    return c * ((a - d) / (y - d) - 1) ** (1 / b)


def jac_four_param_logistic(coef, X, y, sample_weight):
    a, b, c, d = coef
    X_over_c_b = (X / c) ** b
    one_plus_X_over_c_b = 1 + X_over_c_b

    # Derivative with respect to a
    J_a = (1 / one_plus_X_over_c_b) * sample_weight

    # Derivative with respect to b
    J_b = (
        (a - d) * X_over_c_b * np.log((X + EPS) / c) / one_plus_X_over_c_b**2
    ) * sample_weight

    # Derivative with respect to c
    J_c = ((a - d) * b * X**b / (c ** (b + 1) * one_plus_X_over_c_b**2)) * sample_weight

    # Derivative with respect to d
    J_d = (1 - 1 / one_plus_X_over_c_b) * sample_weight

    return np.column_stack((J_a, J_b, J_c, J_d))


four_param_logistic_model = Model(
    func=four_param_logistic,
    inverse=four_param_logistic_inverse,
    coef_init=np.array([0, 1, 1, 30]),
    name="4PL",
    plaintext_formula="y = d + (a - d) / [1 + (x/c)^b]",
    xscale="log",
    yscale="log",
    jac=jac_four_param_logistic,
)


# 5PL
@coerce_array
def five_param_logistic(
    X, a: float = 0, b: float = 1, c: float = 1, d: float = 30, g: float = 1
):
    return d + (a - d) / (1 + (X / c) ** b) ** g


@coerce_array
def five_param_logistic_inverse(
    y, a: float = 0, b: float = 1, c: float = 1, d: float = 30, g: float = 1
):
    return c * (((a - d) / (y - d)) ** (1 / g) - 1) ** (1 / b)


def jac_five_param_logistic(coef, X, y, sample_weight):
    a, b, c, d, g = coef
    X_over_c_b = (X / c) ** b
    one_plus_X_over_c_b = 1 + X_over_c_b
    # Derivative with respect to a
    J_a = (1 / one_plus_X_over_c_b**g) * sample_weight
    # Derivative with respect to b
    J_b = (
        g
        * (a - d)
        * X_over_c_b
        * np.log((X + EPS) / c)
        / one_plus_X_over_c_b ** (g + 1)
    ) * sample_weight
    # Derivative with respect to c
    J_c = (
        g * (a - d) * b * X**b / (c ** (b + 1) * one_plus_X_over_c_b ** (g + 1))
    ) * sample_weight
    # Derivative with respect to d
    J_d = (1 - 1 / one_plus_X_over_c_b**g) * sample_weight
    # Derivative with respect to g
    J_g = (
        -((a - d) * np.log(one_plus_X_over_c_b) / one_plus_X_over_c_b**g)
        * sample_weight
    )
    return np.column_stack((J_a, J_b, J_c, J_d, J_g))


five_param_logistic_model = Model(
    func=five_param_logistic,
    inverse=five_param_logistic_inverse,
    coef_init=np.array([0, 1, 1, 30, 1]),
    name="5PL",
    plaintext_formula="y = d + (a - d) / [1 + (x/c)^b]^g",
    xscale="log",
    yscale="log",
    jac=jac_five_param_logistic,
)


MODELS: dict[str, Model] = {
    "linear": linear_model,
    "power": power_model,
    "Hill": hill_model,
    "logistic": logistic_model,
    "3PL": three_param_logistic_model,
    "4PL": four_param_logistic_model,
    "5PL": five_param_logistic_model,
}
"""Built-in regression models.

Keys are strings giving model names; values are waltlabtools.Model 
objects.

Models
------
"linear" : Linear function.

"power" : Power function.

"Hill" : Hill function.

"logistic" : Logistic function.

"3PL" : Four-parameter logistic (3PL) function.

"4PL" : Four-parameter logistic (4PL) function.

"5PL" : Five-parameter logistic (5PL) function.

"""
