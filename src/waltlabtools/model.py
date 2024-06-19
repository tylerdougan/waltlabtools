from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.scale import ScaleBase

from .core import coerce_array


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
    jac : {'2-point', '3-point', 'cs', callable}, optional
        Method of computing the Jacobian matrix (an m-by-n matrix, where
        element (i, j) is the partial derivative of f[i] with respect to
        x[j]) of the loss function. The keywords select a finite
        difference scheme for numerical estimation. The scheme '3-point'
        is more accurate, but requires twice as many operations as
        '2-point' (default). The scheme 'cs' uses complex steps, and
        while potentially the most accurate, it is applicable only when
        `func` correctly handles complex inputs and can be analytically
        continued to the complex plane. If callable, it is used as
        ``jac(x, *args, **kwargs)`` and should return a good approximation
        (or the exact value) for the Jacobian as an array_like (np.atleast_2d
        is applied), a sparse matrix (csr_matrix preferred for performance) or
        a `scipy.sparse.linalg.LinearOperator`.

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
        jac: str | Callable = "2-point",
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


def jac_linear(
    coef: np.ndarray, *, X: np.ndarray, sample_weight: np.ndarray
) -> np.ndarray:
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


def jac_power(
    coef: np.ndarray, *, X: np.ndarray, sample_weight: np.ndarray
) -> np.ndarray:
    a, b = coef
    J_a = b * X**a * np.where(X > 0, np.log(X), 0) * sample_weight
    J_b = X**a * sample_weight
    return np.column_stack((J_a, J_b))


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


def jac_hill(
    coef: np.ndarray, *, X: np.ndarray, sample_weight: np.ndarray
) -> np.ndarray:
    a, b, c = coef
    X_b = X**b
    c_b = c**b
    denom = (c_b + X_b) ** 2

    J_a = X_b / (c_b + X_b)
    J_b = a * X_b * np.where(X > 0, np.log(X), 0) * (c_b - X_b) / denom
    J_c = -a * X_b * b * c ** (b - 1) / denom
    return np.column_stack((J_a, J_b, J_c)) * sample_weight[:, np.newaxis]


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


def jac_logistic(
    coef: np.ndarray, *, X: np.ndarray, sample_weight: np.ndarray
) -> np.ndarray:
    a, b, c, d = coef
    exp_term = np.exp(-b * (X - c))
    denom = 1 + exp_term
    denom2 = denom**2

    J_a = 1 / denom
    J_b = (a - d) * (X - c) * exp_term / denom2
    J_c = (a - d) * b * exp_term / denom2
    J_d = exp_term / denom
    return np.column_stack((J_a, J_b, J_c, J_d)) * sample_weight[:, np.newaxis]


logistic_model = Model(
    func=logistic,
    inverse=logistic_inverse,
    coef_init=np.array([1, 1, 0, 0]),
    name="logistic",
    plaintext_formula="y = d + (a - d) / {1 + exp[-b (x - c)]}",
    xscale="linear",
    yscale="linear",
    jac=jac_logistic,
)


# 3PL
@coerce_array
def three_param_logistic(X, a: float = 0, c: float = 1, d: float = 30):
    return d + (a - d) / (1 + X / c)


@coerce_array
def three_param_logistic_inverse(y, a: float = 0, c: float = 1, d: float = 30):
    return c * ((a - d) / (y - d) - 1)


def jac_three_param_logistic(
    coef: np.ndarray, *, X: np.ndarray, sample_weight: np.ndarray
) -> np.ndarray:
    a, c, d = coef
    denom = 1 + X / c

    J_a = 1 / denom
    J_c = (a - d) * X / (c**2 * denom**2)
    J_d = X / (c + X)

    return np.column_stack((J_a, J_c, J_d)) * sample_weight[:, np.newaxis]


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


def jac_four_param_logistic(
    coef: np.ndarray, *, X: np.ndarray, sample_weight: np.ndarray
) -> np.ndarray:
    a, b, c, d = coef
    X_div_c = X / c
    X_div_c_b = X_div_c**b
    denom = 1 + X_div_c_b
    denom2 = denom**2

    J_a = 1 / denom
    J_b = -(a - d) * np.where(X > 0, X_div_c_b * np.log(X_div_c) / denom2, 0)
    J_c = (a - d) * b * X**b / (c ** (b + 1) * denom2)
    J_d = 1 - J_a
    return np.column_stack((J_a, J_b, J_c, J_d)) * sample_weight[:, np.newaxis]


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


def jac_five_param_logistic(
    coef: np.ndarray, *, X: np.ndarray, sample_weight: np.ndarray
) -> np.ndarray:
    a, b, c, d, g = coef
    X_div_c = X / c
    X_div_c_b = X_div_c**b
    denom = (1 + X_div_c_b) ** g
    denom_g_plus_1 = (1 + X_div_c_b) ** (g + 1)

    J_a = 1 / denom
    J_b = (
        -(a - d) * g * X_div_c_b * np.where(X > 0, np.log(X_div_c), 0) / denom_g_plus_1
    )

    # Derivative with respect to c
    J_c = (a - d) * g * b * X**b / (c ** (b + 1) * denom_g_plus_1)

    # Derivative with respect to d
    J_d = 1 - J_a

    # Derivative with respect to g
    J_g = -(a - d) * np.log(1 + X_div_c_b) / denom

    # Stack the derivatives and apply sample weights
    J = np.column_stack((J_a, J_b, J_c, J_d, J_g))
    return J * sample_weight[:, np.newaxis]


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
