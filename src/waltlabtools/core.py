"""Core functionality for the waltlabtools module.

Everything in waltlabtools.core is automatically imported with
waltlabtools, so it can be accessed via, e.g.,

.. code-block:: python

   import waltlabtools as wlt  # waltlabtools main functionality

   my_data = wlt.flatten([[[[[0], 1], 2], 3], 4])  # flatten a list


-----


"""

import inspect
import warnings
from functools import wraps
from typing import Any, Callable, Iterable, Literal, Optional

from ._backend import gammaln, gmean, jit, np


def deprecate(replace_with: Optional[str] = None) -> Callable:
    """Decorator to issue a DeprecationWarning with an optional replacement function."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            warning_message = f"{func.__name__} is deprecated as of version 1.0."
            if replace_with:
                warning_message += f" Please use {replace_with} instead."
            warnings.warn(warning_message, DeprecationWarning)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _flatten_recursively(a) -> list:
    """Recursively flatten a primitive or nested iterable.

    Parameters
    ----------
    a : Any
        The iterable to be flattened.

    Returns
    -------
    flat_a : list
        A flattened list of the input iterable's elements.

    Examples
    --------
    >>> _flatten_recursively([1, [2, [3]], 4])
    [1, 2, 3, 4]

    >>> _flatten_recursively({'a': [1, 2], 'b': {'c': 3, 'd': 4}})
    [1, 2, 3, 4]

    >>> _flatten_recursively(1)
    [1]

    """
    flat_a = []
    iterator = a.values() if isinstance(a, dict) else a
    try:
        for datum in iterator:
            flattened_datum = _flatten_recursively(datum)
            flat_a.extend(flattened_datum)
    except TypeError:
        return [a]
    else:
        return flat_a


def flatten(a, order: Literal["C", "F", "A", "K"] = "K") -> np.ndarray:
    """Flatten almost anything into a 1-dimensional numpy array.

    In simple cases, this function is a wrapper for numpy.ravel. In more
    complex cases, it recursively flattens nested iterables.

    Parameters
    ----------
    a : any
        The object to be flattened.
    order : {'C', 'F', 'A', 'K'}, optional
        The elements of `a` are read using this index order. 'C' means
        to index the elements in row-major, C-style order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest.  'F' means to index the elements
        in column-major, Fortran-style order, with the
        first index changing fastest, and the last index changing
        slowest. Note that the 'C' and 'F' options take no account of
        the memory layout of the underlying array, and only refer to
        the order of axis indexing.  'A' means to read the elements in
        Fortran-like index order if `a` is Fortran *contiguous* in
        memory, C-like order otherwise.  'K' means to read the
        elements in the order they occur in memory, except for
        reversing the data when strides are negative.  By default, 'C'
        index order is used.

    Returns
    -------
    numpy.ndarray
        A 1-dimensional numpy array containing the flattened elements.

    See Also
    --------
    numpy.ravel : Flatten a numpy array.

    Notes
    -----
    When jax has been loaded as a backend, this function will raise an
    error if `a` has non-numerical elements.

    Examples
    --------
    >>> a = [[1, 2, 3], [4, 5, 6]]
    >>> flatten(a)
    array([1, 2, 3, 4, 5, 6])

    >>> b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> flatten(b)
    array([1, 2, 3, 4, 5, 6, 7, 8])

    """
    try:
        y = np.ravel(np.asarray(a, order=order), order=order)
        if y.dtype.kind == "O":
            raise ValueError("Cannot flatten object arrays.")
        return y
    except (TypeError, ValueError):
        return np.ravel(np.asarray(_flatten_recursively(a), order=order), order=order)


def coerce_array(func: Callable) -> Callable:
    """Coerce the argument to an array upon TypeError.

    This decorator is intended to wrap functions that are primarily
    called with a first argument that is compatible with `numpy.array`.
    If calling the function results in a `TypeError`, it tries coercing
    the first argument to a numpy array and calling the function again.
    This is particularly useful for functions that might be passed
    lists or other array-like objects but are designed to work with
    numpy arrays.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The wrapped function which coerces its first argument to a
        numpy array upon TypeError.

    Examples
    --------
    >>> @coerce_array
    ... def add_one(arr):
    ...     return arr + 1

    >>> add_one([1, 2, 3])
    array([2, 3, 4])

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            if args:  # if the first argument is positional
                args = (np.asarray(args[0]),) + args[1:]
            else:  # if the first argument is keyword
                sig = inspect.signature(func)
                first_arg_name = next(iter(sig.parameters.keys()))
                kwargs[first_arg_name] = np.asarray(kwargs[first_arg_name])
            return func(*args, **kwargs)

    return wrapper


@coerce_array
def aeb(fon_):
    """Compute the average number of enzymes per bead.

    Converts the fraction of on-beads (fon) to the average number of
    enzymes per bead (AEB) using Poisson statistics. The formula used
    is `aeb_` = -log(1 - `fon_`).

    Parameters
    ----------
    fon_ : numeric or array-like
        A scalar or array of fractions of beads which are "on."

    Returns
    -------
    aeb_ : same as input, or array
        The average number of enzymes per bead.

    See Also
    --------
    fon : inverse of ``aeb``

    """
    return -np.log(1 - fon_)


@coerce_array
def fon(aeb_):
    """Compute the fraction of beads which are on.

    Converts the average enzymes per bead (AEB) to the fraction of
    on-beads (fon) using Poisson statistics. The formula used is
    `fon_` = 1 - exp(-`aeb_`).

    Parameters
    ----------
    aeb_ : numeric or array-like
        A scalar or array of the average number of enzymes per bead.

    Returns
    -------
    fon_ : same as input, or array
        The fractions of beads which are "on."

    See Also
    --------
    aeb : inverse of ``fon``

    """
    return 1 - np.exp(-aeb_)


@coerce_array
def c4(n):
    """Factor `c4` for unbiased estimation of normal standard deviation.

    For a finite sample, the sample standard deviation tends to
    underestimate the population standard deviation. See, e.g.,
    https://www.spcpress.com/pdf/DJW353.pdf for details. Dividing the
    sample standard deviation by the correction factor `c4` gives an
    unbiased estimator of the population standard deviation. This
    correction factor should be applied on top of Bessel's correction,
    so n-1 is used as the degrees of freedom.

    Parameters
    ----------
    n : int or array
        The number of samples.

    Returns
    -------
    numeric or array
        The correction factor, usually written `c4` or `b(n)`.

    See Also
    --------
    std : unbiased standard deviation

    numpy.std : standard deviation

    lod : limit of detection

    """
    return np.sqrt(2 / (n - 1)) * np.exp(gammaln(n / 2) - gammaln((n - 1) / 2))


_DDOFS = {"n": 0, "n-1": 1, "n-1.5": 1.5, "c4": 1}


@coerce_array
def std(
    a, corr: str | int = "c4", axis: Optional[int | tuple[int, ...]] = None, **kwargs
):
    """Compute (an unbiased estimate of) the standard deviation.

    For a finite sample, the sample standard deviation tends to
    underestimate the population standard deviation. This function
    divides the sample standard deviation by the correction factor
    `c4` to give an unbiased estimator of the population standard
    deviation.

    Parameters
    ----------
    a : array-like
        The array of values.
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
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened array.
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.

    Returns
    -------
    numeric
        The unbiased standard deviation.

    See Also
    --------
    numpy.std : standard deviation

    c4 : correction factor used when corr="c4"

    """
    ddof = _DDOFS.get(corr, corr)  # type: ignore
    ret = np.std(a, axis=axis, ddof=ddof, **kwargs)  # type: ignore

    if corr == "c4":
        n = np.prod(np.shape(a))
        if isinstance(axis, int):
            n /= np.prod(np.shape(a)[axis])
        elif isinstance(axis, Iterable):
            n /= np.prod([np.shape(a)[ax] for ax in axis])
        return ret / c4(n)
    else:
        return ret


@jit
def _gmnd_f_step(
    previous_values: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08
) -> tuple[np.ndarray, bool]:
    """Compute one step of the Geothmetic Meandian algorithm.

    Parameters
    ----------
    previous_values : numpy.ndarray
        An array containing the previous values of the Geothmetic
        Meandian algorithm.
    rtol : float, default 1e-05
        The relative tolerance parameter used to determine convergence.
    atol : float, default 1e-08
        The absolute tolerance parameter used to determine convergence.

    Returns
    -------
    next_values : numpy.ndarray
        The next values of the Geothmetic Meandian algorithm.

    converged : bool
        Whether the algorithm has converged.

    Notes
    -----
    This function is decomposed this way in order to make use of
    just-in-time (JIT) compilation with Numba or Jax.
    """
    next_values = np.asarray(
        (
            np.mean(previous_values),
            gmean(previous_values),
            np.median(previous_values),
        )
    )
    converged = np.allclose(
        next_values,
        np.full_like(next_values, fill_value=next_values[2]),
        rtol=rtol,
        atol=atol,
    )
    return next_values, converged


@coerce_array
def geothmetic_meandian(
    a,
    weights=None,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    nan_policy="omit",
) -> float:
    """Compute he Geothmetic Meandian of the input data, as per XKCD.

    For details, see https://xkcd.com/2435/. This function compares
    the three most common measures of central tendency: the arithmetic
    mean, the geometric mean, and the median.
    The geometric meandian uses an iterative process that stops when
    the arithmetic mean, geometric mean, and median converge within
    (`atol` + `rtol` * their magnitudes).

    Parameters
    ----------
    data : array_like
        The input data, which can be any array-like object.
    rtol : float, default 1e-05
        The relative tolerance parameter used to determine convergence.
    atol : float, default 1e-08
        The absolute tolerance parameter used to determine convergence.

    Returns
    -------
    float
        The Geothmetic Meandian of the input data.

    Examples
    --------
    >>> geothmetic_meandian([1, 1, 2, 3, 5])
    2.089440951883
    """
    if nan_policy == "omit":
        a = dropna(a)

    current_values = np.asarray(
        (np.average(a, weights=weights), gmean(a, weights=weights), np.median(a))
    )

    converged = False
    while not converged:
        current_values, converged = _gmnd_f_step(current_values, rtol=rtol, atol=atol)
    return float(current_values[2])


def match_kwargs(
    func: Callable | Iterable[Callable] | str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Match keyword arguments to the parameters of a given function.

    Given a function and a dictionary of keyword arguments, return a
    dictionary of only those keyword arguments that match the parameters
    of the function. Typical usage is

    Parameters
    ----------
    func : callable or iterable of callable
        The function or functions to match keyword arguments against.
        If multiple functions, keywords will be matched against any
        of them (i.e., the union of all the functions' parameter names).
    kwargs : dict[str, Any]
        The dictionary of keyword arguments to match.

    Returns
    -------
    dict
        A dictionary of keyword arguments that match the parameters of
        the function, or the parameters of any of the functions.

    Notes
    -----
    The implementation of this function is based on inspect.signature.
    Some functions accept a generic `**kwargs` parameter, which is
    not included in the signature. This function does not handle such
    cases.

    Examples
    --------
    To call a function `func` on only the `kwargs` it accepts:

    >>> func(..., **_match_kwargs(func, kwargs))

    """
    if isinstance(func, str):
        return {
            key.removeprefix(func): value
            for key, value in kwargs.items()
            if key.startswith(func)
        }
    elif callable(func):
        params = [inspect.signature(func).parameters.keys()]
    else:
        params = [inspect.signature(f).parameters.keys() for f in func]
    intersected_kw_names = set(kwargs.keys()).intersection(set().union(*params))
    return {kw_name: kwargs[kw_name] for kw_name in intersected_kw_names}


def _dims_occurring_once(shape: tuple[int, ...]) -> set[int]:
    """Get dimensions that occur only once in the provided shape.

    Parameters
    ----------
    shape : Tuple[int]
        A tuple representing the shape of a numpy array.

    Returns
    -------
    set[int]
        A set of dimensions that occur only once in the input shape.
    """
    unique, counts = np.unique(np.asarray(shape), return_counts=True)
    return set(unique[counts == 1].tolist())


def _find_common_dimension(*args: np.ndarray, common_len=None) -> tuple[int, list[int]]:
    """Find the unique, common dimension length in all given arrays.

    This function identifies dimensions that appear only once in each shape
    and are common to all provided arrays.

    Parameters
    ----------
    *args : np.ndarray
        Variable length argument list of numpy arrays.

    Returns
    -------
    Tuple[int, List[int]]
        A tuple containing the common dimension length that occurs only
        once in all the arrays and a list of indices where this common
        dimension length occurs in each array's shape.

    Raises
    ------
    ValueError
        If no unique non-zero, non-unitary common dimensions are found
        between the arrays.
    """
    shapes = [arg.shape for arg in args]

    if common_len is None:
        individual_unique_dims = [_dims_occurring_once(shp) for shp in shapes]
        # Find common dimensions
        common_len = set.intersection(*individual_unique_dims) - {0, 1}
        if len(common_len) != 1:
            raise ValueError(
                "No unique non-zero, non-unitary common dimensions found "
                f"between arrays of shapes {shapes}."
            )

        common_len = common_len.pop()

    indices = [shape.index(common_len) for shape in shapes]
    return common_len, indices


def dropna(*args: Any, drop_inf: bool = False, common_len=None) -> tuple:
    """Drop rows containing NA values in any of the provided arrays.

    This function is designed for multi-dimensional numpy arrays (and objects convertible
    to them, like pandas Series). It assumes that the arrays share a common dimension
    (like rows in a 2D array) and drops any "row" with NaN or Inf values across the arrays.

    Parameters
    ----------
    *args : array-like
        Variable length argument list of objects. Expected to be numpy arrays
        or objects convertible to numpy arrays (e.g., pandas Series).
    drop_inf : bool, default False
        Whether to drop Inf values in addition to NaN values.

    Returns
    -------
    Tuple[Union[np.ndarray, pd.Series]]
        Tuple of arrays with the same dimensions as the input arrays, but with
        rows containing NaN or Inf values dropped. If the input was a pandas
        Series, the return type for that specific input will also be a pandas Series.

    Examples
    --------
    Consider two numpy arrays of shape (2, 3) and (2,):
    >>> a = np.array([[1, 2, np.nan], [4, 5, 6]])
    >>> b = np.array([7, np.nan])
    >>> dropna(a, b)
    (array([[4., 5., 6.]]), array([7.]))

    For arrays with a shape like (1, 14):
    The output retains the same number of dimensions, so an input of (1, 14)
    will produce an output shape like (1, n) depending on the number of NaN/Inf values.

    Note
    ----
    The function identifies a "common row dimension" which it uses to check for
    NaN or Inf values. This dimension is identified as a unique size that's present
    in all provided arrays.
    """
    arrs = [np.asarray(arg) for arg in args]

    common_len, common_dims = _find_common_dimension(*arrs, common_len=common_len)

    keep_func = np.isfinite if drop_inf else lambda x: np.logical_not(np.isnan(x))

    # Build masks for each array separately
    masks = []
    for arr, dim in zip(arrs, common_dims):
        moved_arr = np.moveaxis(arr, dim, 0)
        mask = keep_func(moved_arr).all(axis=tuple(range(1, arr.ndim)))
        masks.append(mask)

    # Combine masks to determine rows to keep
    keep_mask = np.all(masks, axis=0)

    arrs_na_dropped = (
        np.compress(keep_mask, arr, axis=dim) for arr, dim in zip(arrs, common_dims)
    )

    return tuple(arrs_na_dropped)


def _get_value_or_key(d: dict, key: Any, t: Optional[type | tuple] = None) -> Any:
    """Look up a value, or return the key if it is of a matching type.

    If the key is found in the dictionary, its value is returned. If
    not, the key itself is returned if it is of type t (if specified) or
    the same type as any of the dictionary's values (if not). If the key
    is not found and not of the specified type(s), a KeyError is raised.

    Parameters
    ----------
    d : dict
        The dictionary from which to retrieve the value.
    key : Any
        The key to look up in the dictionary.
    t : type or tuple of types, optional
        The expected type or types of the key. If not provided
        (default), the type(s) will be inferred from the values in the
        dictionary.

    Returns
    -------
    Any
        The value associated with the key in the dictionary, or the key
        itself if the key is not found and is of the specified type(s).
    """
    try:
        return d[key]
    except (TypeError, KeyError) as e:
        if t is None:
            t = tuple({type(v) for v in d.values()})
        if isinstance(key, t):
            return key
        raise KeyError(f"key {key} not recognized and not of type {t}") from e
