"""Core functionality for the waltlabtools module.

Includes the classes Model and CalCurve, and core functions for assay
analysis.

Everything in waltlabtools.core is automatically imported with
waltlabtools, so it can be accessed via, e.g.,

.. code-block:: python

   import waltlabtools as wlt  # waltlabtools main functionality

   my_data = wlt.flatten([[[[[0], 1], 2], 3], 4])  # flatten a list


-----


"""

import importlib
import sys

import scipy.special as spec

_optional_dependencies = {
    package_name: (importlib.util.find_spec(package_name) is not None)
    for package_name in ["matplotlib", "pandas", "sklearn", "numba"]
}

if "jax" in sys.modules:
    import jax.numpy as np
    from jax import jit

    _optional_dependencies["jax"] = True
else:
    import numpy as np

    if _optional_dependencies["numba"]:
        from numba import jit
    else:

        def jit(func):
            """Dummy jit decorator for when numba is not installed."""
            return func

    _optional_dependencies["jax"] = False


__all__ = ["flatten", "dropna", "aeb", "fon", "c4", "std", "gmnd"]

_DDOFS = {"n": 0, "n-1": 1, "n-1.5": 1.5, "c4": 1}


def _flatten_helper(data, order: str = "C") -> np.ndarray:
    """Flattens data structures that can easily be flattened.

    Raises
    ------
    AssertionError
        If the numpy.ravel returns an array with dtype=object.

    """
    flat_data = np.ravel(np.array(data), order=order)
    assert flat_data.dtype != np.dtype("O")
    return flat_data


def flatten(data, order: str = "C") -> np.ndarray:
    """Flattens most data structures.

    Parameters
    ----------
    data : any
        The data structure to be flattened. Can also be a primitive.

    Returns
    -------
    flat_data : array
        Flattened version of **data**.

    Other Parameters
    ----------------
    order : {"C","F", "A", "K"}, default "C"
        The elements of data are read using this index order. "C" means
        to index the elements in row-major, C-style order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. For full details, see
        numpy.ravel.

    See Also
    --------
    numpy.array : Returns a new array from an array-like object.

    numpy.ravel : Flattens a numpy array.

    """
    try:
        return _flatten_helper(data, order=order)
    except (TypeError, RuntimeError, AssertionError, ValueError):
        flat_data: list = []
        for datum in data:
            try:
                flat_data.extend(flatten(datum, order=order))
            except TypeError:
                flat_data.append(flatten(datum, order=order))
    return _flatten_helper(flat_data, order=order)


def _na_sieve(
    arrayed_datasets: list, common_size: int, common_dims: list, drop_inf: bool = True
) -> list:
    """Compresses away the rows where any dataset has a NaN value.

    Helper function for dropna.

    Parameters
    ----------
    arrayed_datasets : list
        The data structures to be flattened. They must be the same size
        along their first axis or share one common dimension.
    common_size : int
        The length of the common axis that all arrays share.
    common_dims : list
        A list of the dimension in each array which is the dimension
        along which to drop NaNs.
    drop_inf : bool, optional
        If True (default), drop with non-finite or NaN values. If False,
        drop only rows with NaN values.

    Returns
    -------
    list of arrays
        The datasets provided, with rows removed in which
        any of the arrays has a non-finite value.

    """
    notna_array = np.ones(common_size, dtype=bool)
    keep_fn = np.isfinite if drop_inf else lambda x: np.invert(np.isnan(x))
    for a, arrayed_data in enumerate(arrayed_datasets):
        axis_list = list(range(arrayed_data.ndim))
        axis_list.remove(common_dims[a])
        axis = tuple(axis_list)
        data_finitude = np.all(keep_fn(arrayed_data), axis=axis)
        notna_array = np.logical_and(notna_array, data_finitude)
    return [
        np.compress(condition=notna_array, a=arrayed_datasets[i], axis=common_dims[i])
        for i in range(len(common_dims))
    ]


def dropna(datasets, drop_inf: bool = True):
    """Drops rows in which any dataset has a missing values.

    Parameters
    ----------
    datsets : iterable of array-likes
        The data structures to be flattened. They must be the same size
        along their first axis or share one common dimension.
    drop_inf : bool, default True
        If True (default), drop with non-finite or NaN values. If False,
        drop only rows with NaN values.

    Returns
    -------
    list of arrays
        The datasets provided, now as arrays, with rows removed in which
        any of the arrays has a non-finite value.

    """
    arrayed_datasets = [np.array(dataset) for dataset in datasets]
    dataset_shapes = [dataset.shape for dataset in arrayed_datasets]

    # First option: along any dimension, every dataset is the same size.
    max_dim = max(len(shape) for shape in dataset_shapes)
    possible_dims = {}
    for dim in range(1 - max_dim, max_dim):
        possible_sizes = {shape[dim] for shape in dataset_shapes}
        if len(possible_sizes) == 1:
            possible_dims[dim] = possible_sizes.pop()
    if len(possible_dims) == 1:
        common_dim, common_size = possible_dims.popitem()
        common_dims = [common_dim] * len(arrayed_datasets)

    # Second option: there is one size shared by all datasets.
    else:
        possible_sizes = (set(dataset_shapes[0]) - {0, 1}).intersection(*dataset_shapes)
        if len(possible_sizes) == 0:
            raise ValueError("The datasets do not have any common sizes.")
        if len(possible_sizes) > 1:
            raise ValueError("The datasets have multiple common sizes.")

        common_size = possible_sizes.pop()
        common_dims = []
        for shape in dataset_shapes:
            if shape.count(common_size) == 1:
                raise ValueError("A dataset has multiple dimensions of the same size.")
            common_dims.append(shape.index(common_size))

    return _na_sieve(
        arrayed_datasets=arrayed_datasets,
        common_size=common_size,
        common_dims=common_dims,
        drop_inf=drop_inf,
    )


def aeb(fon_):
    """The average number of enzymes per bead.

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
    try:
        return -np.log(1 - fon_)
    except TypeError:
        return -np.log(1 - np.array(fon_))


def fon(aeb_):
    """The fraction of beads which are on.

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
    aeb : inverse of fon

    """
    try:
        return 1 - np.exp(-aeb_)
    except TypeError:
        return 1 - np.exp(-np.array(aeb_))


def c4(n):
    """Factor `c4` for unbiased estimation of the standard deviation.

    For a finite sample, the sample standard deviation tends to
    underestimate the population standard deviation. See, e.g.,
    https://www.spcpress.com/pdf/DJW353.pdf for details. Dividing the
    sample standard deviation by the correction factor `c4` gives an
    unbiased estimator of the population standard deviation.

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
    return np.sqrt(2 / (n - 1)) * spec.gamma(n / 2) / spec.gamma((n - 1) / 2)


def std(data, corr="c4") -> float:
    """Unbiased estimate of the population standard deviation.

    Unlike the corresponding function in numpy, this function allows
    specifying of the correction factor more generally in order to
    provide an unbiased estimate of the standard deviation.

    Parameters
    ----------
    data : numeric or array
        The data for which to take the standard deviation. Will be
        coerced to a 1-D numpy array via flatten.

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
    float
        Unbiased estimate of the standard deviation.

    See Also
    --------
    c4 : correction factor used when corr="c4"

    numpy.std : standard deviation

    """
    flat_data = flatten(data)
    try:
        ddof = _DDOFS[corr]
    except KeyError:
        try:
            ddof = float(corr)
        except ValueError as exc:
            error_text = (
                str(corr) + " is not a valid correction factor. Please try another one."
            )
            raise ValueError(error_text) from exc
    corr_factor = c4(len(flat_data)) if (corr == "c4") else 1
    return float(np.std(flat_data, ddof=ddof) / corr_factor)


def gmnd(data, rtol: float = 1e-05, atol: float = 1e-08) -> float:
    """Geometric meandian.

    For details, see https://xkcd.com/2435/. This function compares
    the three most common measures of central tendency: the arithmetic
    mean, the geometric mean, and the median.
    The geometric meandian uses an iterative process that stops when
    the arithmetic mean, geometric mean, and median converge within
    (`atol` + `rtol` * their magnitudes).

    Parameters
    ----------
    data : array-like
        The data for which to take the measure of central tendency.
    rtol : float, default 1e-05
        The relative tolerance parameter.
    atol : float, default 1e-08
        The absolute tolerance parameter.

    Returns
    -------
    gmnd_ : float
        Geometric meandian.

    Notes
    -----
    The purpose of this function is to demonstrate the functional
    decomposition necessary to perform just-in-time compilation (jit)
    using jax or numba.

    """
    flat_data = flatten(data)
    mean_ = np.mean(flat_data)
    geomean_ = np.exp(np.mean(np.log(flat_data)))
    median_ = np.median(flat_data)
    data_i = np.array((mean_, geomean_, median_))
    converged = False
    while not converged:
        data_i, converged = _gmnd_f(data_i, rtol=rtol, atol=atol)
    gmnd_ = data_i[0]
    return float(gmnd_)


@jit
def _gmnd_f(data_i: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> tuple:
    """Backend for geometric meandian.

    Parameters
    ----------
    data_i : array of length 3
        The current iteration's arithmetic mean, geometric mean, and
        median.
    rtol : float, default 1e-05
        The relative tolerance parameter.
    atol : float, default 1e-08
        The absolute tolerance parameter.

    Returns
    -------
    data_iplus1 : array of length 3
        The next iteration's arithmetic mean, geometric mean, and
        median.
    converged : bool
        Whether the next iteration has converged within the given
        tolerance.

    """
    mean_ = np.mean(data_i)
    geomean_ = np.exp(np.mean(np.log(data_i)))
    median_ = np.median(data_i)
    data_iplus1 = np.array((mean_, geomean_, median_))
    if np.isnan(data_iplus1).any():
        return np.array((np.nan, np.nan, np.nan)), True
    converged = (
        np.isclose(data_iplus1[0], data_i[1], rtol=rtol, atol=atol)
        & np.isclose(data_iplus1[1], data_i[2], rtol=rtol, atol=atol)
        & np.isclose(data_iplus1[2], data_i[0], rtol=rtol, atol=atol)
    )
    return data_iplus1, converged
