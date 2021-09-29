"""Core functionality for the waltlabtools module.

Includes the classes Model and CalCurve, and core functions for assay
analysis.

Everything in waltlabtools.core is automatically imported with
waltlabtools, so it can be accessed via, e.g.,

.. code-block:: python

   import waltlabtools as wlt  # waltlabtools main functionality

   cal_curve = wlt.CalCurve()  # creates a new empty calibration curve


-----


"""

import warnings
import sys
import importlib

import scipy.special as spec
if "jax" in sys.modules.keys():
    import jax.numpy as np
    from jax import jit
    _use_jax = True
else:
    import numpy as np

    def jit(fun):
        return fun
    _use_jax = False


__all__ = ["flatten", "dropna", "aeb", "fon", "c4", "gmnd"]


_optional_dependencies = {package_name: (importlib.util.find_spec(package_name)
        is not None)
    for package_name in ["matplotlib", "pandas", "sklearn", "tkinter"]
}
_optional_dependencies["jax"] = _use_jax


_ddofs = {"n": 0, "n-1": 1, "n-1.5": 1.5, "c4": 1}


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

            - "warn" : Returns as in "ignore", but raises a warning.

    Returns
    -------
    flattened_data : array, list, or primitive
        Flattened version of **data**. If on_bad_data="error",
        always an array.

    """
    try:
        return np.ravel(np.asarray(data))
    except Exception:
        if hasattr(data, "__iter__"):
            flat_data = []
            for datum in data:
                if hasattr(datum, "__iter__"):
                    flat_data.extend(flatten(datum))
                else:
                    flat_data.append(datum)
        else:
            flat_data = data
        try:
            return np.asarray(flat_data)
        except Exception:
            if on_bad_data != "ignore":
                error_text = " ".join(["Input data were coerced from type",
                    str(type(data)), "to type", str(type(flat_data)),
                    "but could not be coerced to an ndarray."])
                if on_bad_data == "error":
                    raise TypeError(error_text)
                else:
                    warnings.warn(error_text, Warning)
            return flat_data


def dropna(datasets):
    """Returns arrays, keeping only rows where all datasets are finite.

    Parameters
    ----------
    datsets : iterable of array-likes
        The data structures to be flattened. They must be the same size
        along their first axis.

    Returns
    -------
    flattened_datasets : list of arrays
        Flattened version of **data**. If on_bad_data="error",
        always an array.

    """
    arrayed_datasets = []
    sizes = set()
    for data in datasets:
        arrayed_data = np.asarray(data)
        arrayed_datasets.append(arrayed_data)
        sizes.add(np.shape(arrayed_data)[0])
    if len(sizes) == 1:
        common_size = sizes.pop()
        notna_array = np.ones(common_size, dtype=bool)
        for arrayed_data in arrayed_datasets:
            data_finitude = np.all(np.isfinite(arrayed_data),
                axis=tuple(range(1, arrayed_data.ndim)))
            notna_array = np.logical_and(notna_array, data_finitude)
        return [arrayed_data[notna_array] for arrayed_data in arrayed_datasets]
    else:
        raise IndexError(
            "Datasets must have the same size along their first axis. "
            + "The supplied datasets have sizes of " + str(sizes) + ".")


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
        return -np.log(1 - flatten(fon_))


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


def std(data, corr="c4"):
    flat_data = flatten(data)
    try:
        ddof = _ddofs[corr]
    except KeyError:
        try:
            ddof = float(corr)
        except ValueError:
            raise ValueError(str(corr)
                + " is not a valid correction factor. Please try another one.")
    corr_factor = c4(len(flat_data)) if (corr == "c4") else 1
    return np.std(flat_data, ddof=ddof)/corr_factor


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
        warnings.warn(" ".join([
            "Geometric mean requires all numbers to be nonnegative.",
            "Because the data provided included", str(float(data_amin)), ",",
            "the geometric meandian is unlikely to provide any insight."]))
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
