import numpy as np


def _flatten_helper(data, order: str = "C") -> np.ndarray:
    flat_data = np.ravel(np.asarray(data), order=order)
    assert flat_data.dtype != np.dtype("O")
    return flat_data


def flatten(data, order: str = "C") -> np.ndarray:
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
    flat_data : array, list, or primitive
        Flattened version of **data**. If on_bad_data="error",
        always an array.

    """

    try:
        return _flatten_helper(data, order=order)
    except (TypeError, RuntimeError, AssertionError):
        if hasattr(data, "__iter__") or hasattr(data, "__getitem__"):
            flat_data = []
            for datum in data:
                try:
                    flat_data.extend(flatten(datum, order=order))
                except TypeError:
                    flat_data.append(flatten(datum, order=order))
        else:
            flat_data = data
    return _flatten_helper(flat_data, order=order)
