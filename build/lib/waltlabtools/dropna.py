import warnings
from collections.abc import Iterable
from typing import Optional

import numpy as np
from sklearn.utils.validation import indexable


def _match_axes_size(shapes, common_len):
    """Helper function to choose axes based on the common length."""
    num_axes = [np.sum(np.array(shape) == common_len) > 1 for shape in shapes]
    if any(num_axes):
        warnings.warn(
            "dropna input includes square array; choosing the "
            f"first axis with length {common_len} from shapes {shapes}"
        )
    axis = [np.argmax(np.array(shape) == common_len) for shape in shapes]
    return axis


def _find_axes(*args) -> tuple[tuple | list, list]:
    try:
        args = indexable(*args)
    except ValueError as e:
        shapes = [np.shape(arg) for arg in args]
        common_lengths = set.intersection(*(set(shape) for shape in shapes))
        if len(common_lengths) == 0:
            raise ValueError(
                f"dropna found input variables with irreconcilable shapes: {shapes}"
            ) from e
        elif 0 in common_lengths:
            raise ValueError(
                f"dropna: all input arrays are empty: they have shapes {shapes}"
            ) from e
        elif len(common_lengths) == 1:
            common_len = common_lengths.pop()
            axis = _match_axes_size(shapes, common_len)
        else:  # len(common_lengths) > 1
            first_axis = (np.argmax(np.array(shape) > 1) for shape in shapes)
            first_lengths = {
                shape[ax] for shape, ax in zip(shapes, first_axis)
            }.intersection(common_lengths)
            common_len = first_lengths.pop()
            if first_lengths:
                warnings.warn(
                    "dropna was given arrays with common lengths "
                    f"{common_lengths}; choosing {common_len} for shapes {shapes}"
                )
            axis = _match_axes_size(shapes, common_len)
    else:
        axis = [0] * len(args)
    return args, axis


def align(
    *args,
    drop_inf: bool = False,
    axis: Optional[int | Iterable[int]] = None,
    common_len: Optional[int] = None,
):
    if common_len is None and axis is None:
        args, axis = _find_axes(*args)
    elif axis is not None and common_len is None:
        if isinstance(axis, int):
            axis = [axis] * len(args)
        common_lengths = {np.shape(arg)[ax] for arg, ax in zip(args, axis)}
        if len(common_lengths) == 1:
            common_len = common_lengths.pop()
        else:
            raise ValueError(
                "dropna found input variables with different lengths "
                f"{common_lengths} along the provided axis: {axis}"
            )
    elif axis is not None and common_len is not None:
        if isinstance(axis, int):
            axis = [axis] * len(args)
        common_lengths = {np.shape(arg)[ax] for arg, ax in zip(args, axis)}
        if (len(common_lengths) != 1) or (common_len not in common_lengths):
            raise ValueError(
                f"dropna found input variables whose lengths {common_lengths} "
                f"along the provided axis {axis} did not match common_len={common_len}"
            )
    elif common_len is not None and axis is None:
        shapes = [np.shape(arg) for arg in args]

        axis = _match_axes_size(shapes, common_len)

    return args, axis, common_len

    # find the axis along which to drop
    #     axes: Iterable[int], length = len(args)
    #     common_len: int = np.shape(arg)[ax] for arg, ax in zip(args, axes)
    # assemble the mask
    #     mask: ArrayLike[bool], shape (common_len,)
    # for each arg:
    #     drop rows


def dropna(
    *args,
    drop_inf: bool = False,
    axis: Optional[int | Iterable[int]] = 0,
    common_len: Optional[int] = None,
):
    keep_func = np.isfinite if drop_inf else lambda x: np.logical_not(np.isnan(x))

    if axis == 0:
        args = indexable(args)
        masks = []
        for arg in args:
            mask = np.all(keep_func(arg), axis=tuple(range(1, np.ndim(arg))))
            masks.append(mask)
        keep_mask = np.all(masks, axis=0)
        args_na_dropped = (np.compress(keep_mask, arg, axis=0) for arg in args)

    return tuple(args_na_dropped)
