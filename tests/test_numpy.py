use_jax = False

import pytest
import pandas as pd

if use_jax:
    import jax.numpy as np
else:
    import numpy as np

import waltlabtools as wlt
from waltlabtools.core import _optional_dependencies

if use_jax:
    import numpy as xnp
else:
    import jax.numpy as xnp

# ["aeb", "fon", "c4", "gmnd"]
# ["regress", "lod", "CalCurve"]
# ["Model", "model_dict"]


# waltlabtools.core._optional_dependencies
def test_imports():
    assert _optional_dependencies["matplotlib"]
    assert _optional_dependencies["pandas"]
    assert _optional_dependencies["sklearn"]
    assert _optional_dependencies["tkinter"]
    assert _optional_dependencies["numba"]
    assert _optional_dependencies["jax"] == use_jax


# waltlabtools.core.flatten
items_to_flatten = (
    [0, 1, 2, 3],
    (0, 1, 2, 3),
    {0, 1, 2, 3},
    np.asarray([0, 1, 2, 3]),
    np.asarray([[0, 1], [2, 3]]),
    xnp.asarray([0, 1, 2, 3]),
    xnp.asarray([[0, 1], [2, 3]]),
    [[0, 1], [2, 3]],
    [0, [1, [2, [3]]]],
    pd.Series([0, 1, 2, 3]),
    pd.DataFrame([0, 1, 2, 3]),
    pd.DataFrame([[0, 1], [2, 3]]),
    range(4)
)
flat_data_should_be = np.asarray([0, 1, 2, 3])

@pytest.mark.parametrize("data", items_to_flatten)
def test_flatten_0123(data):
    flat_data = wlt.flatten(data)
    assert np.array_equal(flat_data, flat_data_should_be)


# waltlabtools.core.dropna
items_to_dropna = (
    (((0.1, 1.1, 2.1, 3.1), (1, 2, 3, 4)), range(4)),
    ([[0.1, np.nan, 1.1, 2.1, 3.1], [1, 1, 2, 3, 4]], [0, np.pi, 1, 2, 3]),
    ([[0.1, 1.1, 2.1, 3.1, np.inf], [1, 2, 3, 4, 5]], [0, 1, 2, 3, -np.pi]),
    ([[-np.inf, 0.1, 1.1, 2.1, 3.1], [0, 1, 2, 3, 4]], [np.pi, 0, 1, 2, 3]),
    ([[-np.inf, 0.1, 1.1, 2.1, 3.1], [np.nan, 1, 2, 3, 4]], [70, 0, 1, 2, 3])
)
nas_dropped_should_be = [np.asarray([[0.1, 1.1, 2.1, 3.1], [1, 2, 3, 4]]),
    np.asarray([0, 1, 2, 3])]

@pytest.mark.parametrize("datasets", items_to_dropna)
def test_dropna_0123(datasets):
    nas_dropped = wlt.dropna(datasets)
    assert all([np.array_equal(nas_dropped[a], nas_dropped_should_be[a])
        for a in range(len(nas_dropped_should_be))])
    assert len(nas_dropped_should_be) == len(nas_dropped)


def test_gmnd():
    my_gmnd = wlt.gmnd([1, 16, 2, 8])
    assert my_gmnd["gmnd"] == 5.127209423437007
    assert my_gmnd["median"] == 5
    assert my_gmnd["geometric"] == 4
    assert my_gmnd["arithmetic"] == 6.75
