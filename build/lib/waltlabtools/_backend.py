import sys
from collections.abc import Callable

_MODULES_CHECK: dict[str, bool] = {
    "jax": "jax" in sys.modules,
    "numba": True,
}

if _MODULES_CHECK["jax"]:
    import jax.numpy as np
    from jax import jit
    from jax.scipy.special import gammaln

    def gmean(a, axis: int = 0, dtype=None, weights=None):
        """Geometric mean."""
        return np.exp(np.average(np.log(a), axis=axis, weights=weights))

    import warnings

    warnings.warn("JAX backend is experimental and may not work as expected.")  # type: ignore
else:
    import numpy as np
    from scipy.special import gammaln
    from scipy.stats import gmean

    try:
        from numba import jit  # type: ignore
    except ImportError:
        _MODULES_CHECK["numba"] = False

        def jit(fun: Callable) -> Callable:
            """Dummy jit decorator for when numba/jax not installed."""
            return fun
