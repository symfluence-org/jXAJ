# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
jXAJ -- Xinanjiang (XAJ) Rainfall-Runoff Model plugin for SYMFLUENCE.

A clean-room JAX implementation of the saturation-excess model (Zhao, 1992),
the foundational model for Chinese operational hydrology.

This package registers itself with symfluence at runtime via the
``symfluence.plugins`` entry-point group.  The ``register()`` function
is called by ``symfluence.core._bootstrap._discover_plugins()`` and
populates all necessary registries.
"""

from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Lazy import mapping (identical to the original xinanjiang __init__)
# ---------------------------------------------------------------------------
_LAZY_IMPORTS = {
    # Configuration
    'XinanjiangConfig': ('.config', 'XinanjiangConfig'),
    'XinanjiangConfigAdapter': ('.config', 'XinanjiangConfigAdapter'),

    # Main components
    'XinanjiangPreProcessor': ('.preprocessor', 'XinanjiangPreProcessor'),
    'XinanjiangRunner': ('.runner', 'XinanjiangRunner'),
    'XinanjiangPostProcessor': ('.postprocessor', 'XinanjiangPostProcessor'),
    'XinanjiangResultExtractor': ('.extractor', 'XinanjiangResultExtractor'),

    # Parameters
    'PARAM_BOUNDS': ('.parameters', 'PARAM_BOUNDS'),
    'DEFAULT_PARAMS': ('.parameters', 'DEFAULT_PARAMS'),
    'PARAM_NAMES': ('.parameters', 'PARAM_NAMES'),
    'XinanjiangParams': ('.parameters', 'XinanjiangParams'),
    'XinanjiangState': ('.parameters', 'XinanjiangState'),

    # Core model
    'simulate': ('.model', 'simulate'),
    'simulate_jax': ('.model', 'simulate_jax'),
    'simulate_numpy': ('.model', 'simulate_numpy'),
    'HAS_JAX': ('.model', 'HAS_JAX'),

    # Loss functions
    'kge_loss': ('.losses', 'kge_loss'),
    'nse_loss': ('.losses', 'nse_loss'),
    'get_kge_gradient_fn': ('.losses', 'get_kge_gradient_fn'),
    'get_nse_gradient_fn': ('.losses', 'get_nse_gradient_fn'),

    # Forcing adapter
    'XinanjiangForcingAdapter': ('.forcing_adapter', 'XinanjiangForcingAdapter'),

    # Calibration
    'XinanjiangWorker': ('.calibration', 'XinanjiangWorker'),
    'XinanjiangParameterManager': ('.calibration', 'XinanjiangParameterManager'),
    'XinanjiangModelOptimizer': ('.calibration', 'XinanjiangModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for jXAJ module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys()) + ['register']


# ---------------------------------------------------------------------------
# Plugin registration — called by symfluence's _discover_plugins()
# ---------------------------------------------------------------------------

def register() -> None:
    """Register all Xinanjiang components with the symfluence registries.

    Called automatically via the ``symfluence.plugins`` entry-point group.
    Must NOT be called at import time — only when the entry-point loader
    invokes it.
    """
    from symfluence.core.registry import model_manifest

    from .calibration.optimizer import XinanjiangModelOptimizer
    from .calibration.parameter_manager import XinanjiangParameterManager
    from .calibration.worker import XinanjiangWorker
    from .config import XinanjiangConfigAdapter
    from .extractor import XinanjiangResultExtractor
    from .forcing_adapter import XinanjiangForcingAdapter
    from .postprocessor import XinanjiangPostProcessor
    from .preprocessor import XinanjiangPreProcessor
    from .runner import XinanjiangRunner

    model_manifest(
        "XINANJIANG",
        preprocessor=XinanjiangPreProcessor,
        runner=XinanjiangRunner,
        runner_method='run_xinanjiang',
        postprocessor=XinanjiangPostProcessor,
        config_adapter=XinanjiangConfigAdapter,
        result_extractor=XinanjiangResultExtractor,
        forcing_adapter=XinanjiangForcingAdapter,
        optimizer=XinanjiangModelOptimizer,
        worker=XinanjiangWorker,
        parameter_manager=XinanjiangParameterManager,
    )

    # Contribute Xinanjiang's calibration bounds to symfluence's catalogue.
    #
    # Xinanjiang predates the register_model_bounds seam, so symfluence carried
    # a get_xinanjiang_bounds() entry as a compatibility shim -- meaning a
    # change to Xinanjiang's bounds needed a FRAMEWORK release. Registering
    # here makes this package the owner: get_model_bounds('XINANJIANG')
    # resolves what we register, ahead of the built-in entry, so this works
    # against current symfluence and lets a later release drop the compat
    # entry entirely.
    #
    # Values come from jxaj.parameters.PARAM_BOUNDS with the log transform from
    # LOG_TRANSFORM_PARAMS, already the single source for every other bounds
    # consumer in this package, and verified identical to what the framework
    # served (13 names, zero differences, same log/linear split) -- so adopting
    # the seam changes no calibration result. Snow-17 coupling parameters are
    # owned by jsnow17 and stay out of this registration, matching the
    # catalogue's Xinanjiang entry.
    from symfluence.core.calibration.parameters import ParameterInfo, register_model_bounds

    from .parameters import LOG_TRANSFORM_PARAMS, PARAM_BOUNDS

    register_model_bounds(
        "XINANJIANG",
        params={
            name: ParameterInfo(
                float(lo),
                float(hi),
                description=f"Xinanjiang {name}",
                transform='log' if name in LOG_TRANSFORM_PARAMS else 'linear',
            )
            for name, (lo, hi) in PARAM_BOUNDS.items()
        },
        names=list(PARAM_BOUNDS),
    )


if TYPE_CHECKING:
    from .calibration import XinanjiangModelOptimizer, XinanjiangParameterManager, XinanjiangWorker
    from .config import XinanjiangConfig, XinanjiangConfigAdapter
    from .extractor import XinanjiangResultExtractor
    from .forcing_adapter import XinanjiangForcingAdapter
    from .losses import get_kge_gradient_fn, get_nse_gradient_fn, kge_loss, nse_loss
    from .model import HAS_JAX, simulate, simulate_jax, simulate_numpy
    from .parameters import (
        DEFAULT_PARAMS,
        PARAM_BOUNDS,
        PARAM_NAMES,
        XinanjiangParams,
        XinanjiangState,
    )
    from .postprocessor import XinanjiangPostProcessor
    from .preprocessor import XinanjiangPreProcessor
    from .runner import XinanjiangRunner


__all__ = [
    'XinanjiangConfig', 'XinanjiangConfigAdapter',
    'XinanjiangPreProcessor', 'XinanjiangRunner', 'XinanjiangPostProcessor', 'XinanjiangResultExtractor',
    'XinanjiangForcingAdapter',
    'PARAM_BOUNDS', 'DEFAULT_PARAMS', 'PARAM_NAMES', 'XinanjiangParams', 'XinanjiangState',
    'simulate', 'simulate_jax', 'simulate_numpy', 'HAS_JAX',
    'kge_loss', 'nse_loss', 'get_kge_gradient_fn', 'get_nse_gradient_fn',
    'XinanjiangWorker', 'XinanjiangParameterManager', 'XinanjiangModelOptimizer',
    'register',
]
