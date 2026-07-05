# jXAJ

[![PyPI version](https://img.shields.io/pypi/v/jxaj.svg)](https://pypi.org/project/jxaj/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Dual-backend (JAX + NumPy) implementation of the Xinanjiang (XAJ) saturation-excess rainfall–runoff model, with optional Snow-17 coupling — usable standalone or as a SYMFLUENCE plugin.

Part of the SYMFLUENCE JAX-native model family — self-contained packages that
run standalone (NumPy fallback, no JAX required) and register automatically with
[SYMFLUENCE](https://github.com/symfluence-org/SYMFLUENCE) when installed alongside it.

## Features

- **Differentiable**: automatic differentiation through the full simulation (JAX)
- **Fast**: JIT compilation via `lax.scan`; `vmap` for ensembles; GPU-capable
- **Dependency-light**: pure-NumPy fallback when JAX is not installed
- **Plugin architecture**: auto-registers with SYMFLUENCE via entry points

## Installation

```bash
pip install jxaj          # NumPy backend
pip install 'jxaj[jax]'    # with JAX (differentiable, JIT)
```

## Quickstart

```python
from jxaj.model import simulate

# snow-free basin: precipitation and PET only
flow, state = simulate(precip, pet)

# snow-affected basin: couple Snow-17 by passing temperature and day-of-year
flow, state = simulate(precip, pet, temp=temp, day_of_year=doy, latitude=51.17)
```

## Gradient-based calibration

The JAX backend makes the full simulation differentiable end-to-end, so model
parameters can be calibrated with gradient descent:

```python
import jax
from jxaj.losses import kge_loss, get_kge_gradient_fn

grad_fn = get_kge_gradient_fn(precip, temp, pet, observed)
value, grads = grad_fn(params)          # dKGE/dparam for every parameter
```

`nse_loss` / `kge_loss` and their gradient factories are JIT-compatible and work
with any `optax` optimizer. Within SYMFLUENCE the same interface powers the
ADAM and L-BFGS calibration options.

## Use with SYMFLUENCE

jxaj registers with [SYMFLUENCE](https://github.com/symfluence-org/SYMFLUENCE)
through the `symfluence.plugins` entry point — installation is the integration:

```bash
pip install symfluence jxaj
```

```yaml
# config.yaml (excerpt)
model:
  hydrological_model: XINANJIANG
```

SYMFLUENCE then handles forcing preparation, calibration, evaluation, and
benchmarking for the model with no further wiring.

## Model structure

The Xinanjiang model (Zhao, 1992) is a saturation-excess model with four components,
implemented clean-room from the published equations:

1. **Evapotranspiration** — three-layer (upper/lower/deep) with PET correction
2. **Runoff generation** — saturation excess with parabolic storage-capacity curve
3. **Source separation** — free-water storage partitioning into surface flow,
   interflow, and groundwater
4. **Routing** — linear reservoirs for interflow and groundwater

13 calibration parameters (`jxaj.parameters.PARAM_BOUNDS`). For snow-affected
basins the model couples to Snow-17 (`snow17_params`, `temp`, `day_of_year`
arguments; see also `simulate_coupled_jax`).

## Testing

```bash
pip install -e '.[dev]'
pytest
```

## How to cite

If you use jXAJ in your research, please cite the SYMFLUENCE companion papers,
which describe the design of the JAX-native model family (registry integration,
differentiability, and the calibration experiments they enable):

> Eythorsson, D., et al. (2026). The registry as social contract: Architectural patterns
> for community hydrological modeling. *Water Resources Research* (submitted).
>
> Eythorsson, D., et al. (2026). From configuration to prediction: Multi-model,
> multi-basin experiments with SYMFLUENCE. *Water Resources Research* (submitted).

Citation metadata for this package is provided in [`CITATION.cff`](CITATION.cff);
a version-specific DOI is minted via Zenodo for each GitHub release.
<!-- After the first Zenodo release, add the concept-DOI badge here. -->

## References

- Zhao, R.-J. (1992). The Xinanjiang model applied in China. Journal of Hydrology, 135(1–4), 371–381. https://doi.org/10.1016/0022-1694(92)90096-E
- Anderson, E. A. (2006). Snow Accumulation and Ablation Model — SNOW-17. NOAA Technical Report NWS HYDRO-17.

## License

Apache-2.0. See [LICENSE](LICENSE).
