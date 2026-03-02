"""Tests for jXAJ plugin registration with symfluence."""

import pytest


class TestPluginRegistration:
    """Verify that jXAJ registers correctly with symfluence registries."""

    def test_register_populates_runners(self):
        """register() should add XINANJIANG to R.runners."""
        from jxaj import register
        from jxaj.runner import XinanjiangRunner
        from symfluence.core.registries import Registries as R

        register()
        assert 'XINANJIANG' in R.runners
        assert R.runners['XINANJIANG'] is XinanjiangRunner

    def test_register_populates_preprocessors(self):
        """register() should add XINANJIANG to R.preprocessors."""
        from jxaj import register
        from jxaj.preprocessor import XinanjiangPreProcessor
        from symfluence.core.registries import Registries as R

        register()
        assert 'XINANJIANG' in R.preprocessors
        assert R.preprocessors['XINANJIANG'] is XinanjiangPreProcessor

    def test_register_populates_postprocessors(self):
        """register() should add XINANJIANG to R.postprocessors."""
        from jxaj import register
        from jxaj.postprocessor import XinanjiangPostprocessor
        from symfluence.core.registries import Registries as R

        register()
        assert 'XINANJIANG' in R.postprocessors
        assert R.postprocessors['XINANJIANG'] is XinanjiangPostprocessor

    def test_register_populates_config_adapters(self):
        """register() should add XINANJIANG to R.config_adapters."""
        from jxaj import register
        from jxaj.config import XinanjiangConfigAdapter
        from symfluence.core.registries import Registries as R

        register()
        assert 'XINANJIANG' in R.config_adapters
        assert R.config_adapters['XINANJIANG'] is XinanjiangConfigAdapter

    def test_register_populates_result_extractors(self):
        """register() should add XINANJIANG to R.result_extractors."""
        from jxaj import register
        from jxaj.extractor import XinanjiangResultExtractor
        from symfluence.core.registries import Registries as R

        register()
        assert 'XINANJIANG' in R.result_extractors
        assert R.result_extractors['XINANJIANG'] is XinanjiangResultExtractor

    def test_register_populates_forcing_adapters(self):
        """register() should add XINANJIANG to R.forcing_adapters."""
        from jxaj import register
        from jxaj.forcing_adapter import XinanjiangForcingAdapter
        from symfluence.core.registries import Registries as R

        register()
        assert 'XINANJIANG' in R.forcing_adapters
        assert R.forcing_adapters['XINANJIANG'] is XinanjiangForcingAdapter

    def test_register_populates_optimizers(self):
        """register() should add XINANJIANG to R.optimizers."""
        from jxaj import register
        from jxaj.calibration.optimizer import XinanjiangModelOptimizer
        from symfluence.core.registries import Registries as R

        register()
        assert 'XINANJIANG' in R.optimizers
        assert R.optimizers['XINANJIANG'] is XinanjiangModelOptimizer

    def test_register_populates_workers(self):
        """register() should add XINANJIANG to R.workers."""
        from jxaj import register
        from jxaj.calibration.worker import XinanjiangWorker
        from symfluence.core.registries import Registries as R

        register()
        assert 'XINANJIANG' in R.workers
        assert R.workers['XINANJIANG'] is XinanjiangWorker

    def test_register_populates_parameter_managers(self):
        """register() should add XINANJIANG to R.parameter_managers."""
        from jxaj import register
        from jxaj.calibration.parameter_manager import XinanjiangParameterManager
        from symfluence.core.registries import Registries as R

        register()
        assert 'XINANJIANG' in R.parameter_managers
        assert R.parameter_managers['XINANJIANG'] is XinanjiangParameterManager

    def test_entry_point_discoverable(self):
        """Entry point should be discoverable via importlib.metadata."""
        from importlib.metadata import entry_points

        eps = entry_points(group='symfluence.plugins')
        ep_names = [ep.name for ep in eps]
        assert 'xinanjiang' in ep_names

    def test_register_idempotent(self):
        """Calling register() twice should not raise."""
        from jxaj import register
        register()
        register()  # Should not raise
