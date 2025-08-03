#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Test suite for the unified benchmark system.

This module provides unit tests to verify that the unified YAML configuration
system works correctly for both module and pipeline benchmarking.
"""

import tempfile
import unittest
from typing import Any, Dict

import torch
import yaml
from torch import nn
from torchrec.distributed.benchmark.unified_benchmark_config import (
    create_module_from_config,
    import_module_class,
    ModuleConfig,
    UnifiedBenchmarkConfig,
    validate_config,
)
from torchrec.distributed.benchmark.unified_benchmark_runner import (
    dict_to_unified_config,
    load_config_from_yaml,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TestUnifiedBenchmark(unittest.TestCase):
    """Test cases for the unified benchmark system."""

    def test_import_module_class(self) -> None:
        """Test dynamic module class importing."""
        # Test importing a known class
        cls = import_module_class("torchrec.modules.embedding_modules", "EmbeddingBagCollection")
        self.assertTrue(issubclass(cls, nn.Module))
        
        # Test invalid module path
        with self.assertRaises(ImportError):
            import_module_class("nonexistent.module", "SomeClass")
            
        # Test invalid class name
        with self.assertRaises(AttributeError):
            import_module_class("torch.nn", "NonexistentClass")

    def test_module_config_creation(self) -> None:
        """Test creating modules from ModuleConfig."""
        # Test EmbeddingBagCollection creation
        tables = [
            EmbeddingBagConfig(
                name="table_0",
                embedding_dim=128,
                num_embeddings=1000,
                feature_names=["feature_0"],
            )
        ]
        
        module_config = ModuleConfig(
            module_path="torchrec.modules.embedding_modules",
            class_name="EmbeddingBagCollection",
            requires_tables=True,
            constructor_args={"device": "meta"},
        )
        
        module = create_module_from_config(module_config, tables=tables)
        self.assertIsInstance(module, nn.Module)
        self.assertEqual(module.device.type, "meta")

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        from torchrec.distributed.benchmark.benchmark_train_pipeline import (
            EmbeddingTablesConfig,
            RunOptions,
        )
        
        # Valid module config
        valid_module_config = UnifiedBenchmarkConfig(
            benchmark_type="module",
            run_options=RunOptions(),
            table_config=EmbeddingTablesConfig(),
            module_config=ModuleConfig(
                module_path="torch.nn",
                class_name="Linear",
                constructor_args={"in_features": 10, "out_features": 1},
            ),
        )
        validate_config(valid_module_config)  # Should not raise
        
        # Invalid benchmark type
        invalid_config = UnifiedBenchmarkConfig(
            benchmark_type="invalid",
            run_options=RunOptions(),
            table_config=EmbeddingTablesConfig(),
        )
        with self.assertRaises(ValueError):
            validate_config(invalid_config)
        
        # Module config without module_config
        missing_module_config = UnifiedBenchmarkConfig(
            benchmark_type="module",
            run_options=RunOptions(),
            table_config=EmbeddingTablesConfig(),
            module_config=None,
        )
        with self.assertRaises(ValueError):
            validate_config(missing_module_config)

    def test_yaml_loading(self) -> None:
        """Test loading configuration from YAML."""
        yaml_content = """
benchmark_type: "module"

run_options:
  world_size: 2
  num_batches: 5

table_config:
  num_unweighted_features: 10
  embedding_feature_dim: 64

module_config:
  module_path: "torch.nn"
  class_name: "Linear"
  constructor_args:
    in_features: 10
    out_features: 1
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = load_config_from_yaml(f.name)
            self.assertEqual(config.benchmark_type, "module")
            self.assertEqual(config.run_options.world_size, 2)
            self.assertEqual(config.run_options.num_batches, 5)
            self.assertEqual(config.table_config.num_unweighted_features, 10)
            self.assertIsNotNone(config.module_config)
            self.assertEqual(config.module_config.module_path, "torch.nn")
            self.assertEqual(config.module_config.class_name, "Linear")

    def test_dict_to_config_conversion(self) -> None:
        """Test converting dictionary to UnifiedBenchmarkConfig."""
        config_dict = {
            "benchmark_type": "pipeline",
            "run_options": {
                "world_size": 4,
                "num_batches": 20,
            },
            "table_config": {
                "num_unweighted_features": 50,
                "embedding_feature_dim": 256,
            },
            "pipeline_config": {
                "pipeline": "sparse",
                "apply_jit": False,
            },
            "model_selection": {
                "model_name": "test_sparse_nn",
                "batch_size": 4096,
            },
        }
        
        config = dict_to_unified_config(config_dict)
        self.assertEqual(config.benchmark_type, "pipeline")
        self.assertEqual(config.run_options.world_size, 4)
        self.assertEqual(config.table_config.num_unweighted_features, 50)
        self.assertIsNotNone(config.pipeline_config)
        self.assertEqual(config.pipeline_config.pipeline, "sparse")
        self.assertIsNotNone(config.model_selection)
        self.assertEqual(config.model_selection["model_name"], "test_sparse_nn")


if __name__ == "__main__":
    unittest.main()