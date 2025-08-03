#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TorchRec Distributed Benchmarking Package

This package provides unified benchmarking capabilities for both module-level
and pipeline-level performance testing of distributed embedding models.
"""

# Unified benchmark system
from .unified_benchmark_config import (
    UnifiedBenchmarkConfig,
    ModuleConfig,
    import_module_class,
    create_module_from_config,
    validate_config,
)

# Legacy benchmark utilities (maintained for backward compatibility)
from .benchmark_utils import (
    benchmark_module,
    benchmark_func,
    BenchmarkResult,
    generate_tables,
    generate_planner,
    generate_sharded_model_and_optimizer,
)

# Pipeline benchmarking
from .benchmark_train_pipeline import (
    RunOptions,
    EmbeddingTablesConfig,
    PipelineConfig,
    ModelSelectionConfig,
    run_pipeline,
)

from .benchmark_pipeline_utils import (
    BaseModelConfig,
    create_model_config,
    generate_pipeline,
    generate_data,
)

__all__ = [
    # Unified system
    "UnifiedBenchmarkConfig",
    "ModuleConfig", 
    "import_module_class",
    "create_module_from_config",
    "validate_config",
    # Benchmark utilities
    "benchmark_module",
    "benchmark_func",
    "BenchmarkResult",
    "generate_tables",
    "generate_planner", 
    "generate_sharded_model_and_optimizer",
    # Pipeline benchmarking
    "RunOptions",
    "EmbeddingTablesConfig",
    "PipelineConfig",
    "ModelSelectionConfig",
    "run_pipeline",
    "BaseModelConfig",
    "create_model_config",
    "generate_pipeline",
    "generate_data",
]
