#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Unified benchmark runner that can handle both module and pipeline level benchmarking
from a single YAML configuration file.
"""

import logging
from dataclasses import fields

from torchrec.distributed.benchmark.benchmark_pipeline_utils import (
    BaseModelConfig,
    create_model_config,
)
from torchrec.distributed.benchmark.benchmark_train_pipeline import (
    BenchmarkType,
    EmbeddingTablesConfig,
    ModelSelectionConfig,
    PipelineConfig,
    run_module_benchmark,
    run_pipeline,
    RunOptions,
    UnifiedBenchmarkConfig,
)
from torchrec.distributed.benchmark.benchmark_utils import cmd_conf

logger = logging.getLogger(__name__)


def _create_base_model_config(model_selection_config: ModelSelectionConfig) -> BaseModelConfig:
    """Convert ModelSelectionConfig to BaseModelConfig using create_model_config."""
    return create_model_config(**{field.name: getattr(model_selection_config, field.name) for field in fields(model_selection_config)})


@cmd_conf
def main(unified_config: UnifiedBenchmarkConfig) -> None:
    """
    Unified main function that routes to appropriate benchmark type.

    Args:
        unified_config: Unified configuration specifying benchmark type, model details, run options, and table config
    """
    # Ensure nested configs exist
    if unified_config.run_options is None:
        raise ValueError("run_options must be specified in unified_config")
    if unified_config.embedding_tables_config is None:
        raise ValueError("embedding_tables_config must be specified in unified_config")
    
    logger.info(f"Starting {unified_config.benchmark_type.value} benchmark...")

    if unified_config.benchmark_type == BenchmarkType.MODULE:
        result = run_module_benchmark(
            unified_config, 
            unified_config.embedding_tables_config, 
            unified_config.run_options
        )
        logger.info(f"Module benchmark completed: {result}")

    elif unified_config.benchmark_type == BenchmarkType.PIPELINE:
        # Convert ModelSelectionConfig to BaseModelConfig
        if unified_config.model_config is None:
            raise ValueError("model_config must be specified for pipeline benchmarking")
        
        base_model_config = _create_base_model_config(unified_config.model_config)
        
        result = run_pipeline(
            unified_config.run_options, 
            unified_config.embedding_tables_config, 
            unified_config.pipeline_config, 
            base_model_config
        )
        logger.info(f"Pipeline benchmark completed: {result}")

    else:
        raise ValueError(
            f"Unknown benchmark_type: {unified_config.benchmark_type}. Must be BenchmarkType.MODULE or BenchmarkType.PIPELINE"
        )


if __name__ == "__main__":
    main()