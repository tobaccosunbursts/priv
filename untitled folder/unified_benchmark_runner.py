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
from typing import Optional

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


@cmd_conf
def main(
    run_option: RunOptions,
    table_config: EmbeddingTablesConfig,
    unified_config: UnifiedBenchmarkConfig,
    model_config: Optional[BaseModelConfig] = None,
) -> None:
    """
    Unified main function that routes to appropriate benchmark type.

    Args:
        run_option: Configuration for running the benchmark
        table_config: Configuration for embedding tables
        unified_config: Unified configuration specifying benchmark type and model details
        model_config: Optional base model configuration (overrides unified_config.model_config)
    """

    logger.info(f"Starting {unified_config.benchmark_type.value} benchmark...")

    if unified_config.benchmark_type == BenchmarkType.MODULE:
        result = run_module_benchmark(unified_config, table_config, run_option)
        logger.info(f"Module benchmark completed: {result}")

    elif unified_config.benchmark_type == BenchmarkType.PIPELINE:
        # Run pipeline-level benchmark
        if model_config is None and unified_config.model_config is not None:
            model_config = create_model_config(
                model_name=unified_config.model_config.model_name,
                batch_size=unified_config.model_config.batch_size,
                batch_sizes=unified_config.model_config.batch_sizes,
                num_float_features=unified_config.model_config.num_float_features,
                feature_pooling_avg=unified_config.model_config.feature_pooling_avg,
                use_offsets=unified_config.model_config.use_offsets,
                dev_str=unified_config.model_config.dev_str,
                long_kjt_indices=unified_config.model_config.long_kjt_indices,
                long_kjt_offsets=unified_config.model_config.long_kjt_offsets,
                long_kjt_lengths=unified_config.model_config.long_kjt_lengths,
                pin_memory=unified_config.model_config.pin_memory,
                embedding_groups=unified_config.model_config.embedding_groups,
                feature_processor_modules=unified_config.model_config.feature_processor_modules,
                max_feature_lengths=unified_config.model_config.max_feature_lengths,
                over_arch_clazz=unified_config.model_config.over_arch_clazz,
                postproc_module=unified_config.model_config.postproc_module,
                zch=unified_config.model_config.zch,
                hidden_layer_size=unified_config.model_config.hidden_layer_size,
                deep_fm_dimension=unified_config.model_config.deep_fm_dimension,
                dense_arch_layer_sizes=unified_config.model_config.dense_arch_layer_sizes,
                over_arch_layer_sizes=unified_config.model_config.over_arch_layer_sizes,
            )

        result = run_pipeline(run_option, table_config, unified_config.pipeline_config, model_config)
        logger.info(f"Pipeline benchmark completed: {result}")

    else:
        raise ValueError(
            f"Unknown benchmark_type: {unified_config.benchmark_type}. Must be BenchmarkType.MODULE or BenchmarkType.PIPELINE"
        )


if __name__ == "__main__":
    main()