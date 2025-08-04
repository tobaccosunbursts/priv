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

Example usage:

OSS (external):
    python -m torchrec.distributed.benchmark.unified_benchmark_runner --yaml_config=config.yaml

Example YAML for module benchmarking:
```yaml
UnifiedBenchmarkConfig:
  benchmark_type: "module"
  module_path: "torchrec.models.deepfm"
  module_class: "SimpleDeepFMNNWrapper"
  module_kwargs:
    num_dense_features: 10
    hidden_layer_size: 20
    deep_fm_dimension: 5

RunOptions:
  world_size: 2
  num_batches: 10
  sharding_type: "TABLE_WISE"

EmbeddingTablesConfig:
  num_unweighted_features: 100
  num_weighted_features: 100
  embedding_feature_dim: 128
```

Example YAML for pipeline benchmarking:
```yaml
UnifiedBenchmarkConfig:
  benchmark_type: "pipeline"

PipelineConfig:
  pipeline: "sparse"
  apply_jit: false

RunOptions:
  world_size: 2
  num_batches: 10

ModelSelectionConfig:
  model_name: "test_sparse_nn"
  batch_size: 8192

EmbeddingTablesConfig:
  num_unweighted_features: 100
  embedding_feature_dim: 128
```
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Union
import importlib

import torch
from torch import nn
from torchrec.distributed.benchmark.benchmark_train_pipeline import (
    EmbeddingTablesConfig,
    ModelSelectionConfig,
    PipelineConfig,
    RunOptions,
    UnifiedBenchmarkConfig,
    run_module_benchmark,
    run_pipeline,
)
from torchrec.distributed.benchmark.benchmark_pipeline_utils import (
    BaseModelConfig,
    create_model_config,
)
from torchrec.distributed.benchmark.benchmark_utils import (
    cmd_conf,
)


@cmd_conf
def main(
    run_option: RunOptions,
    table_config: EmbeddingTablesConfig,
    unified_config: UnifiedBenchmarkConfig,
    model_selection: Optional[ModelSelectionConfig] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    model_config: Optional[BaseModelConfig] = None,
) -> None:
    """
    Unified main function that routes to appropriate benchmark type.
    
    Args:
        run_option: Configuration for running the benchmark
        table_config: Configuration for embedding tables
        unified_config: Unified configuration specifying benchmark type and module details
        model_selection: Configuration for model selection (only needed for pipeline benchmarks)
        pipeline_config: Configuration for pipeline (only needed for pipeline benchmarks)
        model_config: Optional model configuration
    """
    print(f"Starting {unified_config.benchmark_type} benchmark...")
    
    if unified_config.benchmark_type == "module":
        # Run module-level benchmark
        if not unified_config.module_path or not unified_config.module_class:
            raise ValueError("For module benchmarking, both module_path and module_class must be specified")
        
        result = run_module_benchmark(unified_config, table_config, run_option)
        print(f"Module benchmark completed: {result}")
        
    elif unified_config.benchmark_type == "pipeline":
        # Run pipeline-level benchmark
        if model_selection is None:
            # Create default model selection config
            model_selection = ModelSelectionConfig()
        
        if pipeline_config is None:
            # Create default pipeline config
            pipeline_config = PipelineConfig()
        
        if model_config is None:
            model_config = create_model_config(
                model_name=model_selection.model_name,
                batch_size=model_selection.batch_size,
                batch_sizes=model_selection.batch_sizes,
                num_float_features=model_selection.num_float_features,
                feature_pooling_avg=model_selection.feature_pooling_avg,
                use_offsets=model_selection.use_offsets,
                dev_str=model_selection.dev_str,
                long_kjt_indices=model_selection.long_kjt_indices,
                long_kjt_offsets=model_selection.long_kjt_offsets,
                long_kjt_lengths=model_selection.long_kjt_lengths,
                pin_memory=model_selection.pin_memory,
                embedding_groups=model_selection.embedding_groups,
                feature_processor_modules=model_selection.feature_processor_modules,
                max_feature_lengths=model_selection.max_feature_lengths,
                over_arch_clazz=model_selection.over_arch_clazz,
                postproc_module=model_selection.postproc_module,
                zch=model_selection.zch,
                hidden_layer_size=model_selection.hidden_layer_size,
                deep_fm_dimension=model_selection.deep_fm_dimension,
                dense_arch_layer_sizes=model_selection.dense_arch_layer_sizes,
                over_arch_layer_sizes=model_selection.over_arch_layer_sizes,
            )
        
        results = run_pipeline(run_option, table_config, pipeline_config, model_config)
        print(f"Pipeline benchmark completed with {len(results)} results")
        
    else:
        raise ValueError(f"Unknown benchmark_type: {unified_config.benchmark_type}. Must be 'module' or 'pipeline'")


if __name__ == "__main__":
    main()