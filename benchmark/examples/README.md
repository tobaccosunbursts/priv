# Unified Benchmark Configuration Examples

This directory contains example YAML configurations for the unified benchmark system that can handle both module-level and pipeline-level benchmarking.

## Overview

The unified benchmark system provides a single interface for running different types of benchmarks:

1. **Module Benchmarking**: Benchmark any PyTorch module with dynamic importing
2. **Pipeline Benchmarking**: Benchmark complete training pipelines with predefined models

## Configuration Structure

### Module Benchmarking

For module benchmarking, set `benchmark_type: "module"` and provide:

- `run_options`: Common benchmark settings (world_size, sharding_type, etc.)
- `table_config`: Embedding table configuration
- `module_config`: Dynamic module importing configuration

### Pipeline Benchmarking

For pipeline benchmarking, set `benchmark_type: "pipeline"` and provide:

- `run_options`: Common benchmark settings
- `table_config`: Embedding table configuration
- `pipeline_config`: Training pipeline configuration
- `model_selection`: Model selection and parameters

## Example Files

### Module Benchmarking Examples

- **`deepfm_module_config.yaml`**: Benchmark DeepFM model directly
- **`dlrm_module_config.yaml`**: Benchmark DLRM model directly
- **`custom_module_config.yaml`**: Benchmark any custom module (e.g., EmbeddingBagCollection)

### Pipeline Benchmarking Examples

- **`sparse_pipeline_config.yaml`**: Benchmark sparse training pipeline with TestSparseNN

## Usage

Run any configuration with the unified runner:

```bash
# Module benchmarking
python -m torchrec.distributed.benchmark.unified_benchmark_runner --config examples/deepfm_module_config.yaml

# Pipeline benchmarking
python -m torchrec.distributed.benchmark.unified_benchmark_runner --config examples/sparse_pipeline_config.yaml

# Validate configuration without running
python -m torchrec.distributed.benchmark.unified_benchmark_runner --config examples/dlrm_module_config.yaml --validate-only

# With debug logging
python -m torchrec.distributed.benchmark.unified_benchmark_runner --config examples/custom_module_config.yaml --loglevel debug
```

## Dynamic Module Import

The `module_config` section allows you to import and benchmark any PyTorch module:

```yaml
module_config:
  module_path: "your.module.path" # Python import path
  class_name: "YourModuleClass" # Class name to instantiate
  requires_tables: true # Whether module needs embedding tables
  requires_weighted_tables: false # Whether module needs weighted tables
  requires_dense_device: true # Whether module needs dense_device parameter
  constructor_args: # Arguments passed to constructor
    param1: value1
    param2: value2
```

## Configuration Parameters

### Common Run Options

- `world_size`: Number of processes/GPUs for distributed training
- `num_batches`: Number of batches to benchmark
- `sharding_type`: Strategy for sharding ("TABLE_WISE", "ROW_WISE", "COLUMN_WISE")
- `compute_kernel`: Compute kernel to use ("FUSED", "DENSE")
- `planner_type`: Sharding planner ("embedding", "hetero")
- `dense_optimizer`: Optimizer for dense parameters ("SGD", "Adam", etc.)
- `sparse_optimizer`: Optimizer for sparse parameters ("EXACT_ADAGRAD", etc.)

### Table Configuration

- `num_unweighted_features`: Number of unweighted embedding features
- `num_weighted_features`: Number of weighted embedding features
- `embedding_feature_dim`: Dimension of embedding vectors

### Pipeline Configuration (pipeline benchmarking only)

- `pipeline`: Type of training pipeline ("base", "sparse", "fused", "semi", "prefetch")
- `emb_lookup_stream`: Stream for embedding lookups
- `apply_jit`: Whether to apply JIT compilation

## Adding New Models

To benchmark a new model with module benchmarking:

1. Create a YAML config with the module path and class name
2. Specify constructor arguments and requirements
3. Run with the unified runner

To add a new model to pipeline benchmarking:

1. Add the model configuration to `benchmark_pipeline_utils.py`
2. Update the model registry in `create_model_config()`
3. Use in YAML with `model_selection.model_name`

## Benefits of Unified System

1. **Single Interface**: One runner for all benchmark types
2. **Configuration as Code**: YAML configs can be version controlled
3. **Dynamic Loading**: Import any module without code changes
4. **Flexible Parameters**: Easy to modify benchmark parameters
5. **Validation**: Built-in configuration validation
6. **Backward Compatibility**: Existing pipeline benchmarks still work
