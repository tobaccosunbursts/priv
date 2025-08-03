# Migration Guide: Unified Benchmark System

This guide explains how to migrate from the existing benchmark scripts to the new unified YAML-based benchmark system.

## Overview

The unified benchmark system provides:

1. **Single Entry Point**: One runner for all benchmark types
2. **YAML Configuration**: All parameters configurable via YAML files
3. **Dynamic Module Loading**: Benchmark any PyTorch module without code changes
4. **Backward Compatibility**: Existing functionality preserved

## Migration Scenarios

### 1. From Direct `benchmark_module()` Calls

**Before:**

```python
from torchrec.distributed.benchmark.benchmark_utils import benchmark_module
from torchrec.modules.embedding_modules import EmbeddingBagCollection

# Create tables and module manually
tables = [...]
module = EmbeddingBagCollection(tables=tables)

# Run benchmark
result = benchmark_module(
    module=module,
    tables=tables,
    world_size=2,
    num_benchmarks=10,
    sharding_type=ShardingType.TABLE_WISE,
)
```

**After:**

```yaml
# config.yaml
benchmark_type: "module"

run_options:
  world_size: 2
  num_batches: 10
  sharding_type: "TABLE_WISE"

table_config:
  num_unweighted_features: 100
  embedding_feature_dim: 128

module_config:
  module_path: "torchrec.modules.embedding_modules"
  class_name: "EmbeddingBagCollection"
  requires_tables: true
  constructor_args:
    device: "meta"
```

```bash
python -m torchrec.distributed.benchmark.unified_benchmark_runner --config config.yaml
```

### 2. From `benchmark_train_pipeline.py`

**Before:**

```bash
python -m torchrec.distributed.benchmark.benchmark_train_pipeline \
  --world_size=2 \
  --pipeline=sparse \
  --batch_size=8192 \
  --num_batches=10
```

**After:**

```yaml
# pipeline_config.yaml
benchmark_type: "pipeline"

run_options:
  world_size: 2
  num_batches: 10

table_config:
  num_unweighted_features: 100
  embedding_feature_dim: 128

pipeline_config:
  pipeline: "sparse"

model_selection:
  model_name: "test_sparse_nn"
  batch_size: 8192
  num_float_features: 10
```

```bash
python -m torchrec.distributed.benchmark.unified_benchmark_runner --config pipeline_config.yaml
```

### 3. From Custom Model Classes

**Before:**

```python
# Custom model benchmarking required writing new code
class MyCustomModel(nn.Module):
    def __init__(self, tables, dense_features):
        # ... implementation

# Manually integrate with benchmark_module
```

**After:**

```yaml
# custom_model_config.yaml
benchmark_type: "module"

run_options:
  world_size: 2
  num_batches: 10

table_config:
  num_unweighted_features: 26
  embedding_feature_dim: 128

module_config:
  module_path: "your.package.models"
  class_name: "MyCustomModel"
  requires_tables: true
  requires_dense_device: true
  constructor_args:
    dense_features: 13
    hidden_size: 512
```

## Parameter Mapping

### Run Options Mapping

| Old Parameter    | New YAML Path                | Notes               |
| ---------------- | ---------------------------- | ------------------- |
| `world_size`     | `run_options.world_size`     | Same                |
| `num_benchmarks` | `run_options.num_batches`    | Renamed for clarity |
| `sharding_type`  | `run_options.sharding_type`  | Same                |
| `compute_kernel` | `run_options.compute_kernel` | Same                |
| `planner_type`   | `run_options.planner_type`   | New parameter       |

### Table Configuration Mapping

| Old Parameter             | New YAML Path                          | Notes                    |
| ------------------------- | -------------------------------------- | ------------------------ |
| `num_unweighted_features` | `table_config.num_unweighted_features` | From `generate_tables()` |
| `num_weighted_features`   | `table_config.num_weighted_features`   | From `generate_tables()` |
| `embedding_feature_dim`   | `table_config.embedding_feature_dim`   | From `generate_tables()` |

### Pipeline Configuration Mapping

| Old Parameter         | New YAML Path                       | Notes                |
| --------------------- | ----------------------------------- | -------------------- |
| `--pipeline`          | `pipeline_config.pipeline`          | Same                 |
| `--emb_lookup_stream` | `pipeline_config.emb_lookup_stream` | Default: "data_dist" |
| `--apply_jit`         | `pipeline_config.apply_jit`         | Default: false       |

## Advanced Migration Examples

### Complex DeepFM Configuration

**Before:**

```python
from torchrec.models.deepfm import SimpleDeepFMNNWrapper
from torchrec.modules.embedding_modules import EmbeddingBagCollection

tables = generate_tables(26, 0, 128)[0]  # DLRM tables
ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))

model = SimpleDeepFMNNWrapper(
    num_dense_features=13,
    embedding_bag_collection=ebc,
    hidden_layer_size=512,
    deep_fm_dimension=16,
)

result = benchmark_module(model=model, tables=tables, ...)
```

**After:**

```yaml
benchmark_type: "module"

run_options:
  world_size: 2
  num_batches: 10

table_config:
  num_unweighted_features: 26
  num_weighted_features: 0
  embedding_feature_dim: 128

module_config:
  module_path: "torchrec.models.deepfm"
  class_name: "SimpleDeepFMNNWrapper"
  requires_tables: true
  constructor_args:
    num_dense_features: 13
    hidden_layer_size: 512
    deep_fm_dimension: 16
```

### Multi-Configuration Benchmarking

**Before:** Required multiple script runs with different parameters

**After:** Create multiple YAML files and run them sequentially:

```bash
# Benchmark different sharding strategies
python -m torchrec.distributed.benchmark.unified_benchmark_runner --config table_wise_config.yaml
python -m torchrec.distributed.benchmark.unified_benchmark_runner --config row_wise_config.yaml
python -m torchrec.distributed.benchmark.unified_benchmark_runner --config column_wise_config.yaml
```

## Benefits of Migration

1. **Reproducibility**: YAML configs can be version controlled and shared
2. **Flexibility**: Easy to modify parameters without code changes
3. **Documentation**: Self-documenting configuration format
4. **Automation**: Easier to script and automate benchmark runs
5. **Consistency**: Unified interface for all benchmark types

## Backward Compatibility

The unified system maintains full backward compatibility:

- All existing `benchmark_utils.py` functions still work
- `benchmark_train_pipeline.py` can still be run directly
- Existing code using the old APIs continues to function

## Migration Checklist

- [ ] Identify your current benchmark usage pattern
- [ ] Create equivalent YAML configuration files
- [ ] Test configurations with `--validate-only` flag
- [ ] Run benchmarks and compare results
- [ ] Update any automation scripts to use new runner
- [ ] Consider consolidating multiple configurations into shared YAML files

## Getting Help

For questions or issues during migration:

1. Check the [examples directory](examples/) for reference configurations
2. Use the `--validate-only` flag to check configuration syntax
3. Use `--loglevel debug` for detailed error information
4. Review the existing test cases in `test_unified_benchmark.py`
