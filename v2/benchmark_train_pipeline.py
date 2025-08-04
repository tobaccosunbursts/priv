#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_train_pipeline -- --world_size=2 --pipeline=sparse --batch_size=10

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_train_pipeline --world_size=4 --pipeline=sparse --batch_size=10

Adding New Model Support:
    See benchmark_pipeline_utils.py for step-by-step instructions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union
import importlib

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import nn
from torchrec.distributed.benchmark.benchmark_pipeline_utils import (
    BaseModelConfig,
    create_model_config,
    DeepFMConfig,
    DLRMConfig,
    generate_data,
    generate_pipeline,
    TestSparseNNConfig,
    TestTowerCollectionSparseNNConfig,
    TestTowerSparseNNConfig,
)
from torchrec.distributed.benchmark.benchmark_utils import (
    benchmark_func,
    benchmark_module,
    BenchmarkResult,
    cmd_conf,
    generate_planner,
    generate_sharded_model_and_optimizer,
    generate_tables,
    ModuleBenchmarkConfig,
    multi_process_benchmark,
)
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import Topology

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)
from torchrec.distributed.test_utils.test_input import ModelInput
from torchrec.distributed.test_utils.test_model import TestOverArchLarge
from torchrec.distributed.train_pipeline import TrainPipeline
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig


@dataclass
class UnifiedBenchmarkConfig:
    """Unified configuration for pipeline, module, and function benchmarking."""
    benchmark_type: str = "pipeline"  # "pipeline", "module", or "function"
    
    # Module benchmarking specific options
    module_path: str = ""  # e.g., "torchrec.models.deepfm"
    module_class: str = ""  # e.g., "SimpleDeepFMNNWrapper"
    module_kwargs: Optional[Dict[str, Any]] = None  # Additional kwargs for module instantiation
    
    # Function benchmarking specific options
    function_path: str = ""  # e.g., "torchrec.distributed.model_tracker.model_delta_tracker"
    function_name: str = ""  # e.g., "ModelDeltaTracker"
    function_kwargs: Optional[Dict[str, Any]] = None  # Additional kwargs for function setup

    def __post_init__(self):
        if self.module_kwargs is None:
            self.module_kwargs = {}
        if self.function_kwargs is None:
            self.function_kwargs = {}


@dataclass
class RunOptions:
    """
    Configuration options for running sparse neural network benchmarks.

    This class defines the parameters that control how the benchmark is executed,
    including distributed training settings, batch configuration, and profiling options.

    Args:
        world_size (int): Number of processes/GPUs to use for distributed training.
            Default is 2.
        num_batches (int): Number of batches to process during the benchmark.
            Default is 10.
        sharding_type (ShardingType): Strategy for sharding embedding tables across devices.
            Default is ShardingType.TABLE_WISE (entire tables are placed on single devices).
        compute_kernel (EmbeddingComputeKernel): Compute kernel to use for embedding tables.
            Default is EmbeddingComputeKernel.FUSED.
        input_type (str): Type of input format to use for the model.
            Default is "kjt" (KeyedJaggedTensor).
        profile (str): Directory to save profiling results. If empty, profiling is disabled.
            Default is "" (disabled).
        planner_type (str): Type of sharding planner to use. Options are:
            - "embedding": EmbeddingShardingPlanner (default)
            - "hetero": HeteroEmbeddingShardingPlanner
        pooling_factors (Optional[List[float]]): Pooling factors for each feature of the table.
            This is the average number of values each sample has for the feature.
        num_poolings (Optional[List[float]]): Number of poolings for each feature of the table.
        dense_optimizer (str): Optimizer to use for dense parameters.
            Default is "SGD".
        dense_lr (float): Learning rate for dense parameters.
            Default is 0.1.
        sparse_optimizer (str): Optimizer to use for sparse parameters.
            Default is "EXACT_ADAGRAD".
        sparse_lr (float): Learning rate for sparse parameters.
            Default is 0.1.
    """

    world_size: int = 2
    num_batches: int = 10
    sharding_type: ShardingType = ShardingType.TABLE_WISE
    compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.FUSED
    input_type: str = "kjt"
    profile: str = ""
    planner_type: str = "embedding"
    pooling_factors: Optional[List[float]] = None
    num_poolings: Optional[List[float]] = None
    dense_optimizer: str = "SGD"
    dense_lr: float = 0.1
    dense_momentum: Optional[float] = None
    dense_weight_decay: Optional[float] = None
    sparse_optimizer: str = "EXACT_ADAGRAD"
    sparse_lr: float = 0.1
    sparse_momentum: Optional[float] = None
    sparse_weight_decay: Optional[float] = None
    export_stacks: bool = False


@dataclass
class EmbeddingTablesConfig:
    """
    Configuration for embedding tables.

    This class defines the parameters for generating embedding tables with both weighted
    and unweighted features.

    Args:
        num_unweighted_features (int): Number of unweighted features to generate.
            Default is 100.
        num_weighted_features (int): Number of weighted features to generate.
            Default is 100.
        embedding_feature_dim (int): Dimension of the embedding vectors.
            Default is 128.
    """

    num_unweighted_features: int = 100
    num_weighted_features: int = 100
    embedding_feature_dim: int = 128


@dataclass
class PipelineConfig:
    """
    Configuration for training pipelines.

    This class defines the parameters for configuring the training pipeline.

    Args:
        pipeline (str): The type of training pipeline to use. Options include:
            - "base": Basic training pipeline
            - "sparse": Pipeline optimized for sparse operations
            - "fused": Pipeline with fused sparse distribution
            - "semi": Semi-synchronous training pipeline
            - "prefetch": Pipeline with prefetching for sparse distribution
            Default is "base".
        emb_lookup_stream (str): The stream to use for embedding lookups.
            Only used by certain pipeline types (e.g., "fused").
            Default is "data_dist".
        apply_jit (bool): Whether to apply JIT (Just-In-Time) compilation to the model.
            Default is False.
    """

    pipeline: str = "base"
    emb_lookup_stream: str = "data_dist"
    apply_jit: bool = False


@dataclass
class ModelSelectionConfig:
    model_name: str = "test_sparse_nn"

    # Common config for all model types
    batch_size: int = 8192
    batch_sizes: Optional[List[int]] = None
    num_float_features: int = 10
    feature_pooling_avg: int = 10
    use_offsets: bool = False
    dev_str: str = ""
    long_kjt_indices: bool = True
    long_kjt_offsets: bool = True
    long_kjt_lengths: bool = True
    pin_memory: bool = True

    # TestSparseNN specific config
    embedding_groups: Optional[Dict[str, List[str]]] = None
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None
    max_feature_lengths: Optional[Dict[str, int]] = None
    over_arch_clazz: Type[nn.Module] = TestOverArchLarge
    postproc_module: Optional[nn.Module] = None
    zch: bool = False

    # DeepFM specific config
    hidden_layer_size: int = 20
    deep_fm_dimension: int = 5

    # DLRM specific config
    dense_arch_layer_sizes: List[int] = field(default_factory=lambda: [20, 128])
    over_arch_layer_sizes: List[int] = field(default_factory=lambda: [5, 1])


def dynamic_import_module(module_path: str, module_class: str) -> Type[nn.Module]:
    """Dynamically import a module class from a given path."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, module_class)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to import {module_class} from {module_path}: {e}")


def create_module_instance(
    unified_config: UnifiedBenchmarkConfig,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    table_config: EmbeddingTablesConfig,
) -> nn.Module:
    """Create a module instance based on the unified config."""
    ModuleClass = dynamic_import_module(unified_config.module_path, unified_config.module_class)
    
    # Handle common module instantiation patterns
    if unified_config.module_class == "SimpleDeepFMNNWrapper":
        from torchrec.modules.embedding_modules import EmbeddingBagCollection
        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
        return ModuleClass(
            embedding_bag_collection=ebc,
            num_dense_features=10,  # Default value, can be overridden via module_kwargs
            **unified_config.module_kwargs
        )
    elif unified_config.module_class == "DLRMWrapper":
        from torchrec.modules.embedding_modules import EmbeddingBagCollection
        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
        return ModuleClass(
            embedding_bag_collection=ebc,
            dense_in_features=10,  # Default value, can be overridden via module_kwargs
            dense_arch_layer_sizes=[20, 128],  # Default value
            over_arch_layer_sizes=[5, 1],  # Default value
            **unified_config.module_kwargs
        )
    elif unified_config.module_class == "EmbeddingBagCollection":
        return ModuleClass(tables=tables, **unified_config.module_kwargs)
    else:
        # Generic instantiation - try with tables and weighted_tables
        try:
            return ModuleClass(
                tables=tables,
                weighted_tables=weighted_tables,
                **unified_config.module_kwargs
            )
        except TypeError:
            # Fallback to just tables
            try:
                return ModuleClass(tables=tables, **unified_config.module_kwargs)
            except TypeError:
                # Fallback to no embedding tables
                return ModuleClass(**unified_config.module_kwargs)


def run_module_benchmark(
    unified_config: UnifiedBenchmarkConfig,
    table_config: EmbeddingTablesConfig,
    run_option: RunOptions,
) -> BenchmarkResult:
    """Run module-level benchmarking."""
    tables, weighted_tables = generate_tables(
        num_unweighted_features=table_config.num_unweighted_features,
        num_weighted_features=table_config.num_weighted_features,
        embedding_feature_dim=table_config.embedding_feature_dim,
    )
    
    module = create_module_instance(unified_config, tables, weighted_tables, table_config)
    
    return benchmark_module(
        module=module,
        tables=tables,
        weighted_tables=weighted_tables,
        num_float_features=10,  # Default value
        sharding_type=run_option.sharding_type,
        planner_type=run_option.planner_type,
        world_size=run_option.world_size,
        num_benchmarks=5,  # Default value
        batch_size=2048,  # Default value
        compute_kernel=run_option.compute_kernel,
        device_type="cuda",
    )


def run_function_benchmark(
    unified_config: UnifiedBenchmarkConfig,
    table_config: EmbeddingTablesConfig,
    run_option: RunOptions,
) -> BenchmarkResult:
    """Run function-level benchmarking."""
    import importlib
    from torchrec.distributed.test_utils.multi_process import MultiProcessContext
    from torchrec.distributed.benchmark.benchmark_utils import benchmark_func, multi_process_benchmark
    
    module = importlib.import_module(unified_config.function_path)
    FunctionClass = getattr(module, unified_config.function_name)
    
    def _run_benchmark(rank: int, world_size: int, **kwargs) -> BenchmarkResult:
        from torchrec.distributed.test_utils.test_input import ModelInput
        with MultiProcessContext(rank=rank, world_size=world_size, backend="nccl") as ctx:
            tables, weighted_tables = generate_tables(table_config.num_unweighted_features, table_config.num_weighted_features, table_config.embedding_feature_dim)
            test_inputs = [ModelInput.generate(batch_size=2048, tables=tables, weighted_tables=weighted_tables, num_float_features=0, device=ctx.device) for _ in range(5)]
            
            if unified_config.function_name == "ModelDeltaTracker":
                from torchrec.modules.embedding_modules import EmbeddingBagCollection
                from torchrec.distributed.model_tracker.types import TrackingMode
                
                model = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
                sharded_model, _ = generate_sharded_model_and_optimizer(
                    model, run_option.sharding_type.value, run_option.compute_kernel.value,
                    ctx.pg, ctx.device, {"optimizer": "SGD", "learning_rate": 0.1}
                )
                tracker = FunctionClass(model=sharded_model, tracking_mode=TrackingMode.ID_ONLY)
                
                return benchmark_func(
                    name="ModelDeltaTracker", bench_inputs=test_inputs, prof_inputs=test_inputs[:2],
                    num_benchmarks=5, num_profiles=2, profile_dir="", world_size=world_size,
                    func_to_benchmark=lambda inputs, model, tracker: [model(inp.idlist_features) or tracker.step() or tracker.get_delta_ids() for inp in inputs],
                    benchmark_func_kwargs={"model": sharded_model, "tracker": tracker}, rank=rank
                )
            else:
                func_instance = FunctionClass(**(unified_config.function_kwargs or {}))
                return benchmark_func(
                    name=unified_config.function_name, bench_inputs=test_inputs, prof_inputs=test_inputs[:2],
                    num_benchmarks=5, num_profiles=2, profile_dir="", world_size=world_size,
                    func_to_benchmark=lambda inputs, instance: [getattr(instance, '__call__', getattr(instance, 'step', lambda x: None))(inp) for inp in inputs],
                    benchmark_func_kwargs={"instance": func_instance}, rank=rank
                )
    
    return multi_process_benchmark(callable=_run_benchmark, world_size=run_option.world_size)



@cmd_conf
def main(
    run_option: RunOptions,
    table_config: EmbeddingTablesConfig,
    model_selection: ModelSelectionConfig,
    pipeline_config: PipelineConfig,
    unified_config: UnifiedBenchmarkConfig,
    model_config: Optional[BaseModelConfig] = None,
) -> None:
    # Route to appropriate benchmark type based on unified config
    if unified_config.benchmark_type == "module":
        print("Running module-level benchmark...")
        result = run_module_benchmark(unified_config, table_config, run_option)
        print(f"Module benchmark completed: {result}")
    elif unified_config.benchmark_type == "function":
        print("Running function-level benchmark...")
        result = run_function_benchmark(unified_config, table_config, run_option)
        print(f"Function benchmark completed: {result}")
    elif unified_config.benchmark_type == "pipeline":
        print("Running pipeline-level benchmark...")
        tables, weighted_tables = generate_tables(
            num_unweighted_features=table_config.num_unweighted_features,
            num_weighted_features=table_config.num_weighted_features,
            embedding_feature_dim=table_config.embedding_feature_dim,
        )

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

        # launch trainers
        run_multi_process_func(
            func=runner,
            world_size=run_option.world_size,
            tables=tables,
            weighted_tables=weighted_tables,
            run_option=run_option,
            model_config=model_config,
            pipeline_config=pipeline_config,
        )
    else:
        raise ValueError(f"Unknown benchmark_type: {unified_config.benchmark_type}. Must be 'module', 'function', or 'pipeline'")


def run_pipeline(
    run_option: RunOptions,
    table_config: EmbeddingTablesConfig,
    pipeline_config: PipelineConfig,
    model_config: BaseModelConfig,
) -> List[BenchmarkResult]:

    tables, weighted_tables = generate_tables(
        num_unweighted_features=table_config.num_unweighted_features,
        num_weighted_features=table_config.num_weighted_features,
        embedding_feature_dim=table_config.embedding_feature_dim,
    )

    return run_multi_process_func(
        func=runner,
        world_size=run_option.world_size,
        tables=tables,
        weighted_tables=weighted_tables,
        run_option=run_option,
        model_config=model_config,
        pipeline_config=pipeline_config,
    )


def runner(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    run_option: RunOptions,
    model_config: BaseModelConfig,
    pipeline_config: PipelineConfig,
) -> BenchmarkResult:
    # Ensure GPUs are available and we have enough of them
    assert (
        torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    ), "CUDA not available or insufficient GPUs for the requested world_size"

    torch.autograd.set_detect_anomaly(True)
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl",
        use_deterministic_algorithms=False,
    ) as ctx:
        unsharded_model = model_config.generate_model(
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=ctx.device,
        )

        # Create a topology for sharding
        topology = Topology(
            local_world_size=get_local_size(world_size),
            world_size=world_size,
            compute_device=ctx.device.type,
        )

        batch_sizes = model_config.batch_sizes

        if batch_sizes is None:
            batch_sizes = [model_config.batch_size] * run_option.num_batches
        else:
            assert (
                len(batch_sizes) == run_option.num_batches
            ), "The length of batch_sizes must match the number of batches."

        # Create a planner for sharding based on the specified type
        planner = generate_planner(
            planner_type=run_option.planner_type,
            topology=topology,
            tables=tables,
            weighted_tables=weighted_tables,
            sharding_type=run_option.sharding_type,
            compute_kernel=run_option.compute_kernel,
            batch_sizes=batch_sizes,
            pooling_factors=run_option.pooling_factors,
            num_poolings=run_option.num_poolings,
        )
        bench_inputs = generate_data(
            tables=tables,
            weighted_tables=weighted_tables,
            model_config=model_config,
            batch_sizes=batch_sizes,
        )

        # Prepare fused_params for sparse optimizer
        fused_params = {
            "optimizer": getattr(EmbOptimType, run_option.sparse_optimizer.upper()),
            "learning_rate": run_option.sparse_lr,
        }

        # Add momentum and weight_decay to fused_params if provided
        if run_option.sparse_momentum is not None:
            fused_params["momentum"] = run_option.sparse_momentum

        if run_option.sparse_weight_decay is not None:
            fused_params["weight_decay"] = run_option.sparse_weight_decay

        sharded_model, optimizer = generate_sharded_model_and_optimizer(
            model=unsharded_model,
            sharding_type=run_option.sharding_type.value,
            kernel_type=run_option.compute_kernel.value,
            # pyre-ignore
            pg=ctx.pg,
            device=ctx.device,
            fused_params=fused_params,
            dense_optimizer=run_option.dense_optimizer,
            dense_lr=run_option.dense_lr,
            dense_momentum=run_option.dense_momentum,
            dense_weight_decay=run_option.dense_weight_decay,
            planner=planner,
        )

        def _func_to_benchmark(
            bench_inputs: List[ModelInput],
            model: nn.Module,
            pipeline: TrainPipeline,
        ) -> None:
            dataloader = iter(bench_inputs)
            while True:
                try:
                    pipeline.progress(dataloader)
                except StopIteration:
                    break

        pipeline = generate_pipeline(
            pipeline_type=pipeline_config.pipeline,
            emb_lookup_stream=pipeline_config.emb_lookup_stream,
            model=sharded_model,
            opt=optimizer,
            device=ctx.device,
            apply_jit=pipeline_config.apply_jit,
        )
        pipeline.progress(iter(bench_inputs))

        result = benchmark_func(
            name=type(pipeline).__name__,
            bench_inputs=bench_inputs,  # pyre-ignore
            prof_inputs=bench_inputs,  # pyre-ignore
            num_benchmarks=5,
            num_profiles=2,
            profile_dir=run_option.profile,
            world_size=run_option.world_size,
            func_to_benchmark=_func_to_benchmark,
            benchmark_func_kwargs={"model": sharded_model, "pipeline": pipeline},
            rank=rank,
            export_stacks=run_option.export_stacks,
        )

        if rank == 0:
            print(result)

        return result


if __name__ == "__main__":
    main()
