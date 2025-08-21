#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[16]

#!/usr/bin/env python3

import argparse
import contextlib

# Additional imports for the new benchmark_module function
import copy
import inspect
import json
import logging
import os
import resource
import time
import timeit
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from enum import Enum
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    get_args,
    get_origin,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
import yaml
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import multiprocessing as mp, nn, optim
from torch.autograd.profiler import record_function
from torch.optim import Optimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.constants import NUM_POOLINGS, POOLING_FACTOR
from torchrec.distributed.planner.planners import HeteroEmbeddingShardingPlanner
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.test_utils.multi_process import MultiProcessContext
from torchrec.distributed.test_utils.test_input import ModelInput
from torchrec.distributed.test_utils.test_model import TestEBCSharder
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import get_free_port

logger: logging.Logger = logging.getLogger()

# Reference: https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD
DLRM_NUM_EMBEDDINGS_PER_FEATURE = [
    4833188,
    36746,
    17245,
    7413,
    20243,
    3,
    7114,
    1441,
    62,
    29275261,
    1572176,
    345138,
    10,
    2209,
    11267,
    128,
    4,
    974,
    14,
    48937457,
    11316796,
    40094537,
    452104,
    12606,
    104,
    35,
]

EMBEDDING_DIM: int = 128


class CompileMode(Enum):
    EAGER = "eager"
    FX_SCRIPT = "fx_script"


@dataclass
class GPUMemoryStats:
    rank: int
    malloc_retries: int
    max_mem_allocated_mbs: int
    max_mem_reserved_mbs: int

    @classmethod
    def for_device(cls, rank: int) -> "GPUMemoryStats":
        stats = torch.cuda.memory_stats(rank)
        alloc_retries = stats.get("num_alloc_retries", 0)
        max_allocated = stats.get("allocated_bytes.all.peak", 0)
        max_reserved = stats.get("reserved_bytes.all.peak", 0)
        return cls(
            rank,
            alloc_retries,
            max_allocated // 1024 // 1024,
            max_reserved // 1024 // 1024,
        )

    def __str__(self) -> str:
        return f"Rank {self.rank}: retries={self.malloc_retries}, allocated={self.max_mem_allocated_mbs:7}mb, reserved={self.max_mem_reserved_mbs:7}mb"


@dataclass
class CPUMemoryStats:
    rank: int
    peak_rss_mbs: int

    @classmethod
    def for_process(cls, rank: int) -> "CPUMemoryStats":
        # Peak RSS from resource.getrusage (in KB on CentOS/Linux)
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = peak_rss_kb // 1024

        return cls(rank, peak_rss_mb)

    def __str__(self) -> str:
        return f"Rank {self.rank}: CPU Memory Peak RSS: {self.peak_rss_mbs/1000:.2f} GB"


@dataclass
class BenchmarkResult:
    "Class for holding results of benchmark runs"
    short_name: str
    gpu_elapsed_time: torch.Tensor  # milliseconds
    cpu_elapsed_time: torch.Tensor  # milliseconds
    gpu_mem_stats: List[GPUMemoryStats]  # GPU memory stats per rank
    cpu_mem_stats: List[CPUMemoryStats]  # CPU memory stats per rank
    rank: int = -1

    def __str__(self) -> str:
        gpu_runtime = (
            f"GPU Runtime (P90): {self.runtime_percentile(90, device='gpu'):.2f} ms"
        )
        cpu_runtime = (
            f"CPU Runtime (P90): {self.runtime_percentile(90, device='cpu'):.2f} ms"
        )
        cpu_mem = f"CPU Peak RSS (P90): {self.cpu_mem_percentile(90)/1000:.2f} GB"

        if len(self.gpu_mem_stats) == 0:
            return (
                f"{self.short_name: <{35}} |  {gpu_runtime} | {cpu_runtime} | {cpu_mem}"
            )
        mem_alloc = f"GPU Peak Memory alloc (P90): {self.max_mem_alloc_percentile(90)/1000:.2f} GB"
        mem_reserved = f"GPU Peak Memory reserved (P90): {self.max_mem_reserved_percentile(90)/1000:.2f} GB"
        malloc_retries = f"Malloc retries (P50/P90/P100): {self.mem_retries(50)} / {self.mem_retries(90)} / {self.mem_retries(100)}"
        return f"{self.short_name: <{35}} | {malloc_retries} | {gpu_runtime} | {cpu_runtime} | {mem_alloc} | {mem_reserved} | {cpu_mem}"

    def runtime_percentile(
        self,
        percentile: int = 50,
        interpolation: str = "nearest",
        device: str = "gpu",
    ) -> torch.Tensor:
        """Return the runtime percentile for the requested timer.

        Args:
            percentile: Percentile to compute.
            interpolation: See ``torch.quantile``.
            device: 'gpu' for CUDA event timings, 'cpu' for active CPU timings.
        """
        timings = self.gpu_elapsed_time if device == "gpu" else self.cpu_elapsed_time
        return torch.quantile(
            timings,
            percentile / 100.0,
            interpolation=interpolation,
        )

    def max_mem_alloc_percentile(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        return self._mem_percentile(
            lambda m: m.max_mem_allocated_mbs, percentile, interpolation
        )

    def max_mem_reserved_percentile(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        return self._mem_percentile(
            lambda m: m.max_mem_reserved_mbs, percentile, interpolation
        )

    def mem_retries(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        return self._mem_percentile(
            lambda m: m.malloc_retries, percentile, interpolation
        )

    def _mem_percentile(
        self,
        mem_selector: Callable[[GPUMemoryStats], int],
        percentile: int = 50,
        interpolation: str = "nearest",
    ) -> torch.Tensor:
        mem_data = torch.tensor(
            [mem_selector(mem_stat) for mem_stat in self.gpu_mem_stats],
            dtype=torch.float,
        )
        return torch.quantile(mem_data, percentile / 100.0, interpolation=interpolation)

    def cpu_mem_percentile(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        """Return the CPU memory percentile for peak RSS."""
        cpu_mem_data = torch.tensor(
            [cpu_stat.peak_rss_mbs for cpu_stat in self.cpu_mem_stats],
            dtype=torch.float,
        )
        return torch.quantile(
            cpu_mem_data, percentile / 100.0, interpolation=interpolation
        )


T = TypeVar("T", bound=torch.nn.Module)


def write_report(
    benchmark_results: List[BenchmarkResult],
    report_file: str,
    report_str: str,
    num_requests: int,
) -> None:
    for benchmark_res in benchmark_results:
        # GPU statistics
        avg_dur_s_gpu = benchmark_res.gpu_elapsed_time.mean().item() * 1e-3  # sec
        std_dur_s_gpu = benchmark_res.gpu_elapsed_time.std().item() * 1e-3  # sec

        # CPU statistics
        avg_dur_s_cpu = benchmark_res.cpu_elapsed_time.mean().item() * 1e-3  # sec
        std_dur_s_cpu = benchmark_res.cpu_elapsed_time.std().item() * 1e-3  # sec

        qps_gpu = int(num_requests / avg_dur_s_gpu)

        mem_str = ""
        for gpu_memory_stats in benchmark_res.gpu_mem_stats:
            mem_str += f"{gpu_memory_stats}\n"

        for cpu_memory_stats in benchmark_res.cpu_mem_stats:
            mem_str += f"{cpu_memory_stats}\n"

        report_str += (
            f"{benchmark_res.short_name:40} "
            f"Avg QPS(GPU):{qps_gpu:10} "
            f"GPU Avg: {int(1000*avg_dur_s_gpu):5}ms ±{(1000*std_dur_s_gpu):.2f}ms "
            f"CPU Avg: {int(1000*avg_dur_s_cpu):5}ms ±{(1000*std_dur_s_cpu):.2f}ms\n"
        )
        report_str += f"\tMemory Allocated Per Rank:\n\t{mem_str}\n"

    with open(report_file, "w") as f:
        f.write(report_str)

    logger.info(f"Report written to {report_file}:\n{report_str}")


def multi_process_benchmark(
    callable: Callable[
        ...,
        None,
    ],
    # pyre-ignore
    **kwargs,
) -> BenchmarkResult:

    def setUp() -> None:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = str("localhost")
            os.environ["MASTER_PORT"] = str(get_free_port())

    assert "world_size" in kwargs
    world_size = kwargs["world_size"]

    setUp()
    benchmark_res_per_rank = []
    # kineto has a known problem with fork-server: it'll hang
    # when dumping the trace. Workaround with spawn
    ctx = mp.get_context("spawn")
    qq = ctx.SimpleQueue()
    processes = []

    for rank in range(world_size):
        kwargs["rank"] = rank
        kwargs["world_size"] = world_size
        kwargs["queue"] = qq
        p = ctx.Process(
            target=callable,
            kwargs=kwargs,
        )
        p.start()
        processes.append(p)

    for _ in range(world_size):
        res = qq.get()

        benchmark_res_per_rank.append(res)
        assert len(res.gpu_mem_stats) == 1
        assert len(res.cpu_mem_stats) == 1

    for p in processes:
        p.join()
        assert 0 == p.exitcode

    total_benchmark_res = BenchmarkResult(
        short_name=benchmark_res_per_rank[0].short_name,
        gpu_elapsed_time=benchmark_res_per_rank[0].gpu_elapsed_time,
        cpu_elapsed_time=benchmark_res_per_rank[0].cpu_elapsed_time,
        gpu_mem_stats=[GPUMemoryStats(rank, 0, 0, 0) for rank in range(world_size)],
        cpu_mem_stats=[CPUMemoryStats(rank, 0) for rank in range(world_size)],
        rank=0,
    )

    for res in benchmark_res_per_rank:
        # Each rank's BenchmarkResult contains 1 GPU and 1 CPU memory measurement
        total_benchmark_res.gpu_mem_stats[res.rank] = res.gpu_mem_stats[0]
        total_benchmark_res.cpu_mem_stats[res.rank] = res.cpu_mem_stats[0]

    return total_benchmark_res


def set_embedding_config(
    embedding_config_json: str,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    the config file should follow this pattern: {feature: {num_embeddings: int, embedding_dim: int}}
    """
    embedding_configs = []
    pooling_configs = []
    has_pooling_config = False
    try:
        if os.path.exists(embedding_config_json):
            with open(embedding_config_json, "r") as f:
                embedding_config_json = json.load(f)

            for _, config in embedding_config_json.items():
                embedding_configs.append(
                    (config["num_embeddings"], config["embedding_dim"])
                )
                if "pooling_factor" in config:
                    pooling_configs.append(config["pooling_factor"])
                    has_pooling_config = True
                else:
                    if has_pooling_config:
                        raise RuntimeError(
                            "We cannot handle some features have pooling factor and others don't."
                        )
        else:
            raise RuntimeError(
                f"Could not find embedding config json at path {embedding_config_json}"
            )
    except BaseException as e:
        logger.warning(
            f"Failed to load embedding config because {e}, fallback to DLRM config"
        )
        embedding_configs = [
            (num_embeddings, EMBEDDING_DIM)
            for num_embeddings in DLRM_NUM_EMBEDDINGS_PER_FEATURE
        ]

    return embedding_configs, pooling_configs


def generate_tables(
    num_unweighted_features: int = 100,
    num_weighted_features: int = 100,
    embedding_feature_dim: int = 128,
) -> Tuple[
    List[EmbeddingBagConfig],
    List[EmbeddingBagConfig],
]:
    """
    Generate embedding bag configurations for both unweighted and weighted features.

    This function creates two lists of EmbeddingBagConfig objects:
    1. Unweighted tables: Named as "table_{i}" with feature names "feature_{i}"
    2. Weighted tables: Named as "weighted_table_{i}" with feature names "weighted_feature_{i}"

    For both types, the number of embeddings scales with the feature index,
    calculated as max(i + 1, 100) * 1000.

    Args:
        num_unweighted_features (int): Number of unweighted features to generate.
        num_weighted_features (int): Number of weighted features to generate.
        embedding_feature_dim (int): Dimension of the embedding vectors.

    Returns:
        Tuple[List[EmbeddingBagConfig], List[EmbeddingBagConfig]]: A tuple containing
        two lists - the first for unweighted embedding tables and the second for
        weighted embedding tables.
    """
    tables = [
        EmbeddingBagConfig(
            num_embeddings=max(i + 1, 100) * 1000,
            embedding_dim=embedding_feature_dim,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
        )
        for i in range(num_unweighted_features)
    ]
    weighted_tables = [
        EmbeddingBagConfig(
            num_embeddings=max(i + 1, 100) * 1000,
            embedding_dim=embedding_feature_dim,
            name="weighted_table_" + str(i),
            feature_names=["weighted_feature_" + str(i)],
        )
        for i in range(num_weighted_features)
    ]
    return tables, weighted_tables


def generate_planner(
    planner_type: str,
    topology: Topology,
    tables: Optional[List[EmbeddingBagConfig]],
    weighted_tables: Optional[List[EmbeddingBagConfig]],
    sharding_type: ShardingType,
    compute_kernel: EmbeddingComputeKernel,
    batch_sizes: List[int],
    pooling_factors: Optional[List[float]] = None,
    num_poolings: Optional[List[float]] = None,
) -> Union[EmbeddingShardingPlanner, HeteroEmbeddingShardingPlanner]:
    """
    Generate an embedding sharding planner based on the specified configuration.

    Args:
        planner_type: Type of planner to use ("embedding" or "hetero")
        topology: Network topology for distributed training
        tables: List of unweighted embedding tables
        weighted_tables: List of weighted embedding tables
        sharding_type: Strategy for sharding embedding tables
        compute_kernel: Compute kernel to use for embedding tables
        batch_sizes: Sizes of each batch
        pooling_factors: Pooling factors for each feature of the table
        num_poolings: Number of poolings for each feature of the table

    Returns:
        An instance of EmbeddingShardingPlanner or HeteroEmbeddingShardingPlanner

    Raises:
        RuntimeError: If an unknown planner type is specified
    """
    # Create parameter constraints for tables
    constraints = {}
    num_batches = len(batch_sizes)

    if pooling_factors is None:
        pooling_factors = [POOLING_FACTOR] * num_batches

    if num_poolings is None:
        num_poolings = [NUM_POOLINGS] * num_batches

    assert (
        len(pooling_factors) == num_batches and len(num_poolings) == num_batches
    ), "The length of pooling_factors and num_poolings must match the number of batches."

    if tables is not None:
        for table in tables:
            constraints[table.name] = ParameterConstraints(
                sharding_types=[sharding_type.value],
                compute_kernels=[compute_kernel.value],
                device_group="cuda",
                pooling_factors=pooling_factors,
                num_poolings=num_poolings,
                batch_sizes=batch_sizes,
            )

    if weighted_tables is not None:
        for table in weighted_tables:
            constraints[table.name] = ParameterConstraints(
                sharding_types=[sharding_type.value],
                compute_kernels=[compute_kernel.value],
                device_group="cuda",
                pooling_factors=pooling_factors,
                num_poolings=num_poolings,
                batch_sizes=batch_sizes,
                is_weighted=True,
            )

    if planner_type == "embedding":
        return EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints if constraints else None,
        )
    elif planner_type == "hetero":
        topology_groups = {"cuda": topology}
        return HeteroEmbeddingShardingPlanner(
            topology_groups=topology_groups,
            constraints=constraints if constraints else None,
        )
    else:
        raise RuntimeError(f"Unknown planner type: {planner_type}")


def generate_sharded_model_and_optimizer(
    model: nn.Module,
    sharding_type: str,
    kernel_type: str,
    pg: dist.ProcessGroup,
    device: torch.device,
    fused_params: Dict[str, Any],
    dense_optimizer: str = "SGD",
    dense_lr: float = 0.1,
    dense_momentum: Optional[float] = None,
    dense_weight_decay: Optional[float] = None,
    planner: Optional[
        Union[
            EmbeddingShardingPlanner,
            HeteroEmbeddingShardingPlanner,
        ]
    ] = None,
) -> Tuple[nn.Module, Optimizer]:
    """
    Generate a sharded model and optimizer for distributed training.

    Args:
        model: The model to be sharded
        sharding_type: Type of sharding strategy
        kernel_type: Type of compute kernel
        pg: Process group for distributed training
        device: Device to place the model on
        fused_params: Parameters for the fused optimizer
        dense_optimizer: Optimizer type for dense parameters
        dense_lr: Learning rate for dense parameters
        dense_momentum: Momentum for dense parameters (optional)
        dense_weight_decay: Weight decay for dense parameters (optional)
        planner: Optional planner for sharding strategy

    Returns:
        Tuple of sharded model and optimizer
    """
    sharder = TestEBCSharder(
        sharding_type=sharding_type,
        kernel_type=kernel_type,
        fused_params=fused_params,
    )
    sharders = [cast(ModuleSharder[nn.Module], sharder)]

    # Use planner if provided
    plan = None
    if planner is not None:
        if pg is not None:
            plan = planner.collective_plan(model, sharders, pg)
        else:
            plan = planner.plan(model, sharders)

    sharded_model = DistributedModelParallel(
        module=copy.deepcopy(model),
        env=ShardingEnv.from_process_group(pg),
        init_data_parallel=True,
        device=device,
        sharders=sharders,
        plan=plan,
    ).to(device)

    # Get dense parameters
    dense_params = [
        param
        for name, param in sharded_model.named_parameters()
        if "sparse" not in name
    ]

    # Create optimizer based on the specified type
    optimizer_class = getattr(optim, dense_optimizer)

    # Create optimizer with momentum and/or weight_decay if provided
    optimizer_kwargs = {"lr": dense_lr}

    if dense_momentum is not None:
        optimizer_kwargs["momentum"] = dense_momentum

    if dense_weight_decay is not None:
        optimizer_kwargs["weight_decay"] = dense_weight_decay

    optimizer = optimizer_class(dense_params, **optimizer_kwargs)

    return sharded_model, optimizer


def _init_module_and_run_benchmark(
    module: torch.nn.Module,
    sharding_type: ShardingType,
    planner_type: str,
    compute_kernel: EmbeddingComputeKernel,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    batch_size: int,
    num_benchmarks: int,
    world_size: int,
    num_float_features: int = 0,
    rank: int = -1,
    queue: Optional[mp.Queue] = None,
    device_type: str = "cuda",
    warmup_iters: int = 20,
    bench_iters: int = 100,
    prof_iters: int = 20,
) -> None:
    """
    Initialize module and run benchmark for a single process.

    This is a simplified version of init_module_and_run_benchmark from benchmark_ebc.py
    that doesn't handle compile modes and focuses on the core benchmarking functionality.
    """
    from torchrec.distributed.comm import get_local_size

    # Generate input data
    num_inputs_to_gen = warmup_iters + bench_iters + prof_iters

    batch_sizes = [batch_size] * num_inputs_to_gen
    inputs_batch = []

    for _ in range(num_inputs_to_gen):
        model_input_by_rank = []
        for _ in range(world_size):
            model_input_by_rank.append(
                ModelInput.generate(
                    batch_size=batch_size,
                    num_float_features=num_float_features,
                    tables=tables,
                    weighted_tables=weighted_tables,
                    indices_dtype=torch.int32,
                    lengths_dtype=torch.int32,
                )
            )

        inputs_batch.append(model_input_by_rank)

    # Transpose to get inputs by rank: [R x B] format
    inputs_by_rank = list(zip(*inputs_batch))

    if rank >= 0:
        warmup_inputs_cuda = [
            warmup_input.to(torch.device(f"{device_type}:{rank}"))
            for warmup_input in inputs_by_rank[rank][:warmup_iters]
        ]
        bench_inputs_cuda = [
            bench_input.to(torch.device(f"{device_type}:{rank}"))
            for bench_input in inputs_by_rank[rank][
                warmup_iters : warmup_iters + bench_iters
            ]
        ]
        prof_inputs_cuda = [
            prof_input.to(torch.device(f"{device_type}:{rank}"))
            for prof_input in inputs_by_rank[rank][-prof_iters:]
        ]
    else:
        warmup_inputs_cuda = [
            warmup_input.to(torch.device(f"{device_type}:0"))
            for warmup_input in inputs_by_rank[0][:warmup_iters]
        ]
        bench_inputs_cuda = [
            bench_input.to(torch.device(f"{device_type}:0"))
            for bench_input in inputs_by_rank[0][
                warmup_iters : warmup_iters + bench_iters
            ]
        ]
        prof_inputs_cuda = [
            prof_input.to(torch.device(f"{device_type}:0"))
            for prof_input in inputs_by_rank[0][-prof_iters:]
        ]

    with (
        MultiProcessContext(
            rank, world_size, "nccl", use_deterministic_algorithms=False
        )
        if rank != -1
        else contextlib.nullcontext()
    ) as ctx:
        # Create topology and planner
        topology = Topology(
            local_world_size=get_local_size(world_size),
            world_size=world_size,
            compute_device=device_type,
        )

        planner = generate_planner(
            planner_type=planner_type,
            topology=topology,
            tables=tables,
            weighted_tables=weighted_tables,
            sharding_type=sharding_type,
            compute_kernel=compute_kernel,
            batch_sizes=batch_sizes[
                :num_benchmarks
            ],  # Use only benchmark batches for planning
        )

        # Prepare fused_params for sparse optimizer
        fused_params = {
            "optimizer": EmbOptimType.EXACT_ADAGRAD,
            "learning_rate": 0.1,
        }

        device = ctx.device if rank != -1 else torch.device(device_type)
        pg = ctx.pg if rank != -1 else None

        sharded_model, _ = generate_sharded_model_and_optimizer(
            model=module,
            sharding_type=sharding_type.value,
            kernel_type=compute_kernel.value,
            pg=pg,
            device=device,
            fused_params=fused_params,
            planner=planner,
        )

        def _func_to_benchmark(
            model: torch.nn.Module, bench_inputs: List[KeyedJaggedTensor]
        ) -> None:
            with torch.inference_mode():
                for bench_input in bench_inputs:
                    model(bench_input)

        name = f"{sharding_type.value}-{planner_type}"

        res = benchmark(
            name,
            sharded_model,
            warmup_inputs_cuda,
            bench_inputs_cuda,
            prof_inputs_cuda,
            world_size=world_size,
            output_dir="",
            num_benchmarks=num_benchmarks,
            func_to_benchmark=_func_to_benchmark,
            benchmark_func_kwargs=None,
            rank=rank,
            device_type=device_type,
            benchmark_unsharded_module=False,
        )

        if queue is not None:
            queue.put(res)


def benchmark_module(
    module: torch.nn.Module,
    tables: List[EmbeddingBagConfig],
    weighted_tables: Optional[List[EmbeddingBagConfig]] = None,
    num_float_features: int = 0,
    sharding_type: ShardingType = ShardingType.TABLE_WISE,
    planner_type: str = "embedding",
    world_size: int = 2,
    num_benchmarks: int = 5,
    batch_size: int = 2048,
    compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.FUSED,
    device_type: str = "cuda",
) -> BenchmarkResult:
    """
    Benchmark any PyTorch module with distributed sharding.

    This function provides a simple interface to benchmark arbitrary PyTorch modules
    using TorchRec's distributed sharding capabilities. It uses the provided embedding
    tables to generate input data, sets up multiprocessing for distributed training,
    and returns comprehensive benchmark results.

    Args:
        module: PyTorch module to benchmark
        tables: List of unweighted embedding table configurations
        weighted_tables: Optional list of weighted embedding table configurations
        sharding_type: Strategy for sharding embedding tables across devices
        planner_type: Type of planner to use ("embedding" or "hetero")
        world_size: Number of processes/GPUs to use for distributed training
        num_benchmarks: Number of iterations to run for statistical analysis
        batch_size: Batch size to use for benchmarking
        compute_kernel: Compute kernel to use for embedding tables
        device_type: Device type to use ("cuda" or "cpu")

    Returns:
        BenchmarkResult containing timing and memory statistics

    Example:
        from torchrec.modules.embedding_modules import EmbeddingBagCollection
        from torchrec.modules.embedding_configs import EmbeddingBagConfig

        # Create embedding tables
        tables = [
            EmbeddingBagConfig(
                name="table_0", embedding_dim=128, num_embeddings=100000,
                feature_names=["feature_0"]
            )
        ]

        # Create a simple EBC module
        ebc = EmbeddingBagCollection(tables=tables)

        # Benchmark it
        result = benchmark_module(
            module=ebc,
            tables=tables,
            world_size=2,
            num_benchmarks=10
        )
        print(result)
    """
    logger.info(f"Starting benchmark for module: {type(module).__name__}")
    logger.info(f"Sharding type: {sharding_type}")
    logger.info(f"Planner type: {planner_type}")
    logger.info(f"World size: {world_size}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of benchmarks: {num_benchmarks}")

    assert (
        num_benchmarks > 2
    ), "num_benchmarks needs to be greater than 2 for statistical analysis"

    # Use provided tables or default to empty list for weighted tables
    if weighted_tables is None:
        weighted_tables = []

    # Use multiprocessing for distributed benchmarking (always assume train mode)
    res = multi_process_benchmark(
        callable=_init_module_and_run_benchmark,
        module=module,
        sharding_type=sharding_type,
        planner_type=planner_type,
        compute_kernel=compute_kernel,
        tables=tables,
        weighted_tables=weighted_tables,
        batch_size=batch_size,
        num_benchmarks=num_benchmarks,
        world_size=world_size,
        num_float_features=num_float_features,
        device_type=device_type,
    )

    return res


def _load_config_file(config_path: str, is_json: bool = False) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    if not config_path:
        return {}
    
    try:
        with open(config_path, "r") as f:
            if is_json:
                return json.load(f) or {}
            else:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}. Proceeding without it.")
        return {}


def _get_default_value(field_info: Any, config_data: Dict[str, Any], cls_name: str) -> Any:
    """Get default value for a dataclass field from config or field default."""
    arg_name = field_info.name
    
    # Try hierarchical lookup: class_name.field_name
    snake_case_class_name = ''.join(['_' + c.lower() if c.isupper() else c for c in cls_name]).lstrip('_')
    
    # Check nested config structure first
    for class_key in [snake_case_class_name, cls_name.lower(), cls_name]:
        if class_key in config_data and arg_name in config_data[class_key]:
            return config_data[class_key][arg_name]
    
    # Check flat structure
    if arg_name in config_data:
        return config_data[arg_name]
    
    # Use field default
    if field_info.default_factory is not MISSING:
        return field_info.default_factory()
    return field_info.default


def _get_field_type_info(field_type: Any) -> Tuple[Any, Any]:
    """Extract the actual type and origin from a field type, handling Optional."""
    origin = get_origin(field_type)
    
    # Handle Optional[X] -> X
    if origin is Union and type(None) in get_args(field_type):
        non_none_types = [t for t in get_args(field_type) if t is not type(None)]
        if len(non_none_types) == 1:
            field_type = non_none_types[0]
            origin = get_origin(field_type)
    
    return field_type, origin


def _create_argument_parser_kwargs(field_type: Any, origin: Any, default_value: Any, cls_name: str, field_name: str) -> Dict[str, Any]:
    """Create kwargs for argparse.add_argument based on field type."""
    arg_kwargs = {
        "default": default_value,
        "help": f"({cls_name}) {field_name}",
    }
    
    if origin in (list, List):
        elem_type = get_args(field_type)[0]
        arg_kwargs.update(nargs="*", type=elem_type)
    elif field_type is bool:
        arg_kwargs.update(type=lambda x: x.lower() in ["true", "1", "yes"])
    elif isinstance(field_type, type) and issubclass(field_type, Enum):
        # Create a safe enum converter that handles all cases
        def safe_enum_converter(value):
            # If it's already the right enum type, return as-is
            if isinstance(value, field_type):
                return value
            # If it's a string, try to convert
            if isinstance(value, str):
                return field_type(value)
            # For any other type, try to convert to string first, then enum
            return field_type(str(value))
        arg_kwargs.update(type=safe_enum_converter)
    else:
        arg_kwargs.update(type=field_type)
    
    return arg_kwargs


def _build_dataclass_from_args(cls: Any, args: argparse.Namespace) -> Any:
    """Build a dataclass instance from parsed arguments."""
    data = {}
    for field_info in fields(cls):
        value = getattr(args, field_info.name)
        
        # Note: Enum conversion should already be handled by argparse type converters
        
        data[field_info.name] = value
    
    return cls(**data)


def cmd_conf(func: Callable) -> Callable:
    """
    Simplified command-line configuration decorator.
    
    This decorator automatically creates command-line arguments for all dataclass
    parameters of the decorated function and supports loading defaults from YAML/JSON files.
    """
    def wrapper() -> Any:
        sig = inspect.signature(func)
        parser = argparse.ArgumentParser(description=func.__doc__)
        
        # Add config file arguments
        parser.add_argument("--yaml_config", type=str, help="YAML config file path")
        parser.add_argument("--json_config", type=str, help="JSON config file path")
        parser.add_argument(
            "--loglevel", 
            type=str, 
            default="INFO",
            help="Set the logging level (DEBUG, INFO, WARNING, ERROR)"
        )
        
        # Parse config files first
        pre_args, _ = parser.parse_known_args()
        
        yaml_config = _load_config_file(pre_args.yaml_config, is_json=False)
        json_config = _load_config_file(pre_args.json_config, is_json=True)
        merged_config = {**yaml_config, **json_config}  # JSON overrides YAML
        
        # Add arguments for each dataclass parameter
        seen_args = set()
        for param_name, param in sig.parameters.items():
            cls = param.annotation
            if not is_dataclass(cls):
                continue
            
            for field_info in fields(cls):
                arg_name = field_info.name
                if arg_name in seen_args:
                    logger.warning(f"Duplicate argument: {arg_name}")
                    continue
                seen_args.add(arg_name)
                
                field_type, origin = _get_field_type_info(field_info.type)
                default_value = _get_default_value(field_info, merged_config, cls.__name__)
                
                # Let argparse handle enum conversion - don't convert defaults here
                
                arg_kwargs = _create_argument_parser_kwargs(
                    field_type, origin, default_value, cls.__name__, field_info.name
                )
                
                parser.add_argument(f"--{arg_name}", **arg_kwargs)
        
        # Parse all arguments
        args = parser.parse_args()
        
        # Set logging level
        loglevel = getattr(logging, args.loglevel.upper(), logging.INFO)
        logger.setLevel(loglevel)
        
        # Build dataclass instances
        kwargs = {}
        for param_name, param in sig.parameters.items():
            cls = param.annotation
            if is_dataclass(cls):
                config_instance = _build_dataclass_from_args(cls, args)
                kwargs[param_name] = config_instance
                logger.info(f"{cls.__name__}: {config_instance}")
        
        return func(**kwargs)
    
    return wrapper


def init_argparse_and_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=500)
    parser.add_argument("--prof_iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--max_num_embeddings", type=int, default=1000000)
    parser.add_argument("--output_dir", type=str, default="/var/tmp/torchrec-bench")
    parser.add_argument("--num_benchmarks", type=int, default=5)
    parser.add_argument("--embedding_config_json", type=str, default="")
    parser.add_argument("--device_type", type=str, default="cuda")

    args = parser.parse_args()

    return args


def _run_benchmark_core(
    name: str,
    run_iter_fn: Callable[[], None],
    profile_iter_fn: Optional[Callable[[Any], None]],  # pyre-ignore [2]
    world_size: int,
    rank: int,
    num_benchmarks: int,
    device_type: str,
    output_dir: str,
    pre_gpu_load: int = 0,
    export_stacks: bool = False,
    reset_accumulated_memory_stats: bool = False,
) -> BenchmarkResult:
    """Internal helper that contains the core benchmarking logic shared by
    ``benchmark`` and ``benchmark_func``.  All heavy–lifting (timing, memory
    accounting, optional profiling) happens here so the public helpers can stay
    small and focused on preparing the callables to execute.

    Args:
        name: Human-readable benchmark name.
        run_iter_fn: Zero-arg callable that executes one measured iteration.
        profile_iter_fn: Optional callable that receives a ``torch.profiler``
            instance and runs the iterations that should be captured.
        world_size, rank: Distributed context to correctly reset / collect GPU
            stats. ``rank == -1`` means single-process mode.
        num_benchmarks: Number of measured iterations.
        device_type: "cuda" or "cpu".
        output_dir: Where to write chrome traces / stack files.
        pre_gpu_load: Number of dummy matmul operations to run before the first
            measured iteration (helps simulating a loaded allocator).
        export_stacks: Whether to export flamegraph-compatible stack files.
        reset_accumulated_memory_stats: Whether to reset accumulated memory
            stats in addition to peak memory stats.
    """

    # Preparation & memory reset
    if device_type == "cuda":
        if rank == -1:
            for di in range(world_size):
                torch.cuda.reset_peak_memory_stats(di)
                if reset_accumulated_memory_stats:
                    torch.cuda.reset_accumulated_memory_stats(di)
        else:
            torch.cuda.reset_peak_memory_stats(rank)
            if reset_accumulated_memory_stats:
                torch.cuda.reset_accumulated_memory_stats(rank)

        # Optional allocator warm-up to create fragmentation similar to production
        if pre_gpu_load:
            _tmp = torch.rand(16384, 16384, device="cuda")
            for _ in range(pre_gpu_load):
                _tmp = _tmp * torch.rand(16384, 16384, device="cuda")

    # Timings
    start_events, end_events, times = [], [], []

    if device_type == "cuda":
        start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)
        ]
        end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)
        ]
        # Capture per-iteration active CPU cycles (excludes time the thread is truly idle/asleep) using `process_time_ns`.
        cpu_times_active_ns: List[int] = []

        for i in range(num_benchmarks):
            # Ensure that outstanding GPU work from the previous iteration has
            # finished so that we do not attribute its wait time to the next
            # CPU measurement.
            if i > 0:
                torch.cuda.synchronize(rank if rank >= 0 else 0)

            start_events[i].record()
            cpu_start_active_ns = time.process_time_ns()

            run_iter_fn()

            cpu_end_active_ns = time.process_time_ns()
            end_events[i].record()
            cpu_times_active_ns.append(cpu_end_active_ns - cpu_start_active_ns)

        # Convert to milliseconds and drop the first iteration
        cpu_elapsed_time = torch.tensor(
            [t / 1e6 for t in cpu_times_active_ns[1:]], dtype=torch.float
        )

        # Make sure all kernels are finished before reading timers / stats
        if rank == -1:
            for di in range(world_size):
                torch.cuda.synchronize(di)
        else:
            torch.cuda.synchronize(rank)

        gpu_elapsed_time = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_events[1:], end_events[1:])]
        )
    else:
        # For CPU-only benchmarks we fall back to wall-clock timing via ``timeit``.
        times = timeit.repeat(run_iter_fn, number=1, repeat=num_benchmarks)
        cpu_elapsed_time = torch.tensor(times) * 1e3  # convert to ms

        # mirror CPU timings for overall consistency
        gpu_elapsed_time = cpu_elapsed_time.clone()

    # Memory statistics collection
    gpu_mem_stats: List[GPUMemoryStats] = []
    cpu_mem_stats = [CPUMemoryStats.for_process(rank)]

    if device_type == "cuda":
        if rank == -1:
            for di in range(world_size):
                gpu_mem_stats.append(GPUMemoryStats.for_device(di))
        else:
            gpu_mem_stats.append(GPUMemoryStats.for_device(rank))
    # CPU memory stats are collected for both GPU and CPU-only runs

    # Optional detailed profiling
    if output_dir and profile_iter_fn and device_type == "cuda":

        def _trace_handler(prof: torch.profiler.profile) -> None:
            total_avg = prof.profiler.total_average()
            logger.info(f" TOTAL_AVERAGE:\n{name}\n{total_avg}")
            if rank > 0:
                return
            trace_file = f"{output_dir}/trace-{name}.json"
            logger.info(f" PROFILE[{name}].chrome_trace:{trace_file}")
            prof.export_chrome_trace(trace_file)
            if export_stacks:
                prof.export_stacks(
                    f"{output_dir}/stacks-cpu-{name}.stacks", "self_cpu_time_total"
                )
                prof.export_stacks(
                    f"{output_dir}/stacks-cuda-{name}.stacks", "self_cuda_time_total"
                )

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=export_stacks,
            on_trace_ready=_trace_handler,
        ) as prof:
            profile_iter_fn(prof)

        # Synchronize again after profiling to guarantee deterministic ordering
        if rank == -1:
            for di in range(torch.cuda.device_count()):
                torch.cuda.synchronize(torch.device(f"cuda:{di}"))
        else:
            torch.cuda.synchronize(rank)

    return BenchmarkResult(
        short_name=name,
        gpu_elapsed_time=gpu_elapsed_time,
        cpu_elapsed_time=cpu_elapsed_time,
        gpu_mem_stats=gpu_mem_stats,
        cpu_mem_stats=cpu_mem_stats,
        rank=rank,
    )


def benchmark(
    name: str,
    model: torch.nn.Module,
    warmup_inputs: Union[List[KeyedJaggedTensor], List[Dict[str, Any]]],
    bench_inputs: Union[List[KeyedJaggedTensor], List[Dict[str, Any]]],
    prof_inputs: Union[List[KeyedJaggedTensor], List[Dict[str, Any]]],
    world_size: int,
    output_dir: str,
    num_benchmarks: int,
    # pyre-ignore[2]
    func_to_benchmark: Any,
    benchmark_func_kwargs: Optional[Dict[str, Any]],
    rank: int,
    enable_logging: bool = True,
    device_type: str = "cuda",
    benchmark_unsharded_module: bool = False,
    export_stacks: bool = False,
) -> BenchmarkResult:
    if enable_logging:
        logger.info(f" BENCHMARK_MODEL[{name}]:\n{model}")

    # Warm-up forwards to stabilize kernels / JIT compilation
    for _input in warmup_inputs:
        model(_input)

    if benchmark_func_kwargs is None:
        benchmark_func_kwargs = {}

    run_iter_fn: Callable[[], None] = lambda: func_to_benchmark(
        model, bench_inputs, **benchmark_func_kwargs
    )

    def _profile_iter_fn(prof: torch.profiler.profile) -> None:
        for _input in prof_inputs:
            with record_function("## forward ##"):
                model(_input)
                prof.step()

    return _run_benchmark_core(
        name=name,
        run_iter_fn=run_iter_fn,
        profile_iter_fn=_profile_iter_fn if output_dir else None,
        world_size=world_size,
        rank=rank,
        num_benchmarks=num_benchmarks,
        device_type=device_type,
        output_dir=output_dir,
        pre_gpu_load=0,
        export_stacks=export_stacks,
        reset_accumulated_memory_stats=False,
    )


def benchmark_func(
    name: str,
    bench_inputs: List[Dict[str, Any]],
    prof_inputs: List[Dict[str, Any]],
    world_size: int,
    profile_dir: str,
    num_benchmarks: int,
    num_profiles: int,
    # pyre-ignore[2]
    func_to_benchmark: Any,
    benchmark_func_kwargs: Optional[Dict[str, Any]],
    rank: int,
    device_type: str = "cuda",
    pre_gpu_load: int = 0,
    export_stacks: bool = False,
) -> BenchmarkResult:
    if benchmark_func_kwargs is None:
        benchmark_func_kwargs = {}

    run_iter_fn: Callable[[], None] = lambda: func_to_benchmark(
        bench_inputs, **benchmark_func_kwargs
    )

    def _profile_iter_fn(prof: torch.profiler.profile) -> None:
        for i in range(num_profiles):
            with record_function(f"## profile {i} ##"):
                func_to_benchmark(prof_inputs, **benchmark_func_kwargs)
                prof.step()

    return _run_benchmark_core(
        name=name,
        run_iter_fn=run_iter_fn,
        profile_iter_fn=_profile_iter_fn if profile_dir else None,
        world_size=world_size,
        rank=rank,
        num_benchmarks=num_benchmarks,
        device_type=device_type,
        output_dir=profile_dir,
        pre_gpu_load=pre_gpu_load,
        export_stacks=export_stacks,
        reset_accumulated_memory_stats=True,
    )
