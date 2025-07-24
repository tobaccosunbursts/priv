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
import copy
import gc
import inspect
import json
import logging
import os
import time
import timeit
from dataclasses import dataclass, fields, is_dataclass, MISSING
from enum import Enum
from typing import (
    Any,
    Callable,
    ContextManager,
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
import yaml
from torch import multiprocessing as mp
from torch.autograd.profiler import record_function
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import ShardingType
from torchrec.distributed.global_settings import set_propogate_device

from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.multi_process import MultiProcessContext
from torchrec.distributed.test_utils.test_model import ModelInput

from torchrec.distributed.types import DataType, ModuleSharder, ShardingEnv
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import get_free_port
import psutil
import resource

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
    peak_rss_mbs: int
    current_rss_mbs: int

    @classmethod
    def for_process(cls) -> "CPUMemoryStats":
        # Peak RSS from resource.getrusage (in KB on Linux)
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = peak_rss_kb // 1024
        
        # Current RSS from psutil (in bytes)
        process = psutil.Process()
        current_rss_mb = process.memory_info().rss // 1024 // 1024
        
        return cls(peak_rss_mb, current_rss_mb)

    def __str__(self) -> str:
        return f"CPU: peak_rss={self.peak_rss_mbs:7}mb, current_rss={self.current_rss_mbs:7}mb"


@dataclass
class BenchmarkResult:
    "Class for holding results of benchmark runs"
    short_name: str
    gpu_elapsed_time: torch.Tensor  # milliseconds
    cpu_elapsed_time: torch.Tensor  # milliseconds
    gpu_mem_stats: List[GPUMemoryStats]  # GPU memory stats per rank
    cpu_mem_stats: CPUMemoryStats  # CPU memory stats
    rank: int = -1

    def __str__(self) -> str:
        gpu_runtime = (
            f"GPU Runtime (P90): {self.runtime_percentile(90, device='gpu'):.2f} ms"
        )
        cpu_runtime = (
            f"CPU Runtime (P90): {self.runtime_percentile(90, device='cpu'):.2f} ms"
        )
        if len(self.gpu_mem_stats) == 0:
            return f"{self.short_name: <{35}} |  {gpu_runtime} | {cpu_runtime} | {self.cpu_mem_stats}"
        mem_alloc = (
            f"Peak Memory alloc (P90): {self.max_mem_alloc_percentile(90)/1000:.2f} GB"
        )
        mem_reserved = f"Peak Memory reserved (P90): {self.max_mem_reserved_percentile(90)/1000:.2f} GB"
        malloc_retries = f"Malloc retries (P50/P90/P100): {self.mem_retries(50)} / {self.mem_retries(90)} / {self.mem_retries(100)}"
        return f"{self.short_name: <{35}} | {malloc_retries} | {gpu_runtime} | {cpu_runtime} | {mem_alloc} | {mem_reserved} | {self.cpu_mem_stats}"

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
            [mem_selector(mem_stat) for mem_stat in self.gpu_mem_stats], dtype=torch.float
        )
        return torch.quantile(mem_data, percentile / 100.0, interpolation=interpolation)


class ModuleWrapper(torch.nn.Module):
    """
    A wrapper for nn.modules that allows them to accept inputs
    of type KeyedJaggedTensor or ModelInput and forwards them to the
    underlying module. This wrapper is necessary to provide compatibility
    with FX tracing.

    Args:
        module: The torch.nn.Module to be wrapped.

    Example:
        import torch
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
        from torchrec.distributed.benchmark.benchmark_utils import ModuleWrapper

        # Create a simple module
        module = torch.nn.Linear(10, 5)
        wrapped_module = ModuleWrapper(module)

        # Create a KeyedJaggedTensor input
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        weights = None
        lengths = torch.tensor([2, 0, 1, 1, 3, 1])
        offsets = torch.tensor([0, 2, 2, 3, 4, 7, 8])
        keys = ["F1", "F2", "F3"]
        kjt = KeyedJaggedTensor(
            values=values,
            weights=weights,
            lengths=lengths,
            offsets=offsets,
            keys=keys,
        )

        # Forward the input through the wrapped module
        output = wrapped_module(kjt)
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(self, input):  # pyre-ignore[3]
        """
        Forward pass of the wrapped module.

        Args:
            input: Input of type KeyedJaggedTensor or ModelInput to be forwarded to the underlying module.

        Returns:
            The output from the underlying module's forward pass.
        """
        return self._module.forward(input)


T = TypeVar("T", bound=torch.nn.Module)


def default_func_to_benchmark(
    model: torch.nn.Module, bench_inputs: List[KeyedJaggedTensor]
) -> None:
    with torch.inference_mode():
        for bench_input in bench_inputs:
            model(bench_input)


def get_tables(
    table_sizes: List[Tuple[int, int]],
    is_pooled: bool = True,
    data_type: DataType = DataType.INT8,
) -> Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]:
    if is_pooled:
        tables: List[EmbeddingBagConfig] = [
            EmbeddingBagConfig(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                data_type=data_type,
            )
            for i, (num_embeddings, embedding_dim) in enumerate(table_sizes)
        ]
    else:
        tables: List[EmbeddingConfig] = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                data_type=data_type,
            )
            for i, (num_embeddings, embedding_dim) in enumerate(table_sizes)
        ]

    return tables


def get_inputs(
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    batch_size: int,
    world_size: int,
    num_inputs: int,
    num_float_features: int,
    train: bool,
    pooling_configs: Optional[List[int]] = None,
    variable_batch_embeddings: bool = False,
    only_kjt: bool = False,
) -> Union[List[List[ModelInput]], List[List[KeyedJaggedTensor]]]:
    """
    Generate inputs for benchmarking.

    Args:
        tables: List of embedding tables configurations
        batch_size: Batch size for generated inputs
        world_size: Number of ranks/processes
        num_inputs: Number of input batches to generate
        num_float_features: Number of float features
        train: Whether inputs are for training
        pooling_configs: Optional pooling factors for tables
        variable_batch_embeddings: Whether to use variable batch size
        only_kjt: If True, return KeyedJaggedTensor instead of ModelInput

    Returns:
        If only_kjt is False: List of lists of ModelInput objects
        If only_kjt is True: List of lists of KeyedJaggedTensor objects
    """
    inputs_batch = []

    if variable_batch_embeddings and not train:
        raise RuntimeError("Variable batch size is only supported in training mode")

    for _ in range(num_inputs):
        if variable_batch_embeddings:
            _, model_input_by_rank = ModelInput.generate_variable_batch_input(
                average_batch_size=batch_size,
                world_size=world_size,
                num_float_features=num_float_features,
                tables=tables,
            )
        else:
            _, model_input_by_rank = ModelInput.generate(
                batch_size=batch_size,
                world_size=world_size,
                num_float_features=num_float_features,
                tables=tables,
                weighted_tables=[],
                tables_pooling=pooling_configs,
                indices_dtype=torch.int32,
                lengths_dtype=torch.int32,
            )

        if train:
            inputs_batch.append(
                [
                    model_input
                    for model_input in model_input_by_rank
                    if isinstance(model_input.idlist_features, KeyedJaggedTensor)
                    or not only_kjt
                ]
            )
        else:
            assert (
                isinstance(model_input_by_rank[0].idlist_features, KeyedJaggedTensor)
                or not only_kjt
            )
            inputs_batch.append([model_input_by_rank[0]])

    # If only_kjt is True, extract idlist_features from ModelInput objects
    if only_kjt:
        inputs_batch = [
            [model_input.idlist_features for model_input in batch]
            for batch in inputs_batch
        ]

    # Transpose if train, as inputs_by_rank is currently in  [B X R] format
    inputs_by_rank = [
        list(model_inputs_rank) for model_inputs_rank in zip(*inputs_batch)
    ]

    return inputs_by_rank


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
        mem_str += f"{benchmark_res.cpu_mem_stats}\n"

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


# pyre-ignore [24]
def cmd_conf(func: Callable) -> Callable:

    def _load_config_file(config_path: str, is_json: bool = False) -> Dict[str, Any]:
        if not config_path:
            return {}
        try:
            with open(config_path, "r") as f:
                if is_json:
                    return json.load(f) or {}
                else:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load {'JSON' if is_json else 'YAML'} config because {e}. Proceeding without it.")
            return {}

    # pyre-ignore [3]
    def wrapper() -> Any:
        sig = inspect.signature(func)
        parser = argparse.ArgumentParser(func.__doc__)

        parser.add_argument(
            "--yaml_config",
            type=str,
            default=None,
            help="YAML config file for benchmarking",
        )
        parser.add_argument(
            "--json_config",
            type=str,
            default=None,
            help="JSON config file for benchmarking",
        )
        # Add loglevel argument with current logger level as default
        parser.add_argument(
            "--loglevel",
            type=str,
            default=logging._levelToName[logger.level],
            help="Set the logging level (e.g. info, debug, warning, error)",
        )

        pre_args, _ = parser.parse_known_args()

        # Load YAML and JSON configs, JSON overrides YAML
        yaml_defaults: Dict[str, Any] = _load_config_file(pre_args.yaml_config, is_json=False) if pre_args.yaml_config else {}
        json_defaults: Dict[str, Any] = _load_config_file(pre_args.json_config, is_json=True) if pre_args.json_config else {}
        # Merge: JSON overrides YAML
        merged_defaults = {**yaml_defaults, **json_defaults}

        logger.info(f"Loaded config defaults: {merged_defaults}")

        seen_args = set()  # track all --<name> we've added

        for _name, param in sig.parameters.items():
            cls = param.annotation
            if not is_dataclass(cls):
                continue

            for f in fields(cls):
                arg_name = f.name
                if arg_name in seen_args:
                    logger.warning(f"WARNING: duplicate argument {arg_name}")
                    continue
                seen_args.add(arg_name)

                ftype = f.type
                origin = get_origin(ftype)

                # Unwrapping Optional[X] to X
                if origin is Union and type(None) in get_args(ftype):
                    non_none = [t for t in get_args(ftype) if t is not type(None)]
                    if len(non_none) == 1:
                        ftype = non_none[0]
                        origin = get_origin(ftype)

                # Handle default_factory value and allow config to override it
                default_value = merged_defaults.get(
                    arg_name,  # flat lookup
                    merged_defaults.get(cls.__name__, {}).get(  # hierarchy lookup
                        arg_name,
                        (
                            f.default_factory()  # pyre-ignore [29]
                            if f.default_factory is not MISSING
                            else f.default
                        ),
                    ),
                )

                arg_kwargs = {
                    "default": default_value,
                    "help": f"({cls.__name__}) {arg_name}",
                }

                if origin in (list, List):
                    elem_type = get_args(ftype)[0]
                    arg_kwargs.update(nargs="*", type=elem_type)
                elif ftype is bool:
                    # Special handling for boolean arguments
                    arg_kwargs.update(type=lambda x: x.lower() in ["true", "1", "yes"])
                else:
                    arg_kwargs.update(type=ftype)

                parser.add_argument(f"--{arg_name}", **arg_kwargs)

        args = parser.parse_args()
        logger.setLevel(logging.INFO)

        # Build the dataclasses
        kwargs = {}
        for name, param in sig.parameters.items():
            cls = param.annotation
            if is_dataclass(cls):
                data = {f.name: getattr(args, f.name) for f in fields(cls)}
                config_instance = cls(**data)  # pyre-ignore [29]
                kwargs[name] = config_instance
                logger.info(config_instance)

        loglevel = logging._nameToLevel[args.loglevel.upper()]
        logger.setLevel(loglevel)

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


def transform_module(
    module: torch.nn.Module,
    device: torch.device,
    inputs: Union[List[ModelInput], List[KeyedJaggedTensor]],
    sharder: ModuleSharder[T],
    sharding_type: ShardingType,
    compile_mode: CompileMode,
    world_size: int,
    batch_size: int,
    # pyre-fixme[24]: Generic type `ContextManager` expects 1 type parameter.
    ctx: ContextManager,
    benchmark_unsharded_module: bool = False,
) -> torch.nn.Module:
    def fx_script_module(eager_module: torch.nn.Module) -> torch.nn.Module:
        eager_module(inputs[0])
        graph_module = symbolic_trace(
            eager_module, leaf_modules=["IntNBitTableBatchedEmbeddingBagsCodegen"]
        )
        scripted_module = torch.jit.script(graph_module)
        return scripted_module

    set_propogate_device(True)

    sharded_module = None

    if not benchmark_unsharded_module:
        topology: Topology = Topology(world_size=world_size, compute_device=device.type)
        planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        # Don't want to modify the module outright
        # Since module is on cpu, won't cause cuda oom.
        copied_module = copy.deepcopy(module)
        # pyre-ignore [6]
        plan = planner.plan(copied_module, [sharder])

        if isinstance(ctx, MultiProcessContext):
            sharded_module = DistributedModelParallel(
                copied_module,
                # pyre-ignore[6]
                env=ShardingEnv.from_process_group(ctx.pg),
                plan=plan,
                # pyre-ignore[6]
                sharders=[sharder],
                device=ctx.device,
            )
        else:
            env = ShardingEnv.from_local(world_size=topology.world_size, rank=0)

            sharded_module = _shard_modules(
                module=copied_module,
                # pyre-fixme[6]: For 2nd argument expected
                #  `Optional[List[ModuleSharder[Module]]]` but got
                #  `List[ModuleSharder[Variable[T (bound to Module)]]]`.
                sharders=[sharder],
                device=device,
                plan=plan,
                env=env,
            )

    if compile_mode == CompileMode.FX_SCRIPT:
        return fx_script_module(
            # pyre-fixme[6]: For 1st argument expected `Module` but got
            #  `Optional[Module]`.
            sharded_module
            if not benchmark_unsharded_module
            else module
        )
    else:
        # pyre-fixme[7]: Expected `Module` but got `Optional[Module]`.
        return sharded_module if not benchmark_unsharded_module else module


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
    cpu_mem_stats = CPUMemoryStats.for_process()
    
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
    warmup_inputs: Union[List[KeyedJaggedTensor], List[ModelInput], List[Dict[str, Any]]],
    bench_inputs: Union[List[KeyedJaggedTensor], List[ModelInput], List[Dict[str, Any]]],
    prof_inputs: Union[List[KeyedJaggedTensor], List[ModelInput], List[Dict[str, Any]]],
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
        export_stacks=True,
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
        export_stacks=False,
        reset_accumulated_memory_stats=True,
    )


def benchmark_type_name(compile_mode: CompileMode, sharding_type: ShardingType) -> str:
    if sharding_type == ShardingType.TABLE_WISE:
        name = "tw-sharded"
    elif sharding_type == ShardingType.ROW_WISE:
        name = "rw-sharded"
    elif sharding_type == ShardingType.COLUMN_WISE:
        name = "cw-sharded"
    else:
        raise Exception(f"Unknown sharding type {sharding_type}")

    if compile_mode == CompileMode.EAGER:
        name += "-eager"
    elif compile_mode == CompileMode.FX_SCRIPT:
        name += "-fxjit"

    return name


def init_module_and_run_benchmark(
    module: torch.nn.Module,
    sharder: ModuleSharder[T],
    device: torch.device,
    sharding_type: ShardingType,
    compile_mode: CompileMode,
    world_size: int,
    batch_size: int,
    warmup_inputs: Union[List[List[ModelInput]], List[List[KeyedJaggedTensor]]],
    bench_inputs: Union[List[List[ModelInput]], List[List[KeyedJaggedTensor]]],
    prof_inputs: Union[List[List[ModelInput]], List[List[KeyedJaggedTensor]]],
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    output_dir: str,
    num_benchmarks: int,
    # pyre-ignore[2]
    func_to_benchmark: Any,
    benchmark_func_kwargs: Optional[Dict[str, Any]],
    rank: int = -1,
    queue: Optional[mp.Queue] = None,
    pooling_configs: Optional[List[int]] = None,
    benchmark_unsharded_module: bool = False,
) -> BenchmarkResult:
    """
    There are a couple of caveats here as to why the module has to be initialized
    here:
    1. Device. To accurately track memory usage, when sharding modules the initial
       placement of the module should be on CPU. This is to avoid double counting
       memory allocations and also to prevent CUDA OOMs.
    2. Garbage Collector. Since torch.fx.GraphModule has circular references,
       garbage collection us funky and can lead to ooms. Since this frame is
       called by the loop through compile modes and sharding types, returning the
       benchmark result will mean that the reference to module is lost instead of
       existing in the loop
    """

    if rank >= 0:
        warmup_inputs_cuda = [
            warmup_input.to(torch.device(f"{device.type}:{rank}"))
            for warmup_input in warmup_inputs[rank]
        ]
        bench_inputs_cuda = [
            bench_input.to(torch.device(f"{device.type}:{rank}"))
            for bench_input in bench_inputs[rank]
        ]
        prof_inputs_cuda = [
            prof_input.to(torch.device(f"{device.type}:{rank}"))
            for prof_input in prof_inputs[rank]
        ]
    else:
        warmup_inputs_cuda = [
            warmup_input.to(torch.device(f"{device.type}:0"))
            for warmup_input in warmup_inputs[0]
        ]
        bench_inputs_cuda = [
            bench_input.to(torch.device(f"{device.type}:0"))
            for bench_input in bench_inputs[0]
        ]
        prof_inputs_cuda = [
            prof_input.to(torch.device(f"{device.type}:0"))
            for prof_input in prof_inputs[0]
        ]

    with (
        MultiProcessContext(rank, world_size, "nccl", None)
        if rank != -1
        else contextlib.nullcontext()
    ) as ctx:
        module = transform_module(
            module=module,
            device=device,
            inputs=warmup_inputs_cuda,  # pyre-ignore[6]
            sharder=sharder,
            sharding_type=sharding_type,
            compile_mode=compile_mode,
            world_size=world_size,
            batch_size=batch_size,
            # pyre-ignore[6]
            ctx=ctx,
            benchmark_unsharded_module=benchmark_unsharded_module,
        )

        if benchmark_unsharded_module:
            name = "unsharded" + compile_mode.name
        else:
            name = benchmark_type_name(compile_mode, sharding_type)

        res = benchmark(
            name,
            module,
            warmup_inputs_cuda,  # pyre-ignore[6]
            bench_inputs_cuda,  # pyre-ignore[6]
            prof_inputs_cuda,  # pyre-ignore[6]
            world_size=world_size,
            output_dir=output_dir,
            num_benchmarks=num_benchmarks,
            func_to_benchmark=func_to_benchmark,
            benchmark_func_kwargs=benchmark_func_kwargs,
            rank=rank,
            device_type=device.type,
            benchmark_unsharded_module=benchmark_unsharded_module,
        )

        if queue is not None:
            queue.put(res)

            while not queue.empty():
                time.sleep(1)

    return res


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

    for p in processes:
        p.join()
        assert 0 == p.exitcode

    total_benchmark_res = BenchmarkResult(
        short_name=benchmark_res_per_rank[0].short_name,
        gpu_elapsed_time=benchmark_res_per_rank[0].gpu_elapsed_time,
        cpu_elapsed_time=benchmark_res_per_rank[0].cpu_elapsed_time,
        gpu_mem_stats=[GPUMemoryStats(rank, 0, 0, 0) for rank in range(world_size)],
        cpu_mem_stats=CPUMemoryStats(0, 0),  # Will be updated below
        rank=0,
    )

    for res in benchmark_res_per_rank:
        # Each rank's BenchmarkResult contains 1 GPU memory measurement
        total_benchmark_res.gpu_mem_stats[res.rank] = res.gpu_mem_stats[0]
        # Use the CPU memory stats from the first rank (they should be similar across ranks)
        if res.rank == 0:
            total_benchmark_res.cpu_mem_stats = res.cpu_mem_stats

    return total_benchmark_res


def benchmark_module(
    module: torch.nn.Module,
    sharder: ModuleSharder[T],
    sharding_types: List[ShardingType],
    compile_modes: List[CompileMode],
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    warmup_iters: int = 20,
    bench_iters: int = 500,
    prof_iters: int = 20,
    num_float_features: int = 0,
    batch_size: int = 2048,
    world_size: int = 2,
    num_benchmarks: int = 5,
    output_dir: str = "",
    benchmark_unsharded: bool = False,
    func_to_benchmark: Callable[..., None] = default_func_to_benchmark,
    benchmark_func_kwargs: Optional[Dict[str, Any]] = None,
    pooling_configs: Optional[List[int]] = None,
    variable_batch_embeddings: bool = False,
    device_type: str = "cuda",
    train: bool = True,
) -> List[BenchmarkResult]:
    """
    Args:
        eager_module: Eager mode module to be benchmarked
        sharding_types: Sharding types to be benchmarked
        compile_modes: Compilation modes to be benchmarked
        warmup_iters: Number of iterations to run before profiling
        bench_iters: Number of iterations to run during profiling
        prof_iters: Number of iterations to run after profiling
        batch_size: Batch size used in the model
        world_size: World size used in the
        num_benchmarks: How many times to run over benchmark inputs for statistics
        output_dir: Directory to output profiler outputs (traces, stacks)
        pooling_configs: The pooling factor for the tables.
            (Optional; if not set, we'll use 10 as default)
        func_to_benchmark: Custom function to benchmark, check out default_func_to_benchmark for default
        benchmark_func_kwargs: Custom keyword arguments to pass to func_to_benchmark

    Returns:
        A list of BenchmarkResults
    """

    # logging.info(f"###### Benchmarking Module: {eager_module} ######\n")
    logging.info(f"Warmup iterations: {warmup_iters}")
    logging.info(f"Benchmark iterations: {bench_iters}")
    logging.info(f"Profile iterations: {prof_iters}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"World Size: {world_size}")
    logging.info(f"Number of Benchmarks: {num_benchmarks}")
    logging.info(f"Output Directory: {output_dir}")

    assert (
        num_benchmarks > 2
    ), "num_benchmarks needs to be greater than 2 for statistical analysis"

    benchmark_results: List[BenchmarkResult] = []

    wrapped_module = ModuleWrapper(module)

    num_inputs_to_gen: int = warmup_iters + bench_iters + prof_iters
    inputs = get_inputs(
        tables,
        batch_size,
        world_size,
        num_inputs_to_gen,
        num_float_features,
        train,
        pooling_configs,
        variable_batch_embeddings,
        only_kjt=True,
    )

    warmup_inputs = [rank_inputs[:warmup_iters] for rank_inputs in inputs]
    bench_inputs = [
        rank_inputs[warmup_iters : (warmup_iters + bench_iters)]
        for rank_inputs in inputs
    ]
    prof_inputs = [rank_inputs[-prof_iters:] for rank_inputs in inputs]

    for sharding_type in sharding_types if not benchmark_unsharded else ["Unsharded"]:
        for compile_mode in compile_modes:
            if not benchmark_unsharded:
                # Test sharders should have a singular sharding_type
                sharder._sharding_type = sharding_type.value
                # pyre-ignore [6]
                benchmark_type = benchmark_type_name(compile_mode, sharding_type)
            else:
                benchmark_type = "unsharded" + compile_mode.name

            logging.info(
                f"\n\n###### Running Benchmark Type: {benchmark_type} ######\n"
            )

            if train:
                res = multi_process_benchmark(
                    # pyre-ignore[6]
                    callable=init_module_and_run_benchmark,
                    module=wrapped_module,
                    sharder=sharder,
                    device=torch.device(device_type),
                    sharding_type=sharding_type,
                    compile_mode=compile_mode,
                    world_size=world_size,
                    batch_size=batch_size,
                    warmup_inputs=warmup_inputs,
                    bench_inputs=bench_inputs,
                    prof_inputs=prof_inputs,
                    tables=tables,
                    num_benchmarks=num_benchmarks,
                    output_dir=output_dir,
                    func_to_benchmark=func_to_benchmark,
                    benchmark_func_kwargs=benchmark_func_kwargs,
                    pooling_configs=pooling_configs,
                )
            else:
                res = init_module_and_run_benchmark(
                    module=wrapped_module,
                    sharder=sharder,
                    device=torch.device(device_type),
                    # pyre-ignore
                    sharding_type=sharding_type,
                    compile_mode=compile_mode,
                    world_size=world_size,
                    batch_size=batch_size,
                    warmup_inputs=warmup_inputs,  # pyre-ignore[6]
                    bench_inputs=bench_inputs,  # pyre-ignore[6]
                    prof_inputs=prof_inputs,  # pyre-ignore[6]
                    tables=tables,
                    num_benchmarks=num_benchmarks,
                    output_dir=output_dir,
                    func_to_benchmark=func_to_benchmark,
                    benchmark_func_kwargs=benchmark_func_kwargs,
                    pooling_configs=pooling_configs,
                    benchmark_unsharded_module=benchmark_unsharded,
                )

            gc.collect()

            benchmark_results.append(res)

    return benchmark_results
