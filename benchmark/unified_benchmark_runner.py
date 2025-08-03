#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Unified benchmark runner that can execute both module and pipeline benchmarks
based on YAML configuration.

Example usage:

Module benchmarking:
    python -m torchrec.distributed.benchmark.unified_benchmark_runner --config module_config.yaml

Pipeline benchmarking:
    python -m torchrec.distributed.benchmark.unified_benchmark_runner --config pipeline_config.yaml

The runner automatically detects the benchmark type from the YAML configuration
and executes the appropriate benchmark.
"""

import argparse
import logging
import sys
from typing import List

import yaml
from torchrec.distributed.benchmark.benchmark_pipeline_utils import create_model_config
from torchrec.distributed.benchmark.benchmark_train_pipeline import (
    ModelSelectionConfig,
    run_pipeline,
)
from torchrec.distributed.benchmark.benchmark_utils import (
    benchmark_module,
    BenchmarkResult,
    generate_tables,
)
from torchrec.distributed.benchmark.unified_benchmark_config import (
    create_module_from_config,
    UnifiedBenchmarkConfig,
    validate_config,
)

logger = logging.getLogger(__name__)


def load_config_from_yaml(config_path: str) -> UnifiedBenchmarkConfig:
    """
    Load and parse a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Parsed UnifiedBenchmarkConfig
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML is invalid
        ValueError: If the configuration is invalid
    """
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        if not config_dict:
            raise ValueError("Configuration file is empty or invalid")
            
        # Convert dictionary to dataclass instance
        config = dict_to_unified_config(config_dict)
        validate_config(config)
        
        return config
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in {config_path}: {e}")


def dict_to_unified_config(config_dict: dict) -> UnifiedBenchmarkConfig:
    """
    Convert a dictionary loaded from YAML to UnifiedBenchmarkConfig dataclass.
    
    Args:
        config_dict: Dictionary containing configuration data
        
    Returns:
        UnifiedBenchmarkConfig instance
    """
    from torchrec.distributed.benchmark.benchmark_train_pipeline import (
        EmbeddingTablesConfig,
        PipelineConfig,
        RunOptions,
    )
    from torchrec.distributed.benchmark.unified_benchmark_config import ModuleConfig

    # Parse run_options
    run_options_dict = config_dict.get("run_options", {})
    run_options = RunOptions(**run_options_dict)

    # Parse table_config
    table_config_dict = config_dict.get("table_config", {})
    table_config = EmbeddingTablesConfig(**table_config_dict)

    # Parse optional configs based on benchmark type
    module_config = None
    pipeline_config = None
    model_selection = None

    benchmark_type = config_dict.get("benchmark_type", "")
    
    if benchmark_type == "module":
        module_config_dict = config_dict.get("module_config", {})
        if module_config_dict:
            module_config = ModuleConfig(**module_config_dict)
    
    elif benchmark_type == "pipeline":
        pipeline_config_dict = config_dict.get("pipeline_config", {})
        if pipeline_config_dict:
            pipeline_config = PipelineConfig(**pipeline_config_dict)
            
        model_selection = config_dict.get("model_selection", {})

    return UnifiedBenchmarkConfig(
        benchmark_type=benchmark_type,
        run_options=run_options,
        table_config=table_config,
        module_config=module_config,
        pipeline_config=pipeline_config,
        model_selection=model_selection,
    )


def run_module_benchmark(config: UnifiedBenchmarkConfig) -> BenchmarkResult:
    """
    Run a module-level benchmark based on the configuration.
    
    Args:
        config: Unified benchmark configuration
        
    Returns:
        BenchmarkResult containing timing and memory statistics
    """
    assert config.module_config is not None, "module_config is required for module benchmarking"
    
    logger.info("Running module benchmark...")
    logger.info(f"Module: {config.module_config.module_path}.{config.module_config.class_name}")
    logger.info(f"World size: {config.run_options.world_size}")
    
    # Generate embedding tables
    tables, weighted_tables = generate_tables(
        num_unweighted_features=config.table_config.num_unweighted_features,
        num_weighted_features=config.table_config.num_weighted_features,
        embedding_feature_dim=config.table_config.embedding_feature_dim,
    )
    
    # Create the module
    module = create_module_from_config(
        module_config=config.module_config,
        tables=tables,
        weighted_tables=weighted_tables,
        dense_device=None,  # Will be set during sharding
    )
    
    # Run the benchmark
    result = benchmark_module(
        module=module,
        tables=tables,
        weighted_tables=weighted_tables if config.module_config.requires_weighted_tables else None,
        sharding_type=config.run_options.sharding_type,
        planner_type=config.run_options.planner_type,
        world_size=config.run_options.world_size,
        num_benchmarks=config.run_options.num_batches,
        compute_kernel=config.run_options.compute_kernel,
    )
    
    return result


def run_pipeline_benchmark(config: UnifiedBenchmarkConfig) -> List[BenchmarkResult]:
    """
    Run a pipeline-level benchmark based on the configuration.
    
    Args:
        config: Unified benchmark configuration
        
    Returns:
        List of BenchmarkResult containing timing and memory statistics
    """
    assert config.pipeline_config is not None, "pipeline_config is required for pipeline benchmarking"
    assert config.model_selection is not None, "model_selection is required for pipeline benchmarking"
    
    logger.info("Running pipeline benchmark...")
    logger.info(f"Pipeline type: {config.pipeline_config.pipeline}")
    logger.info(f"Model: {config.model_selection.get('model_name', 'unknown')}")
    logger.info(f"World size: {config.run_options.world_size}")
    
    # Create model config from the model_selection dictionary
    model_config = create_model_config(**config.model_selection)
    
    # Run the pipeline benchmark
    results = run_pipeline(
        run_option=config.run_options,
        table_config=config.table_config,
        pipeline_config=config.pipeline_config,
        model_config=model_config,
    )
    
    return results


def main() -> None:
    """Main entry point for the unified benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for both module and pipeline benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Module benchmarking
  python -m torchrec.distributed.benchmark.unified_benchmark_runner --config deepfm_module.yaml
  
  # Pipeline benchmarking  
  python -m torchrec.distributed.benchmark.unified_benchmark_runner --config sparse_pipeline.yaml
  
  # With custom log level
  python -m torchrec.distributed.benchmark.unified_benchmark_runner --config config.yaml --loglevel debug
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the configuration without running benchmarks",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config_from_yaml(args.config)
        logger.info(f"Configuration loaded successfully. Benchmark type: {config.benchmark_type}")
        
        if args.validate_only:
            logger.info("Configuration validation passed. Exiting.")
            return
        
        # Run the appropriate benchmark
        if config.benchmark_type == "module":
            result = run_module_benchmark(config)
            print("\n" + "="*80)
            print("MODULE BENCHMARK RESULTS")
            print("="*80)
            print(result)
            
        elif config.benchmark_type == "pipeline":
            results = run_pipeline_benchmark(config)
            print("\n" + "="*80)
            print("PIPELINE BENCHMARK RESULTS")
            print("="*80)
            for result in results:
                print(result)
                
        else:
            raise ValueError(f"Unknown benchmark type: {config.benchmark_type}")
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()