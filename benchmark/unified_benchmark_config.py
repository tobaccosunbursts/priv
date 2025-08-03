#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Unified configuration system for both module and pipeline benchmarking.

This module provides a unified YAML configuration interface that can handle:
1. Module-level benchmarking with dynamic module importing
2. Pipeline-level benchmarking with existing pipeline configurations
3. Common configuration shared between both approaches
"""

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.types import ShardingType

from .benchmark_pipeline_utils import BaseModelConfig
from .benchmark_train_pipeline import (
    EmbeddingTablesConfig,
    PipelineConfig,
    RunOptions,
)


@dataclass
class ModuleConfig:
    """
    Configuration for dynamic module importing and instantiation.
    
    This allows specifying any PyTorch module via YAML configuration by providing
    the module path, class name, and constructor arguments.
    
    Args:
        module_path: Python import path to the module (e.g., "torchrec.models.deepfm")
        class_name: Name of the class to instantiate (e.g., "SimpleDeepFMNNWrapper")
        constructor_args: Dictionary of arguments to pass to the class constructor
        requires_tables: Whether this module requires embedding tables for construction
        requires_weighted_tables: Whether this module requires weighted embedding tables
        requires_dense_device: Whether this module requires a dense_device parameter
    """
    module_path: str
    class_name: str
    constructor_args: Dict[str, Any] = field(default_factory=dict)
    requires_tables: bool = True
    requires_weighted_tables: bool = False
    requires_dense_device: bool = False


@dataclass
class UnifiedBenchmarkConfig:
    """
    Unified configuration that can handle both module and pipeline benchmarking.
    
    Args:
        benchmark_type: Type of benchmark to run ("module" or "pipeline")
        run_options: Common run options for both benchmark types
        table_config: Configuration for embedding tables
        module_config: Configuration for module benchmarking (only used when benchmark_type="module")
        pipeline_config: Configuration for pipeline benchmarking (only used when benchmark_type="pipeline")
        model_selection: Model selection config for pipeline benchmarking (only used when benchmark_type="pipeline")
    """
    benchmark_type: str  # "module" or "pipeline"
    run_options: RunOptions
    table_config: EmbeddingTablesConfig
    module_config: Optional[ModuleConfig] = None
    pipeline_config: Optional[PipelineConfig] = None
    model_selection: Optional[Dict[str, Any]] = None


def import_module_class(module_path: str, class_name: str) -> Type[nn.Module]:
    """
    Dynamically import a class from a module path.
    
    Args:
        module_path: Python import path to the module (e.g., "torchrec.models.deepfm")
        class_name: Name of the class to import (e.g., "SimpleDeepFMNNWrapper")
        
    Returns:
        The imported class
        
    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the class doesn't exist in the module
    """
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        
        if not inspect.isclass(cls):
            raise ValueError(f"{class_name} is not a class in {module_path}")
            
        if not issubclass(cls, nn.Module):
            raise ValueError(f"{class_name} is not a subclass of torch.nn.Module")
            
        return cls
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class {class_name} not found in {module_path}: {e}")


def create_module_from_config(
    module_config: ModuleConfig,
    tables: Optional[List[Any]] = None,
    weighted_tables: Optional[List[Any]] = None,
    dense_device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Create a module instance from a ModuleConfig.
    
    Args:
        module_config: Configuration specifying the module to create
        tables: Embedding tables to pass to the constructor (if required)
        weighted_tables: Weighted embedding tables to pass to the constructor (if required)
        dense_device: Dense device to pass to the constructor (if required)
        
    Returns:
        An instantiated PyTorch module
    """
    cls = import_module_class(module_config.module_path, module_config.class_name)
    
    # Build constructor arguments
    constructor_args = module_config.constructor_args.copy()
    
    # Add tables if required
    if module_config.requires_tables and tables is not None:
        constructor_args["tables"] = tables
        
    if module_config.requires_weighted_tables and weighted_tables is not None:
        constructor_args["weighted_tables"] = weighted_tables
        
    if module_config.requires_dense_device and dense_device is not None:
        constructor_args["dense_device"] = dense_device
    
    try:
        return cls(**constructor_args)
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {module_config.class_name} from {module_config.module_path} "
            f"with args {constructor_args}: {e}"
        )


def validate_config(config: UnifiedBenchmarkConfig) -> None:
    """
    Validate that the unified benchmark configuration is properly structured.
    
    Args:
        config: The configuration to validate
        
    Raises:
        ValueError: If the configuration is invalid
    """
    if config.benchmark_type not in ["module", "pipeline"]:
        raise ValueError(f"benchmark_type must be 'module' or 'pipeline', got '{config.benchmark_type}'")
    
    if config.benchmark_type == "module":
        if config.module_config is None:
            raise ValueError("module_config is required when benchmark_type='module'")
        if not config.module_config.module_path:
            raise ValueError("module_config.module_path cannot be empty")
        if not config.module_config.class_name:
            raise ValueError("module_config.class_name cannot be empty")
    
    elif config.benchmark_type == "pipeline":
        if config.pipeline_config is None:
            raise ValueError("pipeline_config is required when benchmark_type='pipeline'")
        if config.model_selection is None:
            raise ValueError("model_selection is required when benchmark_type='pipeline'")


# Example YAML configurations for reference:

EXAMPLE_MODULE_CONFIG_YAML = """
# Example YAML configuration for module benchmarking
benchmark_type: "module"

run_options:
  world_size: 2
  num_batches: 10
  sharding_type: "TABLE_WISE"
  compute_kernel: "FUSED"
  planner_type: "embedding"

table_config:
  num_unweighted_features: 26
  num_weighted_features: 0
  embedding_feature_dim: 128

module_config:
  module_path: "torchrec.models.deepfm"
  class_name: "SimpleDeepFMNNWrapper"
  requires_tables: true
  requires_dense_device: false
  constructor_args:
    num_dense_features: 13
    hidden_layer_size: 512
    deep_fm_dimension: 16
"""

EXAMPLE_PIPELINE_CONFIG_YAML = """
# Example YAML configuration for pipeline benchmarking
benchmark_type: "pipeline"

run_options:
  world_size: 2
  num_batches: 10
  sharding_type: "TABLE_WISE"
  compute_kernel: "FUSED"
  planner_type: "embedding"

table_config:
  num_unweighted_features: 100
  num_weighted_features: 100
  embedding_feature_dim: 128

pipeline_config:
  pipeline: "sparse"
  emb_lookup_stream: "data_dist"
  apply_jit: false

model_selection:
  model_name: "test_sparse_nn"
  batch_size: 8192
  num_float_features: 10
  feature_pooling_avg: 10
"""