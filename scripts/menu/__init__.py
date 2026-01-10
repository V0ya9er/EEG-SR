#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SR-EEG 菜单系统模块

提供重构后的交互式菜单系统组件。
"""

from .config import (
    ExperimentConfig,
    CONFIGURABLE_PARAMS,
    MODELS, DATASETS, DATASETS_LOSO, MECHANISMS, NOISES,
    MODEL_NAMES, DATASET_NAMES, DATASET_LOSO_NAMES,
    MECHANISM_NAMES, NOISE_NAMES,
    validate_param_value
)
from .ui import (
    Table, Panel, Spinner, ConfigCard, ProgressBar,
    SelectList, Menu, MenuItem,
    print_header, print_menu_header, print_separator,
    print_success, print_error, print_warning, print_info,
    clear_screen, wait_for_enter, get_input, get_choice,
    confirm, get_validated_input
)
from .gpu import GPUManager, GPUInfo, get_gpu_manager
from .help import show_help, show_help_menu, HELP_TEXTS
from .experiments import (
    run_single_experiment, run_loso_experiment, run_batch_experiments,
    run_analysis_only, run_visualization_only, select_checkpoint,
    find_latest_checkpoint, find_latest_version_dir
)

__all__ = [
    # Config
    'ExperimentConfig',
    'CONFIGURABLE_PARAMS',
    'MODELS', 'DATASETS', 'DATASETS_LOSO', 'MECHANISMS', 'NOISES',
    'MODEL_NAMES', 'DATASET_NAMES', 'DATASET_LOSO_NAMES',
    'MECHANISM_NAMES', 'NOISE_NAMES',
    'validate_param_value',
    
    # UI
    'Table', 'Panel', 'Spinner', 'ConfigCard', 'ProgressBar',
    'SelectList', 'Menu', 'MenuItem',
    'print_header', 'print_menu_header', 'print_separator',
    'print_success', 'print_error', 'print_warning', 'print_info',
    'clear_screen', 'wait_for_enter', 'get_input', 'get_choice',
    'confirm', 'get_validated_input',
    
    # GPU
    'GPUManager', 'GPUInfo', 'get_gpu_manager',
    
    # Help
    'show_help', 'show_help_menu', 'HELP_TEXTS',
    
    # Experiments
    'run_single_experiment', 'run_loso_experiment', 'run_batch_experiments',
    'run_analysis_only', 'run_visualization_only', 'select_checkpoint',
    'find_latest_checkpoint', 'find_latest_version_dir',
]