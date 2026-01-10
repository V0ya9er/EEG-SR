#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SR-EEG 实验运行助手 - 统一交互式菜单

重构版本：实现新的层级菜单结构，改善用户体验。
"""

import os
import sys
import platform
from pathlib import Path
from typing import List, Optional

# 确保可以导入本地模块
sys.path.insert(0, str(Path(__file__).parent))

from menu.config import (
    ExperimentConfig, CONFIGURABLE_PARAMS,
    MODELS, DATASETS, DATASETS_LOSO, MECHANISMS, NOISES,
    MODEL_NAMES, DATASET_NAMES, DATASET_LOSO_NAMES, 
    MECHANISM_NAMES, NOISE_NAMES,
    validate_param_value
)
from menu.gpu import GPUManager, get_gpu_manager
from menu.ui import (
    clear_screen, print_header, print_menu_header, print_separator,
    print_success, print_error, print_warning, print_info,
    wait_for_enter, get_input, get_choice, confirm,
    get_validated_input, Table, ConfigCard, SelectList
)
from menu.help import show_help_menu
from menu.experiments import (
    run_single_experiment, run_loso_experiment, run_batch_experiments,
    run_analysis_only, run_visualization_only, select_checkpoint,
    find_latest_checkpoint, find_latest_loso_output_dir,
    find_latest_experiment_results, find_checkpoint_in_dir
)


# ============================================================================
# 全局配置
# ============================================================================

config: ExperimentConfig = None


def load_global_config():
    """加载全局配置"""
    global config
    config = ExperimentConfig.load()


def save_global_config():
    """保存全局配置"""
    global config
    if config:
        config.save()


# ============================================================================
# 快速开始
# ============================================================================

def menu_quick_start():
    """快速开始 - 默认使用 LOSO 单折测试"""
    print_menu_header("快速开始 (LOSO)", "🚀")
    
    # 计算每折留出被试数
    subjects_per_fold = 9 // config.loso_n_folds
    remainder = 9 % config.loso_n_folds
    subjects_info = f"{subjects_per_fold}" + (f"-{subjects_per_fold+1}" if remainder > 0 else "")
    
    # 显示 LOSO 配置摘要
    print("当前 LOSO 配置摘要:\n")
    print(f"  LOSO 数据集:   {DATASET_LOSO_NAMES.get(config.loso_dataset, config.loso_dataset)}")
    print(f"  折数:          {config.loso_n_folds}")
    print(f"  每折留出被试:  {subjects_info} 个")
    print(f"  当前折:        第 {config.loso_fold_id} 折")
    print()
    print(f"  模型:          {config.get_model_display_name()}")
    print(f"  SR 机制:       {config.get_mechanism_display_name()}")
    print(f"  噪声类型:      {config.get_noise_display_name()}")
    print(f"  Epochs:        {config.epochs}")
    print(f"  GPU:           {'CPU' if config.use_cpu else f'#{config.gpu_id}'}")
    print()
    
    print_separator("─")
    print()
    
    print("  [Enter] 开始 LOSO 单折训练")
    print("  [A]     运行全部折 (完整 LOSO)")
    print("  [C]     进入配置修改")
    print("  [L]     传统训练 (非 LOSO)")
    print("  [0]     返回主菜单")
    print()
    
    choice = get_input("请选择: ", "").lower()
    
    if choice == "c":
        menu_config()
    elif choice == "0":
        return
    elif choice == "a":
        # 运行全部折
        if confirm(f"确认运行全部 {config.loso_n_folds} 折? 这将执行 {config.loso_n_folds} 次训练。[Y/n]: "):
            # 临时更新 dataset
            original_dataset = config.dataset
            config.dataset = config.loso_dataset
            run_loso_experiment(config, run_all_folds=True, n_folds=config.loso_n_folds)
            config.dataset = original_dataset
            wait_for_enter()
    elif choice == "l":
        # 传统训练
        if confirm("确认使用传统训练 (固定划分)? [Y/n]: "):
            run_single_experiment(config)
            wait_for_enter()
    else:
        # 开始 LOSO 单折训练
        if confirm(f"确认开始 LOSO 第 {config.loso_fold_id} 折训练? [Y/n]: "):
            # 临时更新 dataset
            original_dataset = config.dataset
            config.dataset = config.loso_dataset
            run_loso_experiment(config, fold_id=config.loso_fold_id, n_folds=config.loso_n_folds)
            config.dataset = original_dataset
            wait_for_enter()


# ============================================================================
# 实验配置菜单
# ============================================================================

def menu_config():
    """实验配置菜单"""
    while True:
        print_menu_header("实验配置", "⚙️")
        
        print("  [1] 查看当前配置")
        print("  [2] 模型与数据")
        print("  [3] 随机共振设置")
        print("  [4] 训练参数")
        print("  [5] GPU 设置")
        print("  [6] LOSO 设置 (折数)")
        print()
        print("  [R] 重置为默认值")
        print("  [S] 保存配置")
        print("  [0] ← 返回主菜单")
        print()
        
        choice = get_choice("请选择: ",
                           ["0", "1", "2", "3", "4", "5", "6", "r", "s", "R", "S"])
        
        if choice == "0":
            save_global_config()
            return
        elif choice == "1":
            menu_view_config()
        elif choice == "2":
            menu_model_dataset()
        elif choice == "3":
            menu_sr_settings()
        elif choice == "4":
            menu_training_params()
        elif choice == "5":
            menu_gpu_settings()
        elif choice == "6":
            menu_loso_settings()
        elif choice.lower() == "r":
            if confirm("确认重置所有配置为默认值? [y/N]: ", default=False):
                config.reset_to_defaults()
                print_success("已重置为默认配置")
                wait_for_enter()
        elif choice.lower() == "s":
            if config.save():
                print_success("配置已保存")
            else:
                print_error("保存配置失败")
            wait_for_enter()


def menu_view_config():
    """查看当前配置"""
    print_menu_header("当前配置", "📋")
    
    print("  模型与数据:")
    print(f"    模型:     {config.get_model_display_name()}")
    print(f"    数据集:   {config.get_dataset_display_name()}")
    print()
    print("  随机共振:")
    print(f"    机制:     {config.get_mechanism_display_name()}")
    print(f"    噪声类型: {config.get_noise_display_name()}")
    print()
    print("  训练参数:")
    print(f"    Epochs:            {config.epochs}")
    print(f"    Batch Size:        {config.batch_size}")
    print(f"    Learning Rate:     {config.learning_rate}")
    print(f"    Early Stopping:    {config.early_stopping_patience}")
    print()
    
    # LOSO 设置
    subjects_per_fold = 9 // config.loso_n_folds
    remainder = 9 % config.loso_n_folds
    subjects_info = f"{subjects_per_fold}" + (f"-{subjects_per_fold+1}" if remainder > 0 else "")
    
    print("  LOSO 设置:")
    print(f"    LOSO 数据集:   {DATASET_LOSO_NAMES.get(config.loso_dataset, config.loso_dataset)}")
    print(f"    折数 (n_folds): {config.loso_n_folds}")
    print(f"    每折留出被试:  {subjects_info} 个")
    print(f"    当前折:        第 {config.loso_fold_id} 折")
    print(f"    运行全部折:    {'是' if config.loso_run_all else '否'}")
    print()
    
    print("  硬件:")
    if config.use_cpu:
        print(f"    设备:     CPU 模式")
    else:
        gpu_mgr = get_gpu_manager()
        gpu = gpu_mgr.get_gpu_by_id(config.gpu_id)
        if gpu:
            print(f"    GPU:      #{config.gpu_id} ({gpu.name})")
        else:
            print(f"    GPU:      #{config.gpu_id}")
    print()
    
    wait_for_enter()


def menu_model_dataset():
    """模型与数据集配置"""
    print_menu_header("模型与数据", "🧠")
    
    # 选择模型
    print("选择模型:")
    for i, m in enumerate(MODELS, 1):
        marker = "→" if m == config.model else " "
        print(f"  {marker}[{i}] {MODEL_NAMES[m]}")
    print()
    
    choice = get_input(f"请选择 [1-{len(MODELS)}] (回车保持当前): ", "")
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MODELS):
                config.model = MODELS[idx]
                print_success(f"模型已设为: {config.get_model_display_name()}")
        except ValueError:
            print_error("无效输入")
    
    print()
    
    # 选择数据集
    print("选择数据集:")
    for i, d in enumerate(DATASETS, 1):
        marker = "→" if d == config.dataset else " "
        print(f"  {marker}[{i}] {DATASET_NAMES[d]}")
    print()
    
    choice = get_input(f"请选择 [1-{len(DATASETS)}] (回车保持当前): ", "")
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(DATASETS):
                config.dataset = DATASETS[idx]
                print_success(f"数据集已设为: {config.get_dataset_display_name()}")
        except ValueError:
            print_error("无效输入")
    
    print()
    save_global_config()
    wait_for_enter()


def menu_sr_settings():
    """随机共振设置"""
    print_menu_header("随机共振设置", "🌊")
    
    # 选择机制
    print("选择 SR 机制:")
    for i, m in enumerate(MECHANISMS, 1):
        marker = "→" if m == config.mechanism else " "
        print(f"  {marker}[{i}] {MECHANISM_NAMES[m]}")
    print()
    
    choice = get_input(f"请选择 [1-{len(MECHANISMS)}] (回车保持当前): ", "")
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MECHANISMS):
                config.mechanism = MECHANISMS[idx]
                print_success(f"机制已设为: {config.get_mechanism_display_name()}")
        except ValueError:
            print_error("无效输入")
    
    print()
    
    # 选择噪声类型
    print("选择噪声类型:")
    for i, n in enumerate(NOISES, 1):
        marker = "→" if n == config.noise_type else " "
        print(f"  {marker}[{i}] {NOISE_NAMES[n]}")
    print()
    
    choice = get_input(f"请选择 [1-{len(NOISES)}] (回车保持当前): ", "")
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(NOISES):
                config.noise_type = NOISES[idx]
                print_success(f"噪声类型已设为: {config.get_noise_display_name()}")
        except ValueError:
            print_error("无效输入")
    
    print()
    save_global_config()
    wait_for_enter()


def menu_training_params():
    """训练参数配置"""
    while True:
        print_menu_header("训练参数", "📊")
        
        print("  序号   参数名               当前值        说明")
        print("  " + "─" * 56)
        print(f"  [1]    Epochs               {config.epochs:<12} 最大训练轮数")
        print(f"  [2]    Batch Size           {config.batch_size:<12} 每批样本数")
        print(f"  [3]    Learning Rate        {config.learning_rate:<12} 初始学习率")
        print(f"  [4]    Early Stopping       {config.early_stopping_patience:<12} 提前停止耐心值")
        print()
        print("  [R]    重置为默认值")
        print("  [0]    ← 返回")
        print()
        
        choice = get_choice("请选择要修改的参数: ", 
                           ["0", "1", "2", "3", "4", "r", "R"])
        
        if choice == "0":
            save_global_config()
            return
        elif choice == "1":
            _update_param("epochs", "Epochs", config.epochs)
        elif choice == "2":
            _update_batch_size()
        elif choice == "3":
            _update_param("learning_rate", "Learning Rate", config.learning_rate)
        elif choice == "4":
            _update_param("early_stopping_patience", "Early Stopping", 
                         config.early_stopping_patience)
        elif choice.lower() == "r":
            config.epochs = 50
            config.batch_size = 32
            config.learning_rate = 0.001
            config.early_stopping_patience = 10
            print_success("训练参数已重置为默认值")
            wait_for_enter()


def _update_param(param_name: str, display_name: str, current_value):
    """更新单个参数"""
    new_value = get_input(f"请输入新的 {display_name} [当前: {current_value}]: ", "")
    if not new_value:
        return
    
    valid, result = validate_param_value(param_name, new_value)
    if valid:
        setattr(config, param_name, result)
        print_success(f"{display_name} 已更新为 {result}")
    else:
        print_error(result)
    
    wait_for_enter()


def _update_batch_size():
    """更新 Batch Size（从选项中选择）"""
    choices = [8, 16, 32, 64, 128, 256]
    
    print("\n可选的 Batch Size:")
    for i, size in enumerate(choices, 1):
        marker = "→" if size == config.batch_size else " "
        print(f"  {marker}[{i}] {size}")
    print()
    
    choice = get_input("请选择 [1-6] (回车保持当前): ", "")
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(choices):
                config.batch_size = choices[idx]
                print_success(f"Batch Size 已更新为 {config.batch_size}")
        except ValueError:
            print_error("无效输入")
    
    wait_for_enter()


def menu_loso_settings():
    """LOSO 设置菜单"""
    while True:
        print_menu_header("LOSO 设置", "🔄")
        
        # 计算每折留出被试数
        subjects_per_fold = 9 // config.loso_n_folds
        remainder = 9 % config.loso_n_folds
        subjects_info = f"{subjects_per_fold}" + (f"-{subjects_per_fold+1}" if remainder > 0 else "")
        
        print("  当前 LOSO 配置:")
        print(f"    LOSO 数据集:   {DATASET_LOSO_NAMES.get(config.loso_dataset, config.loso_dataset)}")
        print(f"    折数 (n_folds): {config.loso_n_folds}")
        print(f"    每折留出被试:  {subjects_info} 个")
        print(f"    当前折:        第 {config.loso_fold_id} 折")
        print(f"    运行全部折:    {'是' if config.loso_run_all else '否'}")
        print()
        print_separator("─")
        print()
        print("  [1] 修改 LOSO 数据集")
        print("  [2] 修改折数 (n_folds)")
        print("  [3] 修改当前折 (fold_id)")
        print("  [4] 切换运行模式 (单折/全部)")
        print()
        print("  [0] ← 返回")
        print()
        
        choice = get_choice("请选择: ", ["0", "1", "2", "3", "4"])
        
        if choice == "0":
            save_global_config()
            return
        elif choice == "1":
            _update_loso_dataset()
        elif choice == "2":
            _update_loso_n_folds()
        elif choice == "3":
            _update_loso_fold_id()
        elif choice == "4":
            _toggle_loso_run_all()


def _update_loso_dataset():
    """更新 LOSO 数据集"""
    print("\n选择 LOSO 数据集:")
    for i, d in enumerate(DATASETS_LOSO, 1):
        marker = "→" if d == config.loso_dataset else " "
        print(f"  {marker}[{i}] {DATASET_LOSO_NAMES[d]}")
    print()
    
    choice = get_input(f"请选择 [1-{len(DATASETS_LOSO)}] (回车保持当前): ", "")
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(DATASETS_LOSO):
                config.loso_dataset = DATASETS_LOSO[idx]
                print_success(f"LOSO 数据集已设为: {DATASET_LOSO_NAMES[config.loso_dataset]}")
        except ValueError:
            print_error("无效输入")
    wait_for_enter()


def _update_loso_n_folds():
    """更新 LOSO 折数"""
    print(f"\n当前折数: {config.loso_n_folds}")
    print()
    print("折数说明:")
    print("  n_folds=1: 不使用交叉验证 (留出被试 1-9)")
    print("  n_folds=3: 每折留出 3 个被试")
    print("  n_folds=9: 每折留出 1 个被试 (经典 LOSO)")
    print()
    
    choice = get_input("请输入折数 [1-9] (回车保持当前): ", "")
    if choice:
        try:
            n_folds = int(choice)
            if 1 <= n_folds <= 9:
                config.loso_n_folds = n_folds
                # 确保 fold_id 不超过 n_folds
                if config.loso_fold_id > n_folds:
                    config.loso_fold_id = 1
                
                # 显示分组方案
                print()
                print(f"分组方案 ({config.loso_n_folds} 折):")
                subjects = list(range(1, 10))
                subjects_per_fold = 9 // config.loso_n_folds
                remainder = 9 % config.loso_n_folds
                idx = 0
                for fold in range(1, config.loso_n_folds + 1):
                    count = subjects_per_fold + (1 if fold <= remainder else 0)
                    fold_subjects = subjects[idx:idx + count]
                    print(f"  Fold {fold}: 测试被试 {fold_subjects}")
                    idx += count
                
                print_success(f"折数已设为: {config.loso_n_folds}")
            else:
                print_error("折数必须在 1-9 之间")
        except ValueError:
            print_error("无效输入")
    wait_for_enter()


def _update_loso_fold_id():
    """更新当前运行的折"""
    print(f"\n当前折: {config.loso_fold_id} / {config.loso_n_folds}")
    print()
    
    choice = get_input(f"请输入折 ID [1-{config.loso_n_folds}] (回车保持当前): ", "")
    if choice:
        try:
            fold_id = int(choice)
            if 1 <= fold_id <= config.loso_n_folds:
                config.loso_fold_id = fold_id
                print_success(f"当前折已设为: {config.loso_fold_id}")
            else:
                print_error(f"折 ID 必须在 1-{config.loso_n_folds} 之间")
        except ValueError:
            print_error("无效输入")
    wait_for_enter()


def _toggle_loso_run_all():
    """切换运行模式"""
    if config.loso_run_all:
        config.loso_run_all = False
        print_success(f"已切换为运行单折 (第 {config.loso_fold_id} 折)")
    else:
        config.loso_run_all = True
        print_success(f"已切换为运行全部 {config.loso_n_folds} 折")
    wait_for_enter()


def menu_gpu_settings():
    """GPU 设置"""
    print_menu_header("GPU 设置", "🖥️")
    
    gpu_mgr = get_gpu_manager()
    gpus = gpu_mgr.get_gpu_list(refresh=True)
    
    print("可用设备:\n")
    
    if gpus:
        print(gpu_mgr.format_gpu_table(
            highlight_id=None if config.use_cpu else config.gpu_id
        ))
    else:
        print("  未检测到 NVIDIA GPU")
        print("  [C]  CPU 模式")
    
    print()
    
    if config.use_cpu:
        print(f"  当前选择: CPU 模式")
    else:
        gpu = gpu_mgr.get_gpu_by_id(config.gpu_id)
        if gpu:
            print(f"  当前选择: GPU {config.gpu_id} ({gpu.name})")
        else:
            print(f"  当前选择: GPU {config.gpu_id}")
    
    print()
    
    # 构建有效选项
    valid_choices = ["c", "C"]
    if gpus:
        valid_choices.extend([str(g.id) for g in gpus])
    
    choice = get_input("请选择设备 (直接回车保持当前): ", "")
    
    if not choice:
        return
    
    if choice.lower() == "c":
        config.use_cpu = True
        print_success("已切换到 CPU 模式")
    else:
        try:
            gpu_id = int(choice)
            if gpu_mgr.validate_gpu_id(gpu_id):
                config.gpu_id = gpu_id
                config.use_cpu = False
                print_success(f"已选择 GPU {gpu_id}")
            else:
                print_error(f"无效的 GPU ID: {gpu_id}")
        except ValueError:
            print_error("无效输入")
    
    save_global_config()
    wait_for_enter()


# ============================================================================
# 运行实验菜单
# ============================================================================

def menu_run_experiments():
    """运行实验菜单 - LOSO 优先"""
    while True:
        print_menu_header("运行实验", "🔬")
        
        print("  [1] 📌 标准 LOSO 实验 (推荐)")
        print("      Leave-One-Subject-Out: 评估跨被试泛化能力")
        print()
        print("  [2] 📦 批量 LOSO 实验")
        print("      遍历参数组合，每个组合运行完整 LOSO")
        print()
        print("  [3] 📁 传统训练 (Legacy)")
        print("      使用固定训练/测试集划分")
        print()
        print("  [0] ← 返回主菜单")
        print()
        
        choice = get_choice("请选择: ", ["0", "1", "2", "3"])
        
        if choice == "0":
            return
        elif choice == "1":
            menu_loso_training()
        elif choice == "2":
            menu_batch_loso_training()
        elif choice == "3":
            menu_standard_training()


def menu_standard_training():
    """标准训练"""
    print_menu_header("标准训练", "🎯")
    
    # 显示当前配置
    print("当前配置:")
    print(f"  模型:     {config.get_model_display_name()}")
    print(f"  数据集:   {config.get_dataset_display_name()}")
    print(f"  SR 机制:  {config.get_mechanism_display_name()}")
    print(f"  噪声类型: {config.get_noise_display_name()}")
    print(f"  Epochs:   {config.epochs}")
    print(f"  GPU:      {'CPU' if config.use_cpu else f'#{config.gpu_id}'}")
    print()
    
    if not confirm("确认开始训练? [Y/n]: "):
        return
    
    run_single_experiment(config)
    wait_for_enter()


def menu_loso_training():
    """LOSO 交叉验证"""
    print_menu_header("LOSO 交叉验证", "🔄")
    
    # 显示说明
    print("什么是 LOSO？")
    print("  Leave-One-Subject-Out（留一被试法）是一种交叉验证策略。")
    print("  数据集中有 9 个被试，可以按不同方式分组进行交叉验证。")
    print()
    
    # 计算每折留出被试数
    subjects_per_fold = 9 // config.loso_n_folds
    remainder = 9 % config.loso_n_folds
    subjects_info = f"{subjects_per_fold}" + (f"-{subjects_per_fold+1}" if remainder > 0 else "")
    
    # 显示当前 LOSO 配置
    print_separator("═")
    print("当前 LOSO 配置:")
    print(f"  LOSO 数据集:   {DATASET_LOSO_NAMES.get(config.loso_dataset, config.loso_dataset)}")
    print(f"  折数 (n_folds): {config.loso_n_folds}")
    print(f"  每折留出被试:  {subjects_info} 个")
    if config.loso_run_all:
        print(f"  运行折:        全部 {config.loso_n_folds} 折")
    else:
        print(f"  运行折:        第 {config.loso_fold_id} 折")
    print_separator("─")
    print(f"  模型:          {config.get_model_display_name()}")
    print(f"  SR 机制:       {config.get_mechanism_display_name()}")
    print(f"  噪声类型:      {config.get_noise_display_name()}")
    print(f"  Epochs:        {config.epochs}")
    print(f"  GPU:           {'CPU' if config.use_cpu else f'#{config.gpu_id}'}")
    print_separator("═")
    print()
    
    # 选择操作
    print("  [1] 使用当前配置直接运行")
    print("  [2] 修改 LOSO 设置 (数据集、折数、运行折)")
    print("  [3] 修改模型/SR配置")
    print("  [0] ← 返回")
    print()
    
    mode_choice = get_choice("请选择: ", ["0", "1", "2", "3"])
    
    if mode_choice == "0":
        return
    
    if mode_choice == "2":
        # 修改 LOSO 设置
        _loso_settings_wizard()
    elif mode_choice == "3":
        # 修改模型/SR 配置
        _loso_model_wizard()
    
    if mode_choice == "1":
        # 直接使用当前配置运行
        _run_loso_with_current_config()
    else:
        # 显示更新后的配置并确认
        print()
        print_separator("═")
        print("更新后的配置:")
        print(f"  LOSO 数据集:   {DATASET_LOSO_NAMES.get(config.loso_dataset, config.loso_dataset)}")
        print(f"  折数:          {config.loso_n_folds}")
        if config.loso_run_all:
            print(f"  运行折:        全部 {config.loso_n_folds} 折")
        else:
            print(f"  运行折:        第 {config.loso_fold_id} 折")
        print(f"  模型:          {config.get_model_display_name()}")
        print(f"  SR 机制:       {config.get_mechanism_display_name()}")
        print(f"  噪声类型:      {config.get_noise_display_name()}")
        print_separator("═")
        print()
        
        if confirm("确认使用此配置运行? [Y/n]: "):
            _run_loso_with_current_config()
    
    save_global_config()
    wait_for_enter()


def _loso_settings_wizard():
    """LOSO 设置向导 - 修改数据集、折数、运行折"""
    print()
    print_separator("─")
    print("修改 LOSO 设置 (直接回车保持当前值)")
    print_separator("─")
    
    # 选择 LOSO 数据集
    print("\n选择 LOSO 数据集:")
    for i, d in enumerate(DATASETS_LOSO, 1):
        marker = "→" if d == config.loso_dataset else " "
        print(f"  {marker}[{i}] {DATASET_LOSO_NAMES[d]}")
    
    choice = get_input("请选择 [1-2]: ", "")
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(DATASETS_LOSO):
                config.loso_dataset = DATASETS_LOSO[idx]
                print_success(f"LOSO 数据集已设为: {DATASET_LOSO_NAMES[config.loso_dataset]}")
        except ValueError:
            pass
    
    # 选择折数
    print(f"\n设置折数 (n_folds) [当前: {config.loso_n_folds}]:")
    print("  n_folds=3: 每折留出 3 个被试")
    print("  n_folds=9: 每折留出 1 个被试 (经典 LOSO)")
    
    choice = get_input("请输入折数 [1-9]: ", "")
    if choice:
        try:
            n_folds = int(choice)
            if 1 <= n_folds <= 9:
                config.loso_n_folds = n_folds
                # 确保 fold_id 不超过 n_folds
                if config.loso_fold_id > n_folds:
                    config.loso_fold_id = 1
                print_success(f"折数已设为: {config.loso_n_folds}")
        except ValueError:
            pass
    
    # 显示分组方案
    print(f"\n分组方案 ({config.loso_n_folds} 折):")
    subjects = list(range(1, 10))
    subjects_per_fold = 9 // config.loso_n_folds
    remainder = 9 % config.loso_n_folds
    idx = 0
    for fold in range(1, config.loso_n_folds + 1):
        count = subjects_per_fold + (1 if fold <= remainder else 0)
        fold_subjects = subjects[idx:idx + count]
        print(f"  Fold {fold}: 测试被试 {fold_subjects}")
        idx += count
    
    # 选择运行模式
    print(f"\n选择运行方式 [当前: {'全部' if config.loso_run_all else f'第 {config.loso_fold_id} 折'}]:")
    print(f"  [1] 运行单个 Fold")
    print(f"  [2] 运行全部 {config.loso_n_folds} 个 Folds")
    
    choice = get_input("请选择 [1-2]: ", "")
    if choice == "1":
        config.loso_run_all = False
        fold_str = get_input(f"请输入 Fold ID [1-{config.loso_n_folds}, 当前: {config.loso_fold_id}]: ", "")
        if fold_str:
            try:
                fold_id = int(fold_str)
                if 1 <= fold_id <= config.loso_n_folds:
                    config.loso_fold_id = fold_id
            except ValueError:
                pass
        print_success(f"已设为运行第 {config.loso_fold_id} 折")
    elif choice == "2":
        config.loso_run_all = True
        print_success(f"已设为运行全部 {config.loso_n_folds} 折")


def _loso_model_wizard():
    """LOSO 模型配置向导 - 修改模型、SR机制、噪声"""
    print()
    print_separator("─")
    print("修改模型/SR 配置 (直接回车保持当前值)")
    print_separator("─")
    
    # 选择模型
    print("\n选择模型:")
    for i, m in enumerate(MODELS, 1):
        marker = "→" if m == config.model else " "
        print(f"  {marker}[{i}] {MODEL_NAMES[m]}")
    
    choice = get_input(f"请选择 [1-{len(MODELS)}]: ", "")
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MODELS):
                config.model = MODELS[idx]
                print_success(f"模型已设为: {config.get_model_display_name()}")
        except ValueError:
            pass
    
    # 选择 SR 机制
    print("\n选择 SR 机制:")
    for i, m in enumerate(MECHANISMS, 1):
        marker = "→" if m == config.mechanism else " "
        print(f"  {marker}[{i}] {MECHANISM_NAMES[m]}")
    
    choice = get_input(f"请选择 [1-{len(MECHANISMS)}]: ", "")
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MECHANISMS):
                config.mechanism = MECHANISMS[idx]
                print_success(f"机制已设为: {config.get_mechanism_display_name()}")
        except ValueError:
            pass
    
    # 选择噪声类型
    print("\n选择噪声类型:")
    for i, n in enumerate(NOISES, 1):
        marker = "→" if n == config.noise_type else " "
        print(f"  {marker}[{i}] {NOISE_NAMES[n]}")
    
    choice = get_input(f"请选择 [1-{len(NOISES)}]: ", "")
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(NOISES):
                config.noise_type = NOISES[idx]
                print_success(f"噪声已设为: {config.get_noise_display_name()}")
        except ValueError:
            pass


def _run_loso_with_current_config():
    """使用当前保存的 LOSO 配置运行"""
    # 临时更新 dataset
    original_dataset = config.dataset
    config.dataset = config.loso_dataset
    
    print()
    print_separator("═")
    print("开始运行 LOSO 实验:")
    print(f"  数据集:    {DATASET_LOSO_NAMES.get(config.loso_dataset, config.loso_dataset)}")
    print(f"  n_folds:   {config.loso_n_folds}")
    if config.loso_run_all:
        print(f"  运行折:    全部 {config.loso_n_folds} 折")
        print_warning(f"这将运行 {config.loso_n_folds} 次完整训练！")
    else:
        print(f"  运行折:    第 {config.loso_fold_id} 折")
    print(f"  模型:      {config.get_model_display_name()}")
    print(f"  SR 机制:   {config.get_mechanism_display_name()}")
    print(f"  噪声类型:  {config.get_noise_display_name()}")
    print_separator("═")
    print()
    
    if config.loso_run_all:
        run_loso_experiment(config, run_all_folds=True, n_folds=config.loso_n_folds)
    else:
        run_loso_experiment(config, fold_id=config.loso_fold_id, n_folds=config.loso_n_folds)
    
    # 恢复原始数据集
    config.dataset = original_dataset


def _run_loso_quick_single(loso_dataset: str):
    """快速单折测试 (n_folds=9, 运行单折)"""
    print()
    print("选择留出哪个被试作为测试集:")
    print("  Subject 1-9 分别对应 Fold 1-9")
    print()
    
    fold_str = get_input("请输入被试编号 [1-9]: ", "1")
    try:
        fold_id = int(fold_str)
        if not (1 <= fold_id <= 9):
            fold_id = 1
    except ValueError:
        fold_id = 1
    
    print()
    print_separator("═")
    print("配置确认:")
    print(f"  数据集:       {DATASET_LOSO_NAMES[loso_dataset]}")
    print(f"  模式:         快速单折测试")
    print(f"  n_folds:      9 (每个被试一折)")
    print(f"  测试被试:     Subject {fold_id}")
    print(f"  模型:         {config.get_model_display_name()}")
    print(f"  SR 机制:      {config.get_mechanism_display_name()}")
    print(f"  噪声类型:     {config.get_noise_display_name()}")
    print(f"  Epochs:       {config.epochs}")
    print(f"  GPU:          {'CPU' if config.use_cpu else f'#{config.gpu_id}'}")
    print_separator("═")
    print()
    
    if confirm("确认开始训练? [Y/n]: "):
        run_loso_experiment(config, fold_id=fold_id, n_folds=9)


def _run_loso_custom_groups(loso_dataset: str):
    """自定义分组模式"""
    print()
    print("设置分组数量 (n_folds):")
    print("  n_folds=3: 每折留出 3 个被试 (默认)")
    print("  n_folds=9: 每折留出 1 个被试 (经典 LOSO)")
    print()
    
    n_folds_str = get_input("请输入折数 [1-9, 默认 3]: ", "3")
    try:
        n_folds = int(n_folds_str)
        if not (1 <= n_folds <= 9):
            n_folds = 3
    except ValueError:
        n_folds = 3
    
    # 显示分组方案
    print()
    print(f"分组方案 ({n_folds} 折):")
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    subjects_per_fold = 9 // n_folds
    remainder = 9 % n_folds
    
    fold_assignments = []
    idx = 0
    for fold in range(1, n_folds + 1):
        # 前 remainder 个折多分一个被试
        count = subjects_per_fold + (1 if fold <= remainder else 0)
        fold_subjects = subjects[idx:idx + count]
        fold_assignments.append(fold_subjects)
        print(f"  Fold {fold}: 测试被试 {fold_subjects}")
        idx += count
    print()
    
    # 选择运行单折还是全部
    print("选择运行方式:")
    print(f"  [1] 运行单个 Fold")
    print(f"  [2] 运行全部 {n_folds} 个 Folds")
    print()
    
    run_mode = get_choice("请选择: ", ["1", "2"])
    
    if run_mode == "1":
        fold_str = get_input(f"请输入 Fold ID [1-{n_folds}]: ", "1")
        try:
            fold_id = int(fold_str)
            if not (1 <= fold_id <= n_folds):
                fold_id = 1
        except ValueError:
            fold_id = 1
        
        print()
        print_separator("═")
        print("配置确认:")
        print(f"  数据集:       {DATASET_LOSO_NAMES[loso_dataset]}")
        print(f"  模式:         自定义分组")
        print(f"  n_folds:      {n_folds}")
        print(f"  当前 Fold:    {fold_id}")
        print(f"  测试被试:     {fold_assignments[fold_id - 1]}")
        print(f"  模型:         {config.get_model_display_name()}")
        print(f"  SR 机制:      {config.get_mechanism_display_name()}")
        print(f"  噪声类型:     {config.get_noise_display_name()}")
        print(f"  Epochs:       {config.epochs}")
        print(f"  GPU:          {'CPU' if config.use_cpu else f'#{config.gpu_id}'}")
        print_separator("═")
        print()
        
        if confirm("确认开始训练? [Y/n]: "):
            run_loso_experiment(config, fold_id=fold_id, n_folds=n_folds)
    
    else:
        print()
        print_separator("═")
        print("配置确认:")
        print(f"  数据集:       {DATASET_LOSO_NAMES[loso_dataset]}")
        print(f"  模式:         自定义分组 (全部运行)")
        print(f"  n_folds:      {n_folds}")
        print(f"  模型:         {config.get_model_display_name()}")
        print(f"  SR 机制:      {config.get_mechanism_display_name()}")
        print(f"  噪声类型:     {config.get_noise_display_name()}")
        print(f"  Epochs:       {config.epochs}")
        print(f"  GPU:          {'CPU' if config.use_cpu else f'#{config.gpu_id}'}")
        print_separator("═")
        print()
        print_warning(f"这将运行 {n_folds} 次完整训练！")
        print()
        
        if confirm("确认运行所有 Folds? [Y/n]: "):
            run_loso_experiment(config, run_all_folds=True, n_folds=n_folds)


def _run_loso_full(loso_dataset: str):
    """完整 LOSO (n_folds=9, 运行全部)"""
    print()
    print_separator("═")
    print("配置确认:")
    print(f"  数据集:       {DATASET_LOSO_NAMES[loso_dataset]}")
    print(f"  模式:         完整 LOSO")
    print(f"  n_folds:      9 (每个被试一折)")
    print(f"  Folds:        1-9 (全部)")
    print(f"  模型:         {config.get_model_display_name()}")
    print(f"  SR 机制:      {config.get_mechanism_display_name()}")
    print(f"  噪声类型:     {config.get_noise_display_name()}")
    print(f"  Epochs:       {config.epochs}")
    print(f"  GPU:          {'CPU' if config.use_cpu else f'#{config.gpu_id}'}")
    print_separator("═")
    print()
    print_warning("这将运行 9 次完整训练，需要较长时间！")
    print()
    
    if confirm("确认运行所有 Folds? [Y/n]: "):
        run_loso_experiment(config, run_all_folds=True, n_folds=9)


def menu_batch_loso_training():
    """批量 LOSO 实验 - 智能多 GPU 并行"""
    print_menu_header("批量 LOSO 实验", "📦")
    
    # 检测 GPU
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            print(f"  检测到 {gpu_count} 个 GPU:")
            for i, name in enumerate(gpu_names):
                print(f"    [{i}] {name}")
        else:
            print("  未检测到 GPU，将使用 CPU 模式")
    except Exception:
        gpu_count = 0
        print("  无法检测 GPU，将使用 CPU 模式")
    
    print()
    print_separator("─")
    print()
    
    print("  [1] 🎯 自动模式 (推荐)")
    print("      使用所有可用 GPU，自动负载均衡")
    print()
    print("  [2] ⚙️  自定义模式")
    print("      选择要遍历的参数和使用的 GPU")
    print()
    print("  [0] ← 返回")
    print()
    
    choice = get_choice("请选择: ", ["0", "1", "2"])
    
    if choice == "0":
        return
    elif choice == "1":
        _run_batch_loso_auto()
    elif choice == "2":
        _run_batch_loso_custom()


def _run_batch_loso_auto():
    """自动批量 LOSO 实验"""
    print()
    print("选择要遍历的参数维度 (可多选，空格分隔):")
    print("  [1] 模型 (EEGNet, Conformer)")
    print("  [2] SR 机制 (Additive, Bistable, Tristable)")
    print("  [3] 噪声类型 (5 种)")
    print("  [4] 仅当前配置")
    print()
    
    choice = get_input("请选择 [默认 4]: ", "4")
    
    selected_models = [config.model]
    selected_mechanisms = [config.mechanism]
    selected_noises = [config.noise_type]
    
    if "1" in choice:
        selected_models = MODELS.copy()
    if "2" in choice:
        selected_mechanisms = MECHANISMS.copy()
    if "3" in choice:
        selected_noises = NOISES.copy()
    
    n_folds = config.loso_n_folds
    total_configs = len(selected_models) * len(selected_mechanisms) * len(selected_noises)
    total_runs = total_configs * n_folds
    
    print()
    print_separator("═")
    print(f"  配置组合数:     {total_configs}")
    print(f"  每配置折数:     {n_folds}")
    print(f"  总训练次数:     {total_runs}")
    print(f"  LOSO 数据集:    {config.loso_dataset}")
    print_separator("═")
    print()
    
    if not confirm(f"确认开始批量 LOSO 实验? [Y/n]: "):
        return
    
    # 生成配置列表
    configs = []
    for model in selected_models:
        for mechanism in selected_mechanisms:
            for noise in selected_noises:
                cfg = ExperimentConfig(
                    model=model,
                    dataset=config.loso_dataset,
                    mechanism=mechanism,
                    noise_type=noise,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    gpu_id=config.gpu_id,
                    use_cpu=config.use_cpu,
                    loso_dataset=config.loso_dataset,
                    loso_n_folds=n_folds
                )
                configs.append(cfg)
    
    # 调用批量 LOSO 运行
    from menu.experiments import run_batch_loso_parallel
    run_batch_loso_parallel(configs, n_folds=n_folds)
    wait_for_enter()


def _run_batch_loso_custom():
    """自定义批量 LOSO 实验"""
    print()
    
    # 选择模型（多选）
    print("选择模型 (用空格分隔多个选项):")
    for i, m in enumerate(MODELS, 1):
        print(f"  [{i}] {MODEL_NAMES[m]}")
    selected_models = _parse_multi_selection(
        input("请选择: "), MODELS
    )
    print(f"  → 已选择: {', '.join(selected_models)}")
    print()
    
    # 选择机制（多选）
    print("选择 SR 机制 (用空格分隔多个选项):")
    for i, m in enumerate(MECHANISMS, 1):
        print(f"  [{i}] {MECHANISM_NAMES[m]}")
    selected_mechanisms = _parse_multi_selection(
        input("请选择: "), MECHANISMS
    )
    print(f"  → 已选择: {', '.join(selected_mechanisms)}")
    print()
    
    # 选择噪声（多选）
    print("选择噪声类型 (用空格分隔多个选项):")
    for i, n in enumerate(NOISES, 1):
        print(f"  [{i}] {NOISE_NAMES[n]}")
    selected_noises = _parse_multi_selection(
        input("请选择: "), NOISES
    )
    print(f"  → 已选择: {', '.join(selected_noises)}")
    print()
    
    # 选择 GPU
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"检测到 {gpu_count} 个 GPU，请选择要使用的 (空格分隔，直接回车使用全部):")
            for i in range(gpu_count):
                print(f"  [{i}] {torch.cuda.get_device_name(i)}")
            gpu_input = input("请选择: ").strip()
            if gpu_input:
                gpu_ids = [int(x) for x in gpu_input.split() if x.isdigit()]
            else:
                gpu_ids = list(range(gpu_count))
        elif gpu_count == 1:
            gpu_ids = [0]
        else:
            gpu_ids = None
    except Exception:
        gpu_ids = None
    
    n_folds = config.loso_n_folds
    total_configs = len(selected_models) * len(selected_mechanisms) * len(selected_noises)
    total_runs = total_configs * n_folds
    
    print()
    print_separator("═")
    print(f"  配置组合数:     {total_configs}")
    print(f"  每配置折数:     {n_folds}")
    print(f"  总训练次数:     {total_runs}")
    print(f"  使用 GPU:       {gpu_ids if gpu_ids else 'CPU'}")
    print_separator("═")
    print()
    
    if not confirm(f"确认开始批量 LOSO 实验? [Y/n]: "):
        return
    
    # 生成配置列表
    configs = []
    for model in selected_models:
        for mechanism in selected_mechanisms:
            for noise in selected_noises:
                cfg = ExperimentConfig(
                    model=model,
                    dataset=config.loso_dataset,
                    mechanism=mechanism,
                    noise_type=noise,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    loso_dataset=config.loso_dataset,
                    loso_n_folds=n_folds
                )
                configs.append(cfg)
    
    # 调用批量 LOSO 运行
    from menu.experiments import run_batch_loso_parallel
    run_batch_loso_parallel(configs, n_folds=n_folds, gpu_ids=gpu_ids)
    wait_for_enter()


def menu_batch_training():
    """传统批量实验 (Legacy)"""
    print_menu_header("传统批量实验", "📦")
    
    print_warning("此功能使用固定训练/测试划分，不推荐用于正式实验。")
    print_info("推荐使用 '批量 LOSO 实验' 获得更可靠的评估结果。")
    print()
    
    print("  [1] 全组合模式")
    print("      运行所有可能的参数组合")
    print(f"      ({len(MODELS)}×{len(DATASETS)}×{len(MECHANISMS)}×{len(NOISES)}="
          f"{len(MODELS)*len(DATASETS)*len(MECHANISMS)*len(NOISES)} 个实验)")
    print()
    print("  [2] 自定义模式")
    print("      选择要遍历的参数子集")
    print()
    print("  [0] ← 返回")
    print()
    
    choice = get_choice("请选择: ", ["0", "1", "2"])
    
    if choice == "0":
        return
    elif choice == "1":
        _run_batch_all()
    elif choice == "2":
        _run_batch_custom()


def _run_batch_all():
    """运行所有组合"""
    total = len(MODELS) * len(DATASETS) * len(MECHANISMS) * len(NOISES)
    
    print()
    print_warning(f"这将运行所有 {total} 种组合！")
    print_warning("预计需要非常长的时间，请确保有足够的计算资源。")
    print()
    
    if not confirm("确认运行所有实验? [y/N]: ", default=False):
        return
    
    # 生成所有配置
    configs = []
    for model in MODELS:
        for dataset in DATASETS:
            for mechanism in MECHANISMS:
                for noise in NOISES:
                    cfg = ExperimentConfig(
                        model=model,
                        dataset=dataset,
                        mechanism=mechanism,
                        noise_type=noise,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        gpu_id=config.gpu_id,
                        use_cpu=config.use_cpu
                    )
                    configs.append(cfg)
    
    run_batch_experiments(configs)
    wait_for_enter()


def _run_batch_custom():
    """自定义批量实验"""
    print()
    
    # 选择模型（多选）
    print("选择模型 (用空格分隔多个选项):")
    for i, m in enumerate(MODELS, 1):
        print(f"  [{i}] {MODEL_NAMES[m]}")
    selected_models = _parse_multi_selection(
        input("请选择: "), MODELS
    )
    print(f"  → 已选择: {', '.join(selected_models)}")
    print()
    
    # 选择数据集（多选）
    print("选择数据集 (用空格分隔多个选项):")
    for i, d in enumerate(DATASETS, 1):
        print(f"  [{i}] {DATASET_NAMES[d]}")
    selected_datasets = _parse_multi_selection(
        input("请选择: "), DATASETS
    )
    print(f"  → 已选择: {', '.join(selected_datasets)}")
    print()
    
    # 选择机制（多选）
    print("选择 SR 机制 (用空格分隔多个选项):")
    for i, m in enumerate(MECHANISMS, 1):
        print(f"  [{i}] {MECHANISM_NAMES[m]}")
    selected_mechanisms = _parse_multi_selection(
        input("请选择: "), MECHANISMS
    )
    print(f"  → 已选择: {', '.join(selected_mechanisms)}")
    print()
    
    # 选择噪声（多选）
    print("选择噪声类型 (用空格分隔多个选项):")
    for i, n in enumerate(NOISES, 1):
        print(f"  [{i}] {NOISE_NAMES[n]}")
    selected_noises = _parse_multi_selection(
        input("请选择: "), NOISES
    )
    print(f"  → 已选择: {', '.join(selected_noises)}")
    print()
    
    # 计算总数
    total = (len(selected_models) * len(selected_datasets) * 
             len(selected_mechanisms) * len(selected_noises))
    
    print(f"将运行 {total} 个实验组合。")
    print()
    
    if not confirm("确认运行? [Y/n]: "):
        return
    
    # 生成配置
    configs = []
    for model in selected_models:
        for dataset in selected_datasets:
            for mechanism in selected_mechanisms:
                for noise in selected_noises:
                    cfg = ExperimentConfig(
                        model=model,
                        dataset=dataset,
                        mechanism=mechanism,
                        noise_type=noise,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        gpu_id=config.gpu_id,
                        use_cpu=config.use_cpu
                    )
                    configs.append(cfg)
    
    run_batch_experiments(configs)
    wait_for_enter()


def _parse_multi_selection(input_str: str, options: List[str]) -> List[str]:
    """解析多选输入"""
    selected = []
    for part in input_str.replace(",", " ").split():
        try:
            idx = int(part) - 1
            if 0 <= idx < len(options) and options[idx] not in selected:
                selected.append(options[idx])
        except ValueError:
            pass
    return selected if selected else [options[0]]


# ============================================================================
# 分析与可视化菜单
# ============================================================================

def menu_analysis():
    """分析与可视化菜单 - 优化版"""
    while True:
        print_menu_header("分析与可视化", "📊")
        
        print("  [1] 🎯 一键分析与可视化 (推荐)")
        print("      自动执行分析并生成所有图表")
        print()
        print("  [2] 📈 仅分析数据")
        print("      运行噪声扫描，生成 CSV/JSON 报告")
        print()
        print("  [3] 🖼️  仅生成图表")
        print("      基于已有分析结果绘图")
        print()
        print("  [0] ← 返回主菜单")
        print()
        
        choice = get_choice("请选择: ", ["0", "1", "2", "3"])
        
        if choice == "0":
            return
        elif choice == "1":
            _menu_full_pipeline()
        elif choice == "2":
            _menu_analyze()
        elif choice == "3":
            _menu_visualize()


def _menu_full_pipeline():
    """一键分析与可视化"""
    print_menu_header("一键分析与可视化", "🎯")
    
    # 选择检查点来源
    print("选择检查点来源:")
    print("  [1] 使用最新训练结果 (自动检测)")
    print("  [2] 使用最新 LOSO 实验结果")
    print("  [3] 手动选择检查点")
    print("  [0] ← 返回")
    print()
    
    choice = get_choice("请选择: ", ["0", "1", "2", "3"])
    
    if choice == "0":
        return
    
    ckpt_path = None
    output_dir = None
    
    if choice == "1":
        ckpt_path = find_latest_checkpoint()
        if ckpt_path:
            output_dir = f"./results/{config.get_experiment_name()}"
            print_info(f"检测到最新 checkpoint: {ckpt_path}")
        else:
            print_error("未找到训练结果")
            wait_for_enter()
            return
    
    elif choice == "2":
        latest_loso = find_latest_loso_output_dir()
        if latest_loso:
            ckpt_path = find_checkpoint_in_dir(latest_loso)
            if not ckpt_path:
                ckpt_path = find_latest_checkpoint()
            output_dir = latest_loso
            print_info(f"检测到最新 LOSO 实验: {latest_loso}")
        else:
            print_error("未找到 LOSO 实验结果")
            wait_for_enter()
            return
    
    elif choice == "3":
        ckpt_path = select_checkpoint()
        if not ckpt_path:
            wait_for_enter()
            return
        output_dir = f"./results/{config.get_experiment_name()}"
    
    if not ckpt_path:
        print_error("未找到有效的 checkpoint")
        wait_for_enter()
        return
    
    # 确认并运行完整流程
    print()
    print_separator("─")
    print_info(f"Checkpoint: {ckpt_path}")
    print_info(f"输出目录: {output_dir}")
    print_separator("─")
    print()
    
    if confirm("确认运行一键分析与可视化? [Y/n]: "):
        from menu.experiments import run_full_analysis_pipeline
        run_full_analysis_pipeline(ckpt_path, config, output_dir)
    
    wait_for_enter()


def _select_results_dir() -> Optional[str]:
    """交互式选择结果目录"""
    results_dir = Path("./results")
    if not results_dir.exists():
        print_error("未找到 results 目录")
        return None
    
    # 收集所有包含 noise_sweep_results.csv 的目录
    valid_dirs = []
    for csv_file in results_dir.rglob("noise_sweep_results.csv"):
        valid_dirs.append(csv_file.parent)
    
    if not valid_dirs:
        print_error("未找到任何分析结果目录（包含 noise_sweep_results.csv 的目录）")
        return None
    
    # 按修改时间排序
    valid_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    print("\n可用的结果目录:")
    print(f"{'ID':<5} {'目录名':<40} {'修改时间'}")
    print("-" * 70)
    
    import datetime
    for i, d in enumerate(valid_dirs[:20], 1):  # 最多显示 20 个
        mtime = d.stat().st_mtime
        time_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        # 获取相对于 results 的路径
        rel_path = d.relative_to(results_dir.parent)
        print(f"[{i}]  {str(rel_path):<40} {time_str}")
    
    if len(valid_dirs) > 20:
        print(f"... 还有 {len(valid_dirs) - 20} 个目录未显示")
    
    print("-" * 70)
    choice = input(f"请选择目录 [1-{min(len(valid_dirs), 20)}] (直接回车选择最新的): ").strip() or "1"
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(valid_dirs):
            return str(valid_dirs[idx])
    except ValueError:
        pass
    
    print_error("无效选择")
    return None


def _menu_analyze():
    """分析已训练模型 - 优化版"""
    print_menu_header("模型分析", "🔍")
    
    # 显示快捷选项
    print("选择检查点来源:")
    print("  [1] 使用最新训练结果 (自动检测)")
    print("  [2] 使用最新 LOSO 实验结果")
    print("  [3] 手动选择检查点")
    print("  [0] ← 返回")
    print()
    
    choice = get_choice("请选择: ", ["0", "1", "2", "3"])
    
    if choice == "0":
        return
    
    ckpt_path = None
    output_dir = None
    
    if choice == "1":
        # 使用最新训练结果
        ckpt_path = find_latest_checkpoint()
        if ckpt_path:
            # 从 checkpoint 路径推断输出目录
            output_dir = f"./results/{config.get_experiment_name()}"
            print_info(f"检测到最新 checkpoint: {ckpt_path}")
        else:
            print_error("未找到训练结果，请先运行训练或手动选择检查点")
            wait_for_enter()
            return
    
    elif choice == "2":
        # 查找最新的 LOSO 输出目录
        latest_loso = find_latest_loso_output_dir()
        if latest_loso:
            # 在 LOSO 目录中查找 checkpoint
            ckpt_path = find_checkpoint_in_dir(latest_loso)
            if not ckpt_path:
                # 如果 LOSO 目录内没有 checkpoint，尝试使用最新的 lightning_logs
                ckpt_path = find_latest_checkpoint()
            output_dir = latest_loso
            print_info(f"检测到最新 LOSO 实验: {latest_loso}")
        else:
            print_error("未找到 LOSO 实验结果")
            wait_for_enter()
            return
    
    elif choice == "3":
        # 手动选择检查点
        ckpt_path = select_checkpoint()
        if not ckpt_path:
            wait_for_enter()
            return
        # 从 checkpoint 路径自动推断输出目录
        output_dir = f"./results/{config.get_experiment_name()}"
    
    if not ckpt_path:
        print_error("未找到有效的 checkpoint")
        wait_for_enter()
        return
    
    # 确认并运行
    print()
    print_separator("─")
    print_info(f"Checkpoint: {ckpt_path}")
    print_info(f"输出目录: {output_dir}")
    print_separator("─")
    print()
    
    if confirm("确认运行分析? [Y/n]: "):
        run_analysis_only(ckpt_path, config, output_dir)
    
    wait_for_enter()


def _menu_visualize():
    """生成可视化图表 - 优化版"""
    print_menu_header("可视化", "📈")
    
    # 第一步：选择结果来源
    print("选择结果来源:")
    print("  [1] 使用最新分析结果 (自动检测)")
    print("  [2] 使用最新 LOSO 分析结果")
    print("  [3] 从列表中选择结果目录")
    print("  [0] ← 返回")
    print()
    
    source_choice = get_choice("请选择: ", ["0", "1", "2", "3"])
    
    if source_choice == "0":
        return
    
    results_dir = None
    
    if source_choice == "1":
        # 自动检测最新分析结果
        results_dir = find_latest_experiment_results()
        if results_dir:
            print_info(f"检测到最新结果: {results_dir}")
        else:
            print_error("未找到分析结果，请先运行分析或手动输入路径")
            wait_for_enter()
            return
    
    elif source_choice == "2":
        # 查找最新的 LOSO 输出目录
        results_dir = find_latest_loso_output_dir()
        if results_dir:
            print_info(f"检测到最新 LOSO: {results_dir}")
        else:
            print_error("未找到 LOSO 实验结果")
            wait_for_enter()
            return
    
    elif source_choice == "3":
        # 手动选择 - 从已有结果目录中选择
        results_dir = _select_results_dir()
        if not results_dir:
            wait_for_enter()
            return
    
    if not results_dir:
        print_error("未找到有效的结果目录")
        wait_for_enter()
        return
    
    print()
    
    # 第二步：选择可视化类型
    print("选择可视化类型:")
    print("  [1] 分析结果可视化（噪声扫描、混淆矩阵等）")
    print("  [2] 训练收敛曲线")
    print("  [3] 全部")
    print()
    
    viz_choice = get_choice("请选择 [1-3]: ", ["1", "2", "3"])
    
    output_dir = f"{results_dir}/figures"
    print_info(f"输出目录: {output_dir}")
    print()
    
    if viz_choice in ["1", "3"]:
        run_visualization_only(
            results_dir=results_dir,
            output_dir=output_dir,
            include_training_curves=(viz_choice == "3"),
            dataset=config.dataset
        )
    
    elif viz_choice == "2":
        run_visualization_only(
            results_dir=".",
            output_dir=output_dir,
            include_training_curves=True,
            dataset=config.dataset
        )
    
    wait_for_enter()


# ============================================================================
# 主菜单
# ============================================================================

def main_menu():
    """主菜单 - LOSO 优先设计"""
    while True:
        print_header()
        
        # 显示当前配置简要 - LOSO 信息优先
        loso_info = f"LOSO Fold {config.loso_fold_id}/{config.loso_n_folds}"
        print(f"  当前: {config.get_model_display_name()} | "
              f"{config.loso_dataset} | {loso_info}")
        print(f"        {config.mechanism} | {config.noise_type}")
        print()
        print_separator("─")
        print()
        
        print("  [1] 🚀 快速开始 (LOSO)")
        print("      一键运行 LOSO 单折测试")
        print()
        print("  [2] ⚙️  实验配置")
        print("      修改模型、数据集、SR参数、训练参数")
        print()
        print("  [3] 🔬 运行实验")
        print("      LOSO 实验、批量 LOSO、传统训练")
        print()
        print("  [4] 📊 分析与可视化")
        print("      一键分析、生成图表")
        print()
        print("  [5] ℹ️  帮助")
        print("      查看各功能的详细说明")
        print()
        print("  [0] 退出")
        print()
        
        choice = get_choice("请选择操作 [0-5]: ", ["0", "1", "2", "3", "4", "5"])
        
        if choice == "1":
            menu_quick_start()
        elif choice == "2":
            menu_config()
        elif choice == "3":
            menu_run_experiments()
        elif choice == "4":
            menu_analysis()
        elif choice == "5":
            show_help_menu()
        elif choice == "0":
            save_global_config()
            print("\n感谢使用 SR-EEG 实验助手！\n")
            sys.exit(0)


# ============================================================================
# 入口
# ============================================================================

def check_environment():
    """检查运行环境"""
    # 确保在项目根目录运行
    if not Path("src/train.py").exists():
        # 尝试切换到项目根目录
        script_dir = Path(__file__).parent.parent
        if (script_dir / "src/train.py").exists():
            os.chdir(script_dir)
        else:
            print("错误: 请在项目根目录运行此脚本")
            print(f"当前目录: {os.getcwd()}")
            sys.exit(1)


def main():
    """主函数"""
    check_environment()
    load_global_config()
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n已取消操作")
        save_global_config()
        sys.exit(0)


if __name__ == "__main__":
    main()