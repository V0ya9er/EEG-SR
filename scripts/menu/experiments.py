#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验运行模块

提供实验执行逻辑，包括训练、分析、可视化等。
支持智能多 GPU 并行批量 LOSO 实验。
"""

import os
import subprocess
import platform
import datetime
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Tuple
from queue import Empty

from .config import ExperimentConfig
from .ui import (
    print_menu_header, print_separator, print_success, print_error,
    print_warning, print_info, wait_for_enter, confirm, ProgressBar
)


# ============================================================================
# 工具函数
# ============================================================================

def run_command(cmd: List[str], description: str = "", 
                show_output: bool = True) -> Tuple[bool, str]:
    """
    运行命令
    
    Args:
        cmd: 命令列表
        description: 命令描述
        show_output: 是否显示输出
        
    Returns:
        (是否成功, 输出内容)
    """
    if description:
        print(f"\n{description}")
    
    print(f"执行: {' '.join(cmd)}")
    print()
    
    try:
        if show_output:
            result = subprocess.run(cmd)
            return result.returncode == 0, ""
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        print_error(f"执行错误: {e}")
        return False, str(e)


def find_latest_checkpoint() -> Optional[str]:
    """查找最新的检查点文件"""
    logs_dir = Path("lightning_logs")
    if not logs_dir.exists():
        return None
    
    versions = sorted(
        logs_dir.glob("version_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    for version in versions:
        ckpt_dir = version / "checkpoints"
        if not ckpt_dir.exists():
            continue
        
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            # 优先选择 best 模型
            best_ckpts = [c for c in ckpts if "best" in c.name.lower()]
            if best_ckpts:
                return str(best_ckpts[0])
            # 返回最新的一个检查点
            latest_ckpt = sorted(
                ckpts,
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[0]
            return str(latest_ckpt)
    
    return None


def find_latest_loso_output_dir(exp_name: str = None) -> Optional[str]:
    """
    查找最新的 LOSO 实验输出目录
    
    搜索范围：
    1. ./results/loso/ 目录下的子目录
    2. ./results/ 目录下包含 "loso" 的子目录（如 conformer_bci2a_loso_additive_gaussian）
    
    Args:
        exp_name: 可选的实验名称前缀
    
    Returns:
        最新的输出目录路径，如果找不到则返回 None
    """
    results_dir = Path("./results")
    if not results_dir.exists():
        return None
    
    all_dirs = []
    
    # 搜索 ./results/loso/ 下的目录
    loso_dir = results_dir / "loso"
    if loso_dir.exists():
        if exp_name:
            all_dirs.extend(loso_dir.glob(f"{exp_name}*"))
        else:
            all_dirs.extend([d for d in loso_dir.iterdir() if d.is_dir()])
    
    # 搜索 ./results/ 下包含 "loso" 的目录
    for d in results_dir.iterdir():
        if d.is_dir() and d.name != "loso" and "loso" in d.name.lower():
            if exp_name:
                if d.name.startswith(exp_name) or exp_name in d.name:
                    all_dirs.append(d)
            else:
                all_dirs.append(d)
    
    if not all_dirs:
        return None
    
    # 去重并按修改时间排序，返回最新的
    unique_dirs = list(set(all_dirs))
    unique_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(unique_dirs[0])


def find_latest_experiment_results() -> Optional[str]:
    """
    智能查找最新的实验结果目录
    优先级: results/ 下按修改时间排序，查找包含 noise_sweep_results.csv 的目录
    
    Returns:
        最新的结果目录路径，如果找不到则返回 None
    """
    results_dir = Path("./results")
    if not results_dir.exists():
        return None
    
    # 收集所有包含 noise_sweep_results.csv 的目录
    valid_dirs = []
    for csv_file in results_dir.rglob("noise_sweep_results.csv"):
        valid_dirs.append(csv_file.parent)
    
    if not valid_dirs:
        return None
    
    # 按修改时间排序
    valid_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(valid_dirs[0])


def find_checkpoint_in_dir(directory: str) -> Optional[str]:
    """
    在指定目录中查找 checkpoint 文件
    
    Args:
        directory: 目录路径
    
    Returns:
        checkpoint 路径，如果找不到则返回 None
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    
    # 查找 checkpoints 子目录
    ckpt_dirs = list(dir_path.rglob("checkpoints"))
    
    for ckpt_dir in ckpt_dirs:
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            # 优先选择 best 模型
            best_ckpts = [c for c in ckpts if "best" in c.name.lower()]
            if best_ckpts:
                return str(best_ckpts[0])
            # 返回最新的一个检查点
            latest_ckpt = sorted(
                ckpts,
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[0]
            return str(latest_ckpt)
    
    return None


def find_latest_version_dir() -> Optional[str]:
    """查找最新的 lightning_logs 版本目录"""
    logs_dir = Path("lightning_logs")
    if not logs_dir.exists():
        return None
    
    versions = sorted(
        logs_dir.glob("version_*"), 
        key=lambda p: p.stat().st_mtime, 
        reverse=True
    )
    
    if versions:
        return str(versions[0])
    return None


def get_checkpoint_info(version_dir: Path) -> str:
    """从 hparams.yaml 获取检查点信息"""
    hparams_path = version_dir / "hparams.yaml"
    info = []
    
    if hparams_path.exists():
        try:
            with open(hparams_path, "r", encoding="utf-8") as f:
                content = f.read()
                for line in content.splitlines():
                    if "model_name:" in line:
                        info.append(f"Model: {line.split(':')[-1].strip()}")
                    elif "dataset_name:" in line:
                        info.append(f"Dataset: {line.split(':')[-1].strip()}")
                    elif "mechanism_name:" in line:
                        info.append(f"SR: {line.split(':')[-1].strip()}")
                    elif "noise_name:" in line:
                        info.append(f"Noise: {line.split(':')[-1].strip()}")
        except Exception:
            pass
    
    return ", ".join(info) if info else "无详细信息"


def select_checkpoint() -> Optional[str]:
    """交互式选择检查点"""
    logs_dir = Path("lightning_logs")
    if not logs_dir.exists():
        print_error("未找到 lightning_logs 目录")
        return None
    
    versions = sorted(
        logs_dir.glob("version_*"), 
        key=lambda p: p.stat().st_mtime, 
        reverse=True
    )
    
    valid_versions = []
    
    print("\n可用检查点列表:")
    print(f"{'ID':<5} {'版本':<15} {'时间':<20} {'信息'}")
    print("-" * 80)
    
    count = 0
    for v in versions:
        ckpt_dir = v / "checkpoints"
        if not ckpt_dir.exists():
            continue
        
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            continue
        
        # 获取修改时间
        mtime = v.stat().st_mtime
        time_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        
        # 获取信息
        info = get_checkpoint_info(v)
        
        count += 1
        valid_versions.append((v, ckpts[0]))
        print(f"[{count}]  {v.name:<15} {time_str:<20} {info}")
    
    if count == 0:
        print_error("未找到任何有效的检查点")
        return None
    
    print("-" * 80)
    choice = input(f"请选择检查点 [1-{count}] (直接回车选择最新的): ").strip() or "1"
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(valid_versions):
            return str(valid_versions[idx][1])
    except ValueError:
        pass
    
    print_error("无效选择")
    return None


# ============================================================================
# 实验运行函数
# ============================================================================

def run_single_experiment(config: ExperimentConfig, 
                          run_analysis: bool = True,
                          run_visualization: bool = True) -> bool:
    """
    运行单次实验（训练 + 分析 + 可视化）
    
    Args:
        config: 实验配置
        run_analysis: 是否运行分析
        run_visualization: 是否运行可视化
        
    Returns:
        是否成功
    """
    exp_name = config.get_experiment_name()
    output_dir = config.get_output_dir()
    
    print()
    print("=" * 60)
    print(f"  实验: {exp_name}")
    print("=" * 60)
    
    total_steps = 1 + (1 if run_analysis else 0) + (2 if run_visualization else 0)
    current_step = 0
    
    # 步骤 1: 训练
    current_step += 1
    print(f"\n[{current_step}/{total_steps}] 训练模型...")
    
    train_cmd = [
        "python", "src/train.py",
        f"model={config.model}",
        f"dataset={config.dataset}",
        f"sr/mechanism={config.mechanism}",
        f"sr/noise={config.noise_type}",
        f"trainer.max_epochs={config.epochs}"
    ]
    
    # 添加 GPU 配置
    if config.use_cpu:
        train_cmd.append("trainer.accelerator=cpu")
    elif platform.system() != 'Windows':
        train_cmd.append(f"trainer.devices=[{config.gpu_id}]")
    
    success, _ = run_command(train_cmd)
    if not success:
        print_error("训练失败！跳过后续步骤。")
        return False
    
    # 查找最新检查点
    ckpt_path = find_latest_checkpoint()
    if not ckpt_path:
        print_error("未找到检查点文件！跳过分析和可视化。")
        return False
    
    print_info(f"找到检查点: {ckpt_path}")
    
    # 步骤 2: 生成训练收敛曲线
    if run_visualization:
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 生成训练收敛曲线...")
        
        version_dir = find_latest_version_dir()
        if version_dir:
            curves_output_dir = f"{output_dir}/figures"
            training_curves_cmd = [
                "python", "src/visualize.py",
                "--training-curves",
                "--output-dir", curves_output_dir,
                "--prefix", f"{exp_name}_"
            ]
            run_command(training_curves_cmd)
        else:
            print_warning("未找到训练日志目录，跳过训练曲线生成。")
    
    # 步骤 3: 分析
    if run_analysis:
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 运行分析...")
        
        # 确保路径使用正斜杠
        ckpt_path_safe = ckpt_path.replace("\\", "/")
        output_dir_safe = output_dir.replace("\\", "/")
        
        analyze_cmd = [
            "python", "src/analyze.py",
            f"model={config.model}",
            f"dataset={config.dataset}",
            f"sr/mechanism={config.mechanism}",
            f"sr/noise={config.noise_type}",
            f"+ckpt_path='{ckpt_path_safe}'",
            f"+analysis.output_dir='{output_dir_safe}'"
        ]
        run_command(analyze_cmd)
    
    # 步骤 4: 可视化
    if run_visualization:
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 生成可视化...")
        
        visualize_cmd = [
            "python", "src/visualize.py",
            "--results-dir", output_dir,
            "--output-dir", f"{output_dir}/figures",
            "--dataset", config.dataset
        ]
        run_command(visualize_cmd)
    
    print_success(f"实验 {exp_name} 完成！")
    print_info(f"结果保存在: {output_dir}")
    return True


def run_loso_experiment(config: ExperimentConfig,
                        fold_id: Optional[int] = None,
                        run_all_folds: bool = False,
                        n_folds: int = 9) -> bool:
    """
    运行 LOSO 交叉验证实验
    训练完成后自动运行分析和可视化
    
    Args:
        config: 实验配置
        fold_id: 指定的 fold ID（1-n_folds）
        run_all_folds: 是否运行所有 folds
        n_folds: 折数（默认 9，真正的 LOSO）
        
    Returns:
        是否成功
    """
    exp_name = config.get_experiment_name()
    output_dir = f"./results/loso/{exp_name}"
    
    success = False
    checkpoint_paths = []
    
    if run_all_folds:
        # 运行所有 folds
        print()
        print("=" * 60)
        print(f"  LOSO 实验: {exp_name}")
        print(f"  运行所有 {n_folds} 个 Folds")
        print("=" * 60)
        
        progress = ProgressBar(n_folds, width=30)
        success_count = 0
        
        for fold in range(1, n_folds + 1):
            print(f"\n{'='*60}")
            print(f"  正在运行 Fold {fold}/{n_folds}")
            print(f"{'='*60}")
            
            fold_success, ckpt_path = _run_single_loso_fold(config, fold, output_dir, n_folds)
            if fold_success:
                success_count += 1
                if ckpt_path:
                    checkpoint_paths.append(ckpt_path)
            
            progress.update(fold, f"Fold {fold} 完成")
        
        print_success(f"LOSO 实验完成！成功 {success_count}/{n_folds} 个 Folds")
        success = success_count > 0
    
    elif fold_id:
        # 运行单个 fold
        print()
        print("=" * 60)
        print(f"  LOSO 实验: {exp_name}")
        print(f"  Fold: {fold_id}/{n_folds}")
        print("=" * 60)
        
        fold_success, ckpt_path = _run_single_loso_fold(config, fold_id, output_dir, n_folds)
        success = fold_success
        if ckpt_path:
            checkpoint_paths.append(ckpt_path)
    
    else:
        print_error("请指定 fold_id 或设置 run_all_folds=True")
        return False
    
    # 训练完成后自动运行分析和可视化
    if success and checkpoint_paths:
        print()
        print_separator("═")
        print_info("训练完成，自动开始后处理...")
        print_separator("═")
        
        # 使用最后一个 checkpoint 进行分析
        ckpt_path = checkpoint_paths[-1]
        
        # 步骤 1: 运行噪声扫描分析
        print()
        print_info("[1/3] 运行噪声扫描分析...")
        run_analysis_only(ckpt_path, config, output_dir)
        
        # 步骤 2: 生成训练收敛曲线
        print()
        print_info("[2/3] 生成训练收敛曲线...")
        version_dir = find_latest_version_dir()
        if version_dir:
            curves_output_dir = f"{output_dir}/figures"
            training_curves_cmd = [
                "python", "src/visualize.py",
                "--training-curves",
                "--output-dir", curves_output_dir,
                "--prefix", f"{exp_name}_"
            ]
            run_command(training_curves_cmd)
        else:
            print_warning("未找到训练日志目录，跳过训练曲线生成。")
        
        # 步骤 3: 生成分析结果可视化
        print()
        print_info("[3/3] 生成分析结果可视化...")
        run_visualization_only(
            results_dir=output_dir,
            include_training_curves=False,
            dataset=config.dataset
        )
        
        print()
        print_separator("═")
        print_success(f"所有处理完成！结果保存在: {output_dir}")
        print_separator("═")
    elif success:
        print_warning("未找到 checkpoint，跳过自动分析和可视化")
    
    return success


def _run_single_loso_fold(config: ExperimentConfig,
                          fold_id: int,
                          output_dir: str,
                          n_folds: int = 9) -> Tuple[bool, Optional[str]]:
    """
    运行单个 LOSO fold
    
    Args:
        config: 实验配置
        fold_id: fold ID
        output_dir: 输出目录
        n_folds: 折数
    
    Returns:
        (是否成功, checkpoint路径)
    """
    # 构建 LOSO 训练命令
    train_cmd = [
        "python", "src/loso_train.py",
        f"model={config.model}",
        f"dataset={config.dataset}",
        f"sr/mechanism={config.mechanism}",
        f"sr/noise={config.noise_type}",
        f"trainer.max_epochs={config.epochs}",
        f"dataset.n_folds={n_folds}",
        f"dataset.fold_id={fold_id}"
    ]
    
    # 添加 GPU 配置
    if config.use_cpu:
        train_cmd.append("trainer.accelerator=cpu")
    elif platform.system() != 'Windows':
        train_cmd.append(f"trainer.devices=[{config.gpu_id}]")
    
    success, _ = run_command(train_cmd, "开始 LOSO 训练...")
    
    if not success:
        print_error(f"Fold {fold_id} 训练失败！")
        return False, None
    
    # 查找最新的 checkpoint
    ckpt_path = find_latest_checkpoint()
    
    print_success(f"Fold {fold_id} 完成！")
    return True, ckpt_path


def run_batch_experiments(configs: List[ExperimentConfig],
                          run_analysis: bool = True,
                          run_visualization: bool = True) -> Tuple[int, int]:
    """
    运行批量实验
    
    Args:
        configs: 配置列表
        run_analysis: 是否运行分析
        run_visualization: 是否运行可视化
        
    Returns:
        (成功数, 总数)
    """
    total = len(configs)
    success_count = 0
    
    print()
    print("=" * 60)
    print(f"  批量实验: 共 {total} 个配置")
    print("=" * 60)
    
    for i, config in enumerate(configs, 1):
        exp_name = config.get_experiment_name()
        print(f"\n[{i}/{total}] 运行: {exp_name}")
        
        if run_single_experiment(config, run_analysis, run_visualization):
            success_count += 1
    
    print()
    print("=" * 60)
    print_success(f"批量实验完成！成功 {success_count}/{total}")
    print("=" * 60)
    
    return success_count, total


def run_analysis_only(ckpt_path: str, 
                      config: Optional[ExperimentConfig] = None,
                      output_dir: str = "./results/analysis") -> bool:
    """
    仅运行分析
    
    Args:
        ckpt_path: 检查点路径
        config: 实验配置（可选）
        output_dir: 输出目录
        
    Returns:
        是否成功
    """
    print_menu_header("模型分析", "📊")
    
    print_info(f"检查点: {ckpt_path}")
    print_info(f"输出目录: {output_dir}")
    print()
    
    # 确保路径使用正斜杠
    ckpt_path_safe = ckpt_path.replace("\\", "/")
    output_dir_safe = output_dir.replace("\\", "/")
    
    cmd = [
        "python", "src/analyze.py",
        f"+ckpt_path='{ckpt_path_safe}'",
        f"+analysis.output_dir='{output_dir_safe}'"
    ]
    
    # 如果提供了配置，添加模型和数据集信息
    if config:
        cmd.insert(2, f"model={config.model}")
        cmd.insert(3, f"dataset={config.dataset}")
        cmd.insert(4, f"sr/mechanism={config.mechanism}")
        cmd.insert(5, f"sr/noise={config.noise_type}")
    
    success, _ = run_command(cmd)
    
    if success:
        print_success("分析完成！")
    else:
        print_error("分析失败！")
    
    return success


def run_visualization_only(results_dir: str = "./results/analysis",
                           output_dir: str = None,
                           include_training_curves: bool = True,
                           dataset: str = "bci2a") -> bool:
    """
    仅运行可视化
    
    Args:
        results_dir: 结果目录
        output_dir: 输出目录
        include_training_curves: 是否包含训练曲线
        dataset: 数据集名称
        
    Returns:
        是否成功
    """
    if output_dir is None:
        output_dir = f"{results_dir}/figures"
    
    success = True
    
    # 生成分析结果可视化
    print_info("生成分析结果可视化...")
    cmd = [
        "python", "src/visualize.py",
        "--results-dir", results_dir,
        "--output-dir", output_dir,
        "--dataset", dataset
    ]
    s, _ = run_command(cmd)
    success = success and s
    
    # 生成训练曲线
    if include_training_curves:
        print_info("生成训练收敛曲线...")
        cmd = [
            "python", "src/visualize.py",
            "--training-curves",
            "--output-dir", output_dir
        ]
        s, _ = run_command(cmd)
        success = success and s
    
    if success:
        print_success(f"可视化完成！输出目录: {output_dir}")
    else:
        print_warning("部分可视化任务失败")
    
    return success


# ============================================================================
# 智能多 GPU 并行批量 LOSO 实验
# ============================================================================

def run_batch_loso_parallel(
    configs: List[ExperimentConfig],
    n_folds: int = 9,
    gpu_ids: List[int] = None
) -> Tuple[int, int]:
    """
    智能并行批量 LOSO 实验
    
    自动适配硬件环境:
    - 无 GPU: 单进程 CPU 顺序执行
    - 单 GPU: 单进程 GPU 顺序执行（无进程开销）
    - 多 GPU: multiprocessing 队列 + 多 worker 并行
    
    Args:
        configs: 实验配置列表
        n_folds: 每个配置的 LOSO 折数
        gpu_ids: 要使用的 GPU ID 列表，None 表示自动检测
        
    Returns:
        (成功数, 总任务数)
    """
    import torch
    
    # 自动检测可用 GPU
    if gpu_ids is None:
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print_info("未检测到 GPU，使用 CPU 模式")
            return _run_batch_loso_cpu(configs, n_folds)
        else:
            gpu_ids = list(range(gpu_count))
    
    if len(gpu_ids) == 0:
        print_info("未指定 GPU，使用 CPU 模式")
        return _run_batch_loso_cpu(configs, n_folds)
    elif len(gpu_ids) == 1:
        # 单 GPU 优化：避免 multiprocessing 开销
        print_info(f"使用单 GPU 模式 (GPU {gpu_ids[0]})")
        return _run_batch_loso_single_gpu(configs, n_folds, gpu_ids[0])
    else:
        # 多 GPU：启动 worker 进程
        print_info(f"使用多 GPU 并行模式 (GPU: {gpu_ids})")
        return _run_batch_loso_multi_gpu(configs, n_folds, gpu_ids)


def _run_batch_loso_cpu(
    configs: List[ExperimentConfig],
    n_folds: int
) -> Tuple[int, int]:
    """CPU 模式顺序执行批量 LOSO"""
    total = len(configs) * n_folds
    success = 0
    
    print()
    print("=" * 60)
    print(f"  批量 LOSO (CPU 模式)")
    print(f"  配置数: {len(configs)}, 每配置折数: {n_folds}")
    print(f"  总任务数: {total}")
    print("=" * 60)
    
    progress = ProgressBar(total, width=40)
    task_idx = 0
    
    for config in configs:
        config.use_cpu = True
        exp_name = config.get_experiment_name()
        output_dir = f"./results/loso/{exp_name}"
        
        for fold_id in range(1, n_folds + 1):
            task_idx += 1
            print(f"\n[{task_idx}/{total}] {exp_name} - Fold {fold_id}")
            
            fold_success, _ = _run_single_loso_fold(config, fold_id, output_dir, n_folds)
            if fold_success:
                success += 1
            
            progress.update(task_idx, f"Fold {fold_id} 完成")
    
    print()
    print_separator("═")
    print_success(f"批量 LOSO 完成！成功 {success}/{total}")
    print_separator("═")
    
    return success, total


def _run_batch_loso_single_gpu(
    configs: List[ExperimentConfig],
    n_folds: int,
    gpu_id: int
) -> Tuple[int, int]:
    """单 GPU 顺序执行批量 LOSO（无进程开销）"""
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    total = len(configs) * n_folds
    success = 0
    
    print()
    print("=" * 60)
    print(f"  批量 LOSO (单 GPU 模式, GPU {gpu_id})")
    print(f"  配置数: {len(configs)}, 每配置折数: {n_folds}")
    print(f"  总任务数: {total}")
    print("=" * 60)
    
    progress = ProgressBar(total, width=40)
    task_idx = 0
    
    for config in configs:
        config.use_cpu = False
        config.gpu_id = 0  # CUDA_VISIBLE_DEVICES 已设置，使用设备 0
        exp_name = config.get_experiment_name()
        output_dir = f"./results/loso/{exp_name}"
        
        for fold_id in range(1, n_folds + 1):
            task_idx += 1
            print(f"\n[{task_idx}/{total}] {exp_name} - Fold {fold_id}")
            
            fold_success, _ = _run_single_loso_fold(config, fold_id, output_dir, n_folds)
            if fold_success:
                success += 1
            
            progress.update(task_idx, f"Fold {fold_id} 完成")
    
    print()
    print_separator("═")
    print_success(f"批量 LOSO 完成！成功 {success}/{total}")
    print_separator("═")
    
    return success, total


def _run_batch_loso_multi_gpu(
    configs: List[ExperimentConfig],
    n_folds: int,
    gpu_ids: List[int]
) -> Tuple[int, int]:
    """多 GPU 并行执行批量 LOSO"""
    
    # 构建任务队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    total_tasks = 0
    for config in configs:
        exp_name = config.get_experiment_name()
        output_dir = f"./results/loso/{exp_name}"
        for fold_id in range(1, n_folds + 1):
            # 序列化配置为字典
            task = {
                "model": config.model,
                "dataset": config.loso_dataset,
                "mechanism": config.mechanism,
                "noise_type": config.noise_type,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "fold_id": fold_id,
                "n_folds": n_folds,
                "output_dir": output_dir,
                "exp_name": exp_name
            }
            task_queue.put(task)
            total_tasks += 1
    
    # 添加终止信号（每个 worker 一个）
    for _ in gpu_ids:
        task_queue.put(None)
    
    print()
    print("=" * 60)
    print(f"  批量 LOSO (多 GPU 并行模式)")
    print(f"  GPU: {gpu_ids}")
    print(f"  配置数: {len(configs)}, 每配置折数: {n_folds}")
    print(f"  总任务数: {total_tasks}")
    print("=" * 60)
    print()
    print_info("启动 GPU workers...")
    
    # 每个 GPU 启动一个 worker
    workers = []
    for gpu_id in gpu_ids:
        p = mp.Process(
            target=_gpu_worker,
            args=(task_queue, result_queue, gpu_id)
        )
        p.start()
        workers.append(p)
        print_info(f"  Worker 启动: GPU {gpu_id} (PID: {p.pid})")
    
    # 等待所有 worker 完成
    print()
    print_info("等待任务完成...")
    for p in workers:
        p.join()
    
    # 统计结果
    success_count = 0
    while not result_queue.empty():
        result = result_queue.get()
        if result.get("success", False):
            success_count += 1
    
    print()
    print_separator("═")
    print_success(f"批量 LOSO 完成！成功 {success_count}/{total_tasks}")
    print_separator("═")
    
    return success_count, total_tasks


def _gpu_worker(task_queue: mp.Queue, result_queue: mp.Queue, gpu_id: int):
    """
    单 GPU worker: 不断从队列取任务执行
    
    Args:
        task_queue: 任务队列
        result_queue: 结果队列
        gpu_id: 分配的 GPU ID
    """
    # 设置 CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    while True:
        try:
            task = task_queue.get(timeout=1)
        except Empty:
            continue
        
        if task is None:
            # 收到终止信号
            break
        
        # 执行任务
        success = _run_single_loso_task(task)
        result_queue.put({
            "success": success,
            "exp_name": task.get("exp_name", ""),
            "fold_id": task.get("fold_id", 0),
            "gpu_id": gpu_id
        })


def _run_single_loso_task(task: dict) -> bool:
    """
    执行单个 LOSO 任务
    
    Args:
        task: 任务字典，包含配置信息
        
    Returns:
        是否成功
    """
    # 构建训练命令
    cmd = [
        "python", "src/loso_train.py",
        f"model={task['model']}",
        f"dataset={task['dataset']}",
        f"sr/mechanism={task['mechanism']}",
        f"sr/noise={task['noise_type']}",
        f"trainer.max_epochs={task['epochs']}",
        f"dataset.n_folds={task['n_folds']}",
        f"dataset.fold_id={task['fold_id']}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"任务执行错误: {e}")
        return False


# ============================================================================
# 一键分析与可视化流程
# ============================================================================

def run_full_analysis_pipeline(
    ckpt_path: str,
    config: ExperimentConfig,
    output_dir: str
) -> bool:
    """
    一键运行完整的分析和可视化流程
    
    流程:
    1. 噪声扫描分析 (analyze.py)
    2. 训练收敛曲线 (visualize.py --training-curves)
    3. 分析结果可视化 (visualize.py --results-dir)
    
    Args:
        ckpt_path: 检查点路径
        config: 实验配置
        output_dir: 输出目录
        
    Returns:
        是否全部成功
    """
    print_menu_header("一键分析与可视化", "🎯")
    
    exp_name = config.get_experiment_name()
    success = True
    
    print_info(f"实验: {exp_name}")
    print_info(f"Checkpoint: {ckpt_path}")
    print_info(f"输出目录: {output_dir}")
    print()
    
    # 步骤 1: 噪声扫描分析
    print()
    print_separator("─")
    print_info("[1/3] 运行噪声扫描分析...")
    print_separator("─")
    
    if not run_analysis_only(ckpt_path, config, output_dir):
        print_warning("噪声扫描分析失败，继续后续步骤...")
        success = False
    
    # 步骤 2: 训练收敛曲线
    print()
    print_separator("─")
    print_info("[2/3] 生成训练收敛曲线...")
    print_separator("─")
    
    version_dir = find_latest_version_dir()
    if version_dir:
        curves_output_dir = f"{output_dir}/figures"
        os.makedirs(curves_output_dir, exist_ok=True)
        
        training_curves_cmd = [
            "python", "src/visualize.py",
            "--training-curves",
            "--output-dir", curves_output_dir,
            "--prefix", f"{exp_name}_"
        ]
        s, _ = run_command(training_curves_cmd, show_output=True)
        if not s:
            print_warning("训练曲线生成失败")
            success = False
    else:
        print_warning("未找到训练日志目录，跳过训练曲线生成")
    
    # 步骤 3: 分析结果可视化
    print()
    print_separator("─")
    print_info("[3/3] 生成分析结果可视化...")
    print_separator("─")
    
    # 确定数据集名称（去掉 _loso 后缀用于类别名称）
    dataset_for_viz = config.dataset.replace("_loso", "") if "_loso" in config.dataset else config.dataset
    
    viz_cmd = [
        "python", "src/visualize.py",
        "--results-dir", output_dir,
        "--output-dir", f"{output_dir}/figures",
        "--dataset", dataset_for_viz
    ]
    s, _ = run_command(viz_cmd, show_output=True)
    if not s:
        print_warning("分析结果可视化失败")
        success = False
    
    # 总结
    print()
    print_separator("═")
    if success:
        print_success(f"所有处理完成！结果保存在: {output_dir}")
    else:
        print_warning(f"部分处理失败，请检查日志。结果在: {output_dir}")
    print_separator("═")
    
    return success