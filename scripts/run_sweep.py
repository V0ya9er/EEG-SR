#!/usr/bin/env python
"""
多 GPU 并行实验调度脚本

使用方式:
    # 完整实验（默认 4 GPU）
    python scripts/run_sweep.py --gpus 0,1,2,3
    
    # 指定数据集和模型
    python scripts/run_sweep.py --gpus 0,1 --datasets bci2a_loso --models eegnet
    
    # 从断点恢复
    python scripts/run_sweep.py --gpus 0,1,2,3 --resume
    
    # 仅显示实验列表
    python scripts/run_sweep.py --dry-run
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import itertools
import numpy as np

# 添加项目根目录
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from scripts.experiment_state import ExperimentState


def generate_experiment_configs(
    datasets: List[str] = None,
    models: List[str] = None,
    mechanisms: List[str] = None,
    noises: List[str] = None,
    intensities: List[float] = None,
    n_folds: int = 3
) -> List[Dict]:
    """生成所有实验配置组合"""
    
    # 默认值
    if datasets is None:
        datasets = ["bci2a_loso", "bci2b_loso"]
    if models is None:
        models = ["eegnet", "conformer"]
    if mechanisms is None:
        mechanisms = ["additive", "bistable", "tristable"]
    if noises is None:
        noises = ["gaussian", "uniform", "alpha_stable", "poisson", "colored"]
    if intensities is None:
        intensities = np.arange(0.1, 2.1, 0.1).tolist()
    
    configs = []
    for dataset, model, mechanism, noise, intensity in itertools.product(
        datasets, models, mechanisms, noises, intensities
    ):
        for fold_id in range(1, n_folds + 1):
            configs.append({
                "dataset": dataset,
                "model": model,
                "mechanism": mechanism,
                "noise": noise,
                "intensity": round(intensity, 2),
                "fold_id": fold_id,
                "n_folds": n_folds
            })
    
    return configs


def run_single_experiment(config: Dict, gpu_id: int) -> Tuple[str, bool, str]:
    """在指定 GPU 上运行单个实验"""
    exp_id = ExperimentState.generate_exp_id(config)
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 构建命令
    cmd = [
        sys.executable, "src/loso_train.py",
        f"dataset={config['dataset']}",
        f"model={config['model']}",
        f"sr/mechanism={config['mechanism']}",
        f"sr/noise={config['noise']}",
        f"sr.mechanism.intensity={config['intensity']}",
        f"dataset.fold_id={config['fold_id']}",
        f"dataset.n_folds={config['n_folds']}"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        if result.returncode == 0:
            return exp_id, True, ""
        else:
            return exp_id, False, result.stderr[-500:]  # 最后 500 字符
            
    except subprocess.TimeoutExpired:
        return exp_id, False, "Timeout after 1 hour"
    except Exception as e:
        return exp_id, False, str(e)


def run_experiments_parallel(
    configs: List[Dict],
    gpu_ids: List[int],
    state: ExperimentState,
    resume: bool = False
) -> None:
    """并行运行实验"""
    
    # 获取待运行实验
    all_exp_ids = [ExperimentState.generate_exp_id(c) for c in configs]
    
    if resume:
        pending_ids = set(state.get_pending(all_exp_ids))
        configs = [c for c in configs if ExperimentState.generate_exp_id(c) in pending_ids]
        print(f"恢复模式: 跳过 {len(all_exp_ids) - len(configs)} 个已完成实验")
    
    total = len(configs)
    print(f"待运行实验: {total}")
    print(f"使用 GPU: {gpu_ids}")
    
    if total == 0:
        print("所有实验已完成!")
        return
    
    completed = 0
    failed = 0
    
    # 使用进程池
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        # 提交任务，循环分配 GPU
        futures = {}
        for i, config in enumerate(configs):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            exp_id = ExperimentState.generate_exp_id(config)
            state.mark_running(exp_id)
            
            future = executor.submit(run_single_experiment, config, gpu_id)
            futures[future] = (exp_id, config)
        
        # 收集结果
        for future in as_completed(futures):
            exp_id, config = futures[future]
            try:
                result_id, success, error = future.result()
                if success:
                    state.mark_completed(result_id)
                    completed += 1
                    print(f"✓ [{completed}/{total}] {result_id}")
                else:
                    state.mark_failed(result_id, error)
                    failed += 1
                    print(f"✗ [{completed + failed}/{total}] {result_id}: {error[:100]}")
            except Exception as e:
                state.mark_failed(exp_id, str(e))
                failed += 1
                print(f"✗ [{completed + failed}/{total}] {exp_id}: {e}")
    
    print(f"\n完成: {completed}, 失败: {failed}")
    state.print_status()


def main():
    parser = argparse.ArgumentParser(description="多 GPU 并行实验调度")
    parser.add_argument("--gpus", type=str, default="0",
                        help="GPU ID 列表，逗号分隔 (如 0,1,2,3)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="数据集列表 (如 bci2a_loso bci2b_loso)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="模型列表 (如 eegnet conformer)")
    parser.add_argument("--mechanisms", nargs="+", default=None,
                        help="SR 机制列表")
    parser.add_argument("--noises", nargs="+", default=None,
                        help="噪声类型列表")
    parser.add_argument("--intensities", type=float, nargs="+", default=None,
                        help="噪声强度列表")
    parser.add_argument("--n-folds", type=int, default=3,
                        help="交叉验证折数")
    parser.add_argument("--resume", action="store_true",
                        help="从断点恢复")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示实验列表，不实际运行")
    parser.add_argument("--state-file", type=str, default="experiment_state.json",
                        help="状态文件路径")
    
    args = parser.parse_args()
    
    # 解析 GPU 列表
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    
    # 生成实验配置
    configs = generate_experiment_configs(
        datasets=args.datasets,
        models=args.models,
        mechanisms=args.mechanisms,
        noises=args.noises,
        intensities=args.intensities,
        n_folds=args.n_folds
    )
    
    print(f"生成 {len(configs)} 个实验配置")
    
    if args.dry_run:
        for i, cfg in enumerate(configs[:20]):  # 只显示前 20 个
            exp_id = ExperimentState.generate_exp_id(cfg)
            print(f"  {i+1}. {exp_id}")
        if len(configs) > 20:
            print(f"  ... 还有 {len(configs) - 20} 个实验")
        return
    
    # 运行实验
    state = ExperimentState(args.state_file)
    run_experiments_parallel(configs, gpu_ids, state, resume=args.resume)


if __name__ == "__main__":
    main()