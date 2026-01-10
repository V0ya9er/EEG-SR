#!/usr/bin/env python
"""
LOSO 交叉验证结果聚合分析

功能:
- 收集各折实验结果（从新的 outputs/ 目录）
- 计算每个被试的性能
- 计算平均值和标准差
- 生成汇总报告
- 导出 CSV

使用方式:
    python scripts/loso_analyze.py --results-dir ./outputs
    python scripts/loso_analyze.py --results-dir ./outputs --output ./results/summary.csv
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# 添加项目根目录
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("请安装 pandas 和 numpy: pip install pandas numpy")
    sys.exit(1)


def parse_folder_name(folder_name: str) -> Optional[Dict]:
    """
    解析语义化文件夹名
    
    格式: {dataset}_{model}_{mechanism}_{noise}_fold{id}_D{intensity}
    例如: bci2a_eegnet_add_gauss_fold1_D0.5
    
    Returns:
        解析后的字典，或 None（解析失败）
    """
    # 正则表达式匹配
    pattern = r'^(\w+)_(\w+)_(\w+)_(\w+)_fold(\d+)_D([\d.]+)$'
    match = re.match(pattern, folder_name)
    
    if match:
        return {
            "dataset": match.group(1),
            "model": match.group(2),
            "mechanism": match.group(3),
            "noise": match.group(4),
            "fold_id": int(match.group(5)),
            "intensity": float(match.group(6)),
        }
    return None


def find_experiment_results(results_dir: Path) -> List[Dict]:
    """
    搜索结果目录，收集所有实验的 training_info.json
    
    搜索优先级:
    1. outputs/ 目录（新的语义化命名）
    2. outputs/multirun/ 目录（Hydra multirun 输出）
    3. lightning_logs/ 目录（旧格式，向后兼容）
    
    Returns:
        实验结果列表
    """
    results = []
    
    # 1. 搜索 outputs 目录（新格式）
    outputs_dir = results_dir / "outputs" if (results_dir / "outputs").exists() else results_dir
    
    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # 跳过 multirun 目录，后面单独处理
        if exp_dir.name == "multirun":
            continue
        
        info_file = exp_dir / "training_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            info["result_path"] = str(exp_dir)
            info["folder_name"] = exp_dir.name
            results.append(info)
        else:
            # 尝试从文件夹名解析
            parsed = parse_folder_name(exp_dir.name)
            if parsed:
                parsed["result_path"] = str(exp_dir)
                parsed["folder_name"] = exp_dir.name
                results.append(parsed)
    
    # 2. 搜索 multirun 目录
    multirun_dir = outputs_dir / "multirun"
    if multirun_dir.exists():
        for exp_dir in sorted(multirun_dir.iterdir()):
            if exp_dir.is_dir():
                info_file = exp_dir / "training_info.json"
                if info_file.exists():
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    info["result_path"] = str(exp_dir)
                    info["folder_name"] = exp_dir.name
                    results.append(info)
    
    # 3. 向后兼容：搜索 lightning_logs 目录
    lightning_logs = results_dir / "lightning_logs"
    if lightning_logs.exists():
        for version_dir in sorted(lightning_logs.iterdir()):
            if version_dir.is_dir() and version_dir.name.startswith("version_"):
                info_file = version_dir / "training_info.json"
                if info_file.exists():
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    info["result_path"] = str(version_dir)
                    info["folder_name"] = version_dir.name
                    results.append(info)
    
    return results


def read_metrics(result_path: Path) -> Tuple[float, float, float]:
    """
    读取实验指标
    
    尝试从以下位置读取:
    1. training_info.json 中的 final_metrics
    2. metrics.csv（Lightning 格式）
    
    Returns:
        (accuracy, f1, kappa) 或 (nan, nan, nan)
    """
    # 尝试从 training_info.json 读取
    info_file = result_path / "training_info.json"
    if info_file.exists():
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        if "final_metrics" in info:
            fm = info["final_metrics"]
            return (
                fm.get("test_acc", float('nan')),
                fm.get("test_f1", float('nan')),
                fm.get("test_kappa", float('nan'))
            )
    
    # 尝试从 metrics.csv 读取
    metrics_file = result_path / "metrics.csv"
    if metrics_file.exists():
        try:
            df = pd.read_csv(metrics_file)
            # 查找测试指标列
            for acc_col in ["test_acc", "test/acc", "test_accuracy"]:
                if acc_col in df.columns:
                    test_rows = df.dropna(subset=[acc_col])
                    if not test_rows.empty:
                        last_row = test_rows.iloc[-1]
                        return (
                            last_row.get(acc_col, float('nan')),
                            last_row.get("test_f1", last_row.get("test/f1", float('nan'))),
                            last_row.get("test_kappa", last_row.get("test/kappa", float('nan')))
                        )
        except Exception:
            pass
    
    # 检查子目录（如 lightning_logs/version_0）
    for subdir in result_path.iterdir():
        if subdir.is_dir():
            sub_metrics = subdir / "metrics.csv"
            if sub_metrics.exists():
                try:
                    df = pd.read_csv(sub_metrics)
                    for acc_col in ["test_acc", "test/acc", "test_accuracy"]:
                        if acc_col in df.columns:
                            test_rows = df.dropna(subset=[acc_col])
                            if not test_rows.empty:
                                last_row = test_rows.iloc[-1]
                                return (
                                    last_row.get(acc_col, float('nan')),
                                    last_row.get("test_f1", last_row.get("test/f1", float('nan'))),
                                    last_row.get("test_kappa", last_row.get("test/kappa", float('nan')))
                                )
                except Exception:
                    pass
    
    return (float('nan'), float('nan'), float('nan'))


def aggregate_fold_results(results: List[Dict]) -> pd.DataFrame:
    """
    聚合各折结果
    """
    rows = []
    
    for result in results:
        result_path = Path(result.get("result_path", ""))
        accuracy, f1, kappa = read_metrics(result_path)
        
        rows.append({
            "dataset": result.get("dataset_name", result.get("dataset", "unknown")),
            "model": result.get("model_name", result.get("model", "unknown")),
            "mechanism": result.get("mechanism_name", result.get("mechanism", "unknown")),
            "noise": result.get("noise_name", result.get("noise", "unknown")),
            "intensity": result.get("intensity", 0),
            "fold_id": result.get("fold_id", 0),
            "n_folds": result.get("n_folds", 0),
            "test_subjects": str(result.get("test_subjects", [])),
            "accuracy": accuracy,
            "f1": f1,
            "kappa": kappa,
            "folder": result.get("folder_name", ""),
            "result_path": result.get("result_path", "")
        })
    
    return pd.DataFrame(rows)


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    按实验配置分组，计算平均值和标准差
    """
    group_cols = ["dataset", "model", "mechanism", "noise", "intensity"]
    
    summary_rows = []
    for name, group in df.groupby(group_cols, dropna=False):
        dataset, model, mechanism, noise, intensity = name
        
        summary_rows.append({
            "dataset": dataset,
            "model": model,
            "mechanism": mechanism,
            "noise": noise,
            "intensity": intensity,
            "n_folds_actual": len(group),
            "accuracy_mean": group["accuracy"].mean(),
            "accuracy_std": group["accuracy"].std(),
            "f1_mean": group["f1"].mean(),
            "f1_std": group["f1"].std(),
            "kappa_mean": group["kappa"].mean(),
            "kappa_std": group["kappa"].std()
        })
    
    return pd.DataFrame(summary_rows)


def print_subject_performance(df: pd.DataFrame):
    """打印每个被试/折的性能"""
    print("\n" + "="*70)
    print("📊 每个折的测试性能")
    print("="*70)
    
    # 按配置分组
    for (dataset, model, mech, noise, intensity), group in df.groupby(
        ["dataset", "model", "mechanism", "noise", "intensity"], dropna=False
    ):
        print(f"\n🔬 {dataset} | {model} | {mech} | {noise} | D={intensity}")
        print("-" * 60)
        
        for _, row in group.iterrows():
            acc_str = f"{row['accuracy']:.4f}" if not pd.isna(row['accuracy']) else "N/A"
            f1_str = f"{row['f1']:.4f}" if not pd.isna(row['f1']) else "N/A"
            kappa_str = f"{row['kappa']:.4f}" if not pd.isna(row['kappa']) else "N/A"
            
            print(f"  Fold {row['fold_id']}: 测试被试 {row['test_subjects']}")
            print(f"    Accuracy: {acc_str}  |  F1: {f1_str}  |  Kappa: {kappa_str}")


def print_summary(summary_df: pd.DataFrame):
    """打印汇总统计"""
    print("\n" + "="*70)
    print("📈 实验汇总 (平均 ± 标准差)")
    print("="*70)
    
    for _, row in summary_df.iterrows():
        config = f"{row['dataset']}_{row['model']}_{row['mechanism']}_{row['noise']}_D{row['intensity']}"
        
        acc_str = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}" if not pd.isna(row['accuracy_mean']) else "N/A"
        f1_str = f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f}" if not pd.isna(row['f1_mean']) else "N/A"
        kappa_str = f"{row['kappa_mean']:.4f} ± {row['kappa_std']:.4f}" if not pd.isna(row['kappa_mean']) else "N/A"
        
        print(f"\n📌 {config}")
        print(f"   Accuracy: {acc_str}")
        print(f"   F1 Score: {f1_str}")
        print(f"   Kappa:    {kappa_str}")
        print(f"   (基于 {row['n_folds_actual']} 折)")


def find_optimal_config(summary_df: pd.DataFrame) -> Optional[Dict]:
    """找到最优配置"""
    if summary_df.empty:
        return None
    
    # 过滤掉没有有效准确率的行
    valid_df = summary_df.dropna(subset=["accuracy_mean"])
    if valid_df.empty:
        return None
    
    best_idx = valid_df["accuracy_mean"].idxmax()
    best = valid_df.loc[best_idx]
    
    return {
        "dataset": best["dataset"],
        "model": best["model"],
        "mechanism": best["mechanism"],
        "noise": best["noise"],
        "intensity": best["intensity"],
        "accuracy": f"{best['accuracy_mean']:.4f} ± {best['accuracy_std']:.4f}",
        "f1": f"{best['f1_mean']:.4f} ± {best['f1_std']:.4f}",
        "kappa": f"{best['kappa_mean']:.4f} ± {best['kappa_std']:.4f}"
    }


def main():
    parser = argparse.ArgumentParser(description="LOSO 结果聚合分析")
    parser.add_argument("--results-dir", type=str, default=".",
                        help="结果目录 (默认: 当前目录)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出详细结果 CSV 文件路径")
    parser.add_argument("--summary-output", "-s", type=str, default=None,
                        help="输出汇总 CSV 文件路径")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示每个折的详细信息")
    parser.add_argument("--format", type=str, choices=["table", "json"], default="table",
                        help="输出格式")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # 收集结果
    print(f"🔍 搜索结果目录: {results_dir.absolute()}")
    results = find_experiment_results(results_dir)
    print(f"📂 找到 {len(results)} 个实验结果")
    
    if not results:
        print("\n❌ 未找到任何实验结果")
        print("\n提示: 确保结果目录包含以下之一:")
        print("  - outputs/{dataset}_{model}_{mech}_{noise}_fold{id}_D{intensity}/training_info.json")
        print("  - lightning_logs/version_X/training_info.json")
        return
    
    # 聚合结果
    df = aggregate_fold_results(results)
    
    # 计算汇总统计
    summary_df = compute_summary_statistics(df)
    
    # 输出
    if args.format == "json":
        print(json.dumps({
            "experiments": df.to_dict(orient="records"),
            "summary": summary_df.to_dict(orient="records")
        }, indent=2, ensure_ascii=False))
    else:
        if args.verbose:
            print_subject_performance(df)
        
        print_summary(summary_df)
        
        # 找到最优配置
        optimal = find_optimal_config(summary_df)
        if optimal:
            print("\n" + "="*70)
            print("🏆 最优配置")
            print("="*70)
            for k, v in optimal.items():
                print(f"   {k}: {v}")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n💾 详细结果已保存到: {output_path}")
    
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"💾 汇总统计已保存到: {summary_path}")


if __name__ == "__main__":
    main()