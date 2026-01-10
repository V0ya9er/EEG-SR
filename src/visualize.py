"""
SR-EEG Visualization Script
Generates comprehensive visualizations from analysis results.
"""
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

# Allow imports from the root directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.plotting import (
    set_plot_style,
    plot_noise_vs_metrics,
    plot_confusion_matrix,
    plot_multi_confusion_matrices,
    plot_tsne_features,
    plot_attention_weights,
    plot_channel_activation_map,
    plot_sr_comparison,
    create_summary_figure,
    plot_training_curves,
    find_latest_metrics_csv,
    load_training_info,
    add_figure_annotation,
    create_enhanced_title
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# BCI Competition IV 2a class names
BCI2A_CLASS_NAMES = ["Left Hand", "Right Hand", "Foot", "Tongue"]
# BCI Competition IV 2b class names
BCI2B_CLASS_NAMES = ["Left Hand", "Right Hand"]


def load_results(results_dir: str) -> pd.DataFrame:
    """Load noise sweep results from CSV."""
    csv_path = os.path.join(results_dir, "noise_sweep_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    log.info(f"Loaded results with {len(df)} noise intensity levels")
    return df


def load_confusion_matrices(conf_mat_dir: str) -> Dict[float, np.ndarray]:
    """Load confusion matrices from directory."""
    conf_matrices = {}
    
    if not os.path.exists(conf_mat_dir):
        log.warning(f"Confusion matrix directory not found: {conf_mat_dir}")
        return conf_matrices
    
    for file in os.listdir(conf_mat_dir):
        if file.endswith(".npy") and file.startswith("confusion_matrix_D"):
            # Extract noise intensity from filename
            try:
                d_str = file.replace("confusion_matrix_D", "").replace(".npy", "")
                d = float(d_str)
                conf_matrices[d] = np.load(os.path.join(conf_mat_dir, file))
                log.info(f"Loaded confusion matrix for D={d}")
            except ValueError:
                log.warning(f"Could not parse noise intensity from {file}")
    
    return conf_matrices


def visualize_noise_sweep(
    df: pd.DataFrame,
    output_dir: str,
    prefix: str = "",
    training_info: Optional[Dict] = None
) -> None:
    """Generate noise sweep visualizations with experiment info."""
    log.info("Generating noise sweep visualizations...")
    
    # Main metrics plot with training info
    output_path = os.path.join(output_dir, f"{prefix}noise_vs_metrics.png")
    plot_noise_vs_metrics(df, output_path=output_path, training_info=training_info)
    
    # Also save as PDF for publication
    pdf_path = os.path.join(output_dir, f"{prefix}noise_vs_metrics.pdf")
    plot_noise_vs_metrics(df, output_path=pdf_path, training_info=training_info)
    
    log.info(f"Noise sweep plots saved to {output_dir}")


def visualize_confusion_matrices(
    conf_matrices: Dict[float, np.ndarray],
    output_dir: str,
    class_names: Optional[List[str]] = None,
    prefix: str = ""
) -> None:
    """Generate confusion matrix visualizations."""
    if not conf_matrices:
        log.warning("No confusion matrices to visualize")
        return
    
    log.info("Generating confusion matrix visualizations...")
    
    # Individual confusion matrices
    for d, conf_mat in conf_matrices.items():
        output_path = os.path.join(output_dir, f"{prefix}confusion_matrix_D{d:.2f}.png")
        plot_confusion_matrix(
            conf_mat,
            class_names=class_names,
            output_path=output_path,
            title=f"Confusion Matrix (D={d:.2f})"
        )
    
    # Multi-panel confusion matrix plot
    if len(conf_matrices) > 1:
        output_path = os.path.join(output_dir, f"{prefix}confusion_matrices_grid.png")
        plot_multi_confusion_matrices(
            conf_matrices,
            class_names=class_names,
            output_path=output_path
        )
    
    log.info(f"Confusion matrix plots saved to {output_dir}")


def visualize_summary(
    df: pd.DataFrame,
    conf_matrices: Dict[float, np.ndarray],
    output_dir: str,
    class_names: Optional[List[str]] = None,
    prefix: str = "",
    training_info: Optional[Dict] = None
) -> None:
    """Generate comprehensive summary figure with experiment info."""
    log.info("Generating summary figure...")
    
    # Get optimal confusion matrix
    optimal_idx = df["accuracy"].idxmax()
    optimal_d = df.loc[optimal_idx, "noise_intensity"]
    
    # Find closest confusion matrix to optimal D
    conf_matrix = None
    if conf_matrices:
        closest_d = min(conf_matrices.keys(), key=lambda x: abs(x - optimal_d))
        conf_matrix = conf_matrices[closest_d]
        log.info(f"Using confusion matrix from D={closest_d:.2f} (closest to optimal D={optimal_d:.2f})")
    
    output_path = os.path.join(output_dir, f"{prefix}analysis_summary.png")
    create_summary_figure(
        df,
        conf_matrix=conf_matrix,
        class_names=class_names,
        output_path=output_path,
        training_info=training_info
    )
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, f"{prefix}analysis_summary.pdf")
    create_summary_figure(
        df,
        conf_matrix=conf_matrix,
        class_names=class_names,
        output_path=pdf_path,
        training_info=training_info
    )
    
    log.info(f"Summary figure saved to {output_path}")


def extract_features_for_tsne(
    model,
    dataloader,
    device: torch.device,
    max_samples: int = 500
) -> tuple:
    """
    Extract features from model for t-SNE visualization.
    
    Returns:
        Tuple of (features, labels)
    """
    model.eval()
    model.to(device)
    
    all_features = []
    all_labels = []
    n_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if n_samples >= max_samples:
                break
            
            x, y, _ = batch
            x = x.to(device)
            
            # Try to extract intermediate features
            # For Conformer, we want features before the final classifier
            # This requires modifying the forward pass or using hooks
            
            # Simple approach: use model output as features
            if hasattr(model, 'net'):
                net = model.net
                # Apply SR layer if present
                if model.sr_layer is not None:
                    x = model.sr_layer(x)
                
                # For Conformer, extract features before classifier
                if hasattr(net, 'classifier'):
                    # Run through conv layers
                    if x.dim() == 3:
                        x = x.unsqueeze(1)
                    
                    x = net.conv1(x)
                    x = net.bn1(x)
                    x = torch.nn.functional.elu(x)
                    x = net.conv2(x)
                    x = net.bn2(x)
                    x = torch.nn.functional.elu(x)
                    x = net.pool(x)
                    x = net.drop(x)
                    x = x.squeeze(2)
                    x = x.permute(0, 2, 1)
                    
                    # Add CLS token and positions
                    b, t, e = x.shape
                    cls_tokens = net.cls_token.expand(b, -1, -1)
                    x = torch.cat((cls_tokens, x), dim=1)
                    x = x + net.positions[:t+1, :].unsqueeze(0)
                    
                    # Transformer
                    x = net.transformer(x)
                    
                    # Extract CLS token as feature
                    features = x[:, 0, :].cpu().numpy()
                else:
                    # Fallback: just use flattened output
                    logits = model(x.to(device).squeeze(1) if x.dim() == 4 else x)
                    features = logits.cpu().numpy()
            else:
                logits = model(x)
                features = logits.cpu().numpy()
            
            all_features.append(features)
            all_labels.extend(y.numpy())
            n_samples += len(y)
    
    return np.vstack(all_features), np.array(all_labels)


def visualize_tsne(
    model,
    dataloader,
    device: torch.device,
    output_dir: str,
    class_names: Optional[List[str]] = None,
    prefix: str = "",
    max_samples: int = 500
) -> None:
    """Generate t-SNE visualization of features."""
    log.info("Extracting features for t-SNE...")
    
    try:
        features, labels = extract_features_for_tsne(model, dataloader, device, max_samples)
        
        output_path = os.path.join(output_dir, f"{prefix}tsne_features.png")
        plot_tsne_features(
            features,
            labels,
            class_names=class_names,
            output_path=output_path,
            title="t-SNE Feature Visualization"
        )
        
        log.info(f"t-SNE plot saved to {output_path}")
    except Exception as e:
        log.error(f"Error generating t-SNE visualization: {e}")


def visualize_training_curves(
    metrics_csv_path: Optional[str] = None,
    training_info_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    prefix: str = "",
    lightning_logs_dir: str = "lightning_logs"
) -> Optional[str]:
    """
    生成训练收敛曲线可视化。
    
    Args:
        metrics_csv_path: metrics.csv 文件路径。如果为 None，自动查找最新的。
        training_info_path: training_info.json 文件路径（用于早停标注）
        output_dir: 输出目录。如果为 None，保存到 metrics.csv 所在目录。
        prefix: 输出文件名前缀
        lightning_logs_dir: lightning_logs 目录路径
    
    Returns:
        输出图像的路径，如果失败则返回 None
    """
    log.info("Generating training curves visualization...")
    
    # 自动查找 metrics.csv
    if metrics_csv_path is None:
        result = find_latest_metrics_csv(lightning_logs_dir)
        if result is None:
            log.warning("No metrics.csv found in lightning_logs")
            return None
        metrics_csv_path, auto_training_info = result
        if training_info_path is None:
            training_info_path = auto_training_info
        log.info(f"Found metrics file: {metrics_csv_path}")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = str(Path(metrics_csv_path).parent)
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成图表
    output_path = os.path.join(output_dir, f"{prefix}training_curves.png")
    try:
        plot_training_curves(
            metrics_csv_path=metrics_csv_path,
            output_path=output_path,
            training_info_path=training_info_path
        )
        log.info(f"Training curves saved to {output_path}")
        return output_path
    except Exception as e:
        log.error(f"Error generating training curves: {e}")
        return None


def compare_sr_methods(
    results_dirs: Dict[str, str],
    output_dir: str,
    prefix: str = ""
) -> None:
    """
    Compare multiple SR configurations.
    
    Args:
        results_dirs: Dictionary mapping method name to results directory
        output_dir: Output directory for comparison plots
    """
    log.info("Comparing SR methods...")
    
    results = {}
    for name, dir_path in results_dirs.items():
        try:
            df = load_results(dir_path)
            results[name] = df
        except FileNotFoundError:
            log.warning(f"Results not found for {name} in {dir_path}")
    
    if len(results) < 2:
        log.warning("Need at least 2 methods to compare")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare accuracy
    output_path = os.path.join(output_dir, f"{prefix}comparison_accuracy.png")
    plot_sr_comparison(results, metric="accuracy", output_path=output_path,
                       title="Accuracy Comparison Across SR Methods")
    
    # Compare F1
    output_path = os.path.join(output_dir, f"{prefix}comparison_f1.png")
    plot_sr_comparison(results, metric="f1_score", output_path=output_path,
                       title="F1 Score Comparison Across SR Methods")
    
    # Compare Kappa
    output_path = os.path.join(output_dir, f"{prefix}comparison_kappa.png")
    plot_sr_comparison(results, metric="kappa", output_path=output_path,
                       title="Cohen's Kappa Comparison Across SR Methods")
    
    log.info(f"Comparison plots saved to {output_dir}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="SR-EEG Visualization Tool")
    
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default=None,
        help="Directory containing analysis results (noise_sweep_results.csv)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for figures (default: results_dir/figures)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bci2a", "bci2b", "bci2a_loso", "bci2b_loso"],
        default="bci2a",
        help="Dataset type for class names (LOSO variants use same class names as base dataset)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output filenames"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip summary figure generation"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=None,
        help="Directories to compare (format: name:path name:path ...)"
    )
    parser.add_argument(
        "--training-curves",
        action="store_true",
        help="Generate training convergence curves from metrics.csv"
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Path to metrics.csv file (for --training-curves). Auto-detect if not provided."
    )
    parser.add_argument(
        "--lightning-logs",
        type=str,
        default="lightning_logs",
        help="Path to lightning_logs directory (for auto-detecting metrics.csv)"
    )
    
    args = parser.parse_args()
    
    # Handle training curves mode
    if args.training_curves:
        output_dir = args.output_dir
        if output_dir is None and args.results_dir:
            output_dir = os.path.join(args.results_dir, "figures")
        
        result = visualize_training_curves(
            metrics_csv_path=args.metrics_csv,
            output_dir=output_dir,
            prefix=args.prefix,
            lightning_logs_dir=args.lightning_logs
        )
        
        if result:
            print(f"\nTraining curves saved to: {result}")
        else:
            print("\nFailed to generate training curves")
        return
    
    # 如果没有指定 results-dir 且不是 training-curves 模式，显示帮助
    if args.results_dir is None:
        parser.print_help()
        print("\nError: --results-dir is required unless using --training-curves mode")
        return
    
    # Set up output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class names (LOSO variants use same class names as base dataset)
    class_names = BCI2A_CLASS_NAMES if args.dataset.startswith("bci2a") else BCI2B_CLASS_NAMES
    
    # Handle comparison mode
    if args.compare:
        results_dirs = {}
        for item in args.compare:
            if ":" in item:
                name, path = item.split(":", 1)
            else:
                name = os.path.basename(item.rstrip("/"))
                path = item
            results_dirs[name] = path
        
        compare_sr_methods(results_dirs, output_dir, prefix=args.prefix)
        return
    
    # Load results
    try:
        df = load_results(args.results_dir)
    except FileNotFoundError as e:
        log.error(str(e))
        return
    
    # Load training info for enhanced visualizations
    training_info = load_training_info(args.results_dir)
    if training_info:
        log.info(f"Loaded training info: Model={training_info.get('model_name', 'N/A')}, "
                f"Fold={training_info.get('fold_id', 'N/A')}/{training_info.get('n_folds', 'N/A')}")
    
    # Load confusion matrices
    conf_mat_dir = os.path.join(args.results_dir, "confusion_matrices")
    conf_matrices = load_confusion_matrices(conf_mat_dir)
    
    # Generate visualizations with training info
    visualize_noise_sweep(df, output_dir, prefix=args.prefix, training_info=training_info)
    visualize_confusion_matrices(conf_matrices, output_dir, class_names=class_names, prefix=args.prefix)
    
    if not args.no_summary:
        visualize_summary(df, conf_matrices, output_dir, class_names=class_names,
                         prefix=args.prefix, training_info=training_info)
    
    # Print summary statistics
    optimal_idx = df["accuracy"].idxmax()
    optimal_d = df.loc[optimal_idx, "noise_intensity"]
    
    print("\n" + "="*50)
    print("VISUALIZATION COMPLETE")
    print("="*50)
    print(f"\nOptimal Noise Intensity: D = {optimal_d:.2f}")
    print(f"  Accuracy: {df.loc[optimal_idx, 'accuracy']:.4f}")
    print(f"  F1 Score: {df.loc[optimal_idx, 'f1_score']:.4f}")
    print(f"  Kappa:    {df.loc[optimal_idx, 'kappa']:.4f}")
    print(f"\nFigures saved to: {output_dir}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()