"""
Plotting utilities for SR-EEG visualization.
Migrated and adapted from EEG-Conformer visualization code.
Enhanced with experiment info display capabilities.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
import json
import os


def set_plot_style(style: str = "seaborn-v0_8-whitegrid"):
    """Set consistent plotting style."""
    try:
        plt.style.use(style)
    except OSError:
        # Fallback for older matplotlib versions
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            # Final fallback
            plt.style.use("ggplot")
    
    # Additional style settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


# ============================================================================
# 实验信息加载与显示工具
# ============================================================================

def load_training_info(results_dir: str) -> Optional[Dict[str, Any]]:
    """
    从结果目录或 lightning_logs 加载训练信息。
    
    Args:
        results_dir: 结果目录路径
        
    Returns:
        训练信息字典，如果找不到则返回 None
    """
    # 优先从 results_dir 直接查找
    info_path = os.path.join(results_dir, "training_info.json")
    
    if not os.path.exists(info_path):
        # 尝试在子目录中查找
        for root, dirs, files in os.walk(results_dir):
            if "training_info.json" in files:
                info_path = os.path.join(root, "training_info.json")
                break
    
    if not os.path.exists(info_path):
        # 尝试从 lightning_logs 查找最新的
        logs_dir = Path("lightning_logs")
        if logs_dir.exists():
            versions = sorted(
                logs_dir.glob("version_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            for v in versions:
                candidate = v / "training_info.json"
                if candidate.exists():
                    info_path = str(candidate)
                    break
    
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load training info: {e}")
    
    return None


def format_training_info_text(training_info: Dict[str, Any]) -> str:
    """
    格式化训练信息为显示文本。
    
    Args:
        training_info: 训练信息字典
        
    Returns:
        格式化的文本字符串
    """
    parts = []
    
    # 模型信息
    model_name = training_info.get("model_name", "N/A")
    parts.append(f"Model: {model_name}")
    
    # 数据集信息
    dataset_name = training_info.get("dataset_name", "N/A")
    parts.append(f"Dataset: {dataset_name}")
    
    # LOSO 折数信息
    n_folds = training_info.get("n_folds")
    fold_id = training_info.get("fold_id")
    if n_folds and fold_id:
        parts.append(f"Fold: {fold_id}/{n_folds}")
    
    # SR 信息
    mechanism = training_info.get("mechanism_name", "").replace("SR", "")
    noise = training_info.get("noise_name", "").replace("Noise", "")
    intensity = training_info.get("intensity")
    if mechanism or noise:
        sr_text = f"SR: {mechanism}"
        if noise:
            sr_text += f"+{noise}"
        if intensity is not None:
            sr_text += f" (D={intensity})"
        parts.append(sr_text)
    
    # 训练状态
    early_stopped = training_info.get("early_stopped", False)
    if early_stopped:
        stopped_epoch = training_info.get("stopped_epoch", "?")
        parts.append(f"Early Stopped @ Epoch {stopped_epoch}")
    
    return " | ".join(parts)


def add_figure_annotation(
    fig: plt.Figure,
    training_info: Optional[Dict[str, Any]] = None,
    custom_text: Optional[str] = None,
    position: str = "bottom"
) -> None:
    """
    在图表中添加实验参数注释。
    
    Args:
        fig: matplotlib Figure 对象
        training_info: 训练信息字典
        custom_text: 自定义注释文本（优先于 training_info）
        position: 注释位置 ("bottom", "top")
    """
    if custom_text:
        annotation_text = custom_text
    elif training_info:
        annotation_text = format_training_info_text(training_info)
    else:
        return
    
    # 根据位置设置坐标
    if position == "bottom":
        y_pos = 0.02
        va = 'bottom'
    else:
        y_pos = 0.98
        va = 'top'
    
    fig.text(
        0.5, y_pos, annotation_text,
        ha='center', va=va,
        fontsize=9, color='gray',
        style='italic',
        transform=fig.transFigure
    )


def create_enhanced_title(
    base_title: str,
    training_info: Optional[Dict[str, Any]] = None,
    include_fold: bool = True,
    include_sr: bool = True
) -> str:
    """
    创建包含实验信息的增强标题。
    
    Args:
        base_title: 基础标题
        training_info: 训练信息字典
        include_fold: 是否包含折数信息
        include_sr: 是否包含 SR 信息
        
    Returns:
        增强后的标题字符串
    """
    if not training_info:
        return base_title
    
    subtitle_parts = []
    
    # 折数信息
    if include_fold:
        n_folds = training_info.get("n_folds")
        fold_id = training_info.get("fold_id")
        if n_folds and fold_id:
            subtitle_parts.append(f"Fold {fold_id}/{n_folds}")
    
    # SR 信息
    if include_sr:
        intensity = training_info.get("intensity")
        if intensity is not None:
            subtitle_parts.append(f"D={intensity}")
    
    if subtitle_parts:
        subtitle = " | ".join(subtitle_parts)
        return f"{base_title}\n({subtitle})"
    
    return base_title


def plot_noise_vs_metrics(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "SR Performance vs Noise Intensity",
    figsize: Tuple[int, int] = (14, 5),
    training_info: Optional[Dict[str, Any]] = None,
    show_annotation: bool = True
) -> plt.Figure:
    """
    Plot noise intensity vs performance metrics (Accuracy, F1, Kappa).
    
    Args:
        df: DataFrame with columns: noise_intensity, accuracy, f1_score, kappa
        output_path: Path to save the figure (optional)
        title: Main title for the figure
        figsize: Figure size
        training_info: 训练信息字典，用于在图表中显示实验参数
        show_annotation: 是否显示底部注释
    
    Returns:
        matplotlib Figure object
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metrics = [
        ("accuracy", "Accuracy", "tab:blue"),
        ("f1_score", "F1 Score (Macro)", "tab:green"),
        ("kappa", "Cohen's Kappa", "tab:orange")
    ]
    
    for ax, (col, label, color) in zip(axes, metrics):
        ax.plot(df["noise_intensity"], df[col], marker='o', markersize=3,
                linewidth=2, color=color, alpha=0.8)
        ax.fill_between(df["noise_intensity"], df[col], alpha=0.2, color=color)
        
        # Mark optimal point
        optimal_idx = df[col].idxmax()
        optimal_d = df.loc[optimal_idx, "noise_intensity"]
        optimal_val = df.loc[optimal_idx, col]
        ax.axvline(x=optimal_d, color='red', linestyle='--', alpha=0.7,
                   label=f'Optimal D={optimal_d:.2f}')
        ax.scatter([optimal_d], [optimal_val], color='red', s=100, zorder=5,
                   edgecolor='black', linewidth=2)
        
        ax.set_xlabel("Noise Intensity (D)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # 使用增强标题
    enhanced_title = create_enhanced_title(title, training_info)
    fig.suptitle(enhanced_title, fontsize=16, fontweight='bold', y=1.02)
    
    # 添加底部注释
    if show_annotation and training_info:
        add_figure_annotation(fig, training_info, position="bottom")
    
    plt.tight_layout()
    
    # 调整底部边距以容纳注释
    if show_annotation and training_info:
        plt.subplots_adjust(bottom=0.15)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    return fig


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.
    
    Args:
        conf_matrix: Confusion matrix array (n_classes x n_classes)
        class_names: List of class names for labels
        output_path: Path to save the figure
        title: Figure title
        normalize: Whether to normalize the matrix (show percentages)
        cmap: Colormap name
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_plot_style()
    
    n_classes = conf_matrix.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    if normalize:
        conf_matrix_display = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
        fmt = '.2%'
    else:
        conf_matrix_display = conf_matrix
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        conf_matrix_display,
        annot=True,
        fmt=fmt if not normalize else '.1%',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
        square=True,
        linewidths=0.5
    )
    
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    
    return fig


def plot_multi_confusion_matrices(
    conf_matrices: Dict[float, np.ndarray],
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    suptitle: str = "Confusion Matrices at Different Noise Intensities",
    normalize: bool = True,
    cmap: str = "Blues"
) -> plt.Figure:
    """
    Plot multiple confusion matrices in a grid.
    
    Args:
        conf_matrices: Dictionary mapping noise intensity to confusion matrix
        class_names: List of class names
        output_path: Path to save the figure
        suptitle: Super title for the figure
        normalize: Whether to normalize matrices
        cmap: Colormap
    
    Returns:
        matplotlib Figure object
    """
    set_plot_style()
    
    n_matrices = len(conf_matrices)
    n_cols = min(3, n_matrices)
    n_rows = (n_matrices + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_matrices == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    intensities = sorted(conf_matrices.keys())
    
    for idx, intensity in enumerate(intensities):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        conf_matrix = conf_matrices[intensity]
        n_classes = conf_matrix.shape[0]
        
        if class_names is None:
            labels = [f"C{i}" for i in range(n_classes)]
        else:
            labels = class_names
        
        if normalize:
            conf_matrix_display = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
        else:
            conf_matrix_display = conf_matrix
        
        sns.heatmap(
            conf_matrix_display,
            annot=True,
            fmt='.1%' if normalize else 'd',
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False,
            square=True,
            linewidths=0.5
        )
        
        ax.set_title(f"D = {intensity:.2f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    
    # Hide empty axes
    for idx in range(len(intensities), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Multi-confusion matrix figure saved to {output_path}")
    
    return fig


def plot_tsne_features(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: str = "t-SNE Feature Visualization",
    perplexity: int = 30,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot t-SNE visualization of features.
    Migrated from EEG-Conformer tSNE.py.
    
    Args:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,)
        class_names: List of class names
        output_path: Path to save the figure
        title: Figure title
        perplexity: t-SNE perplexity parameter
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    from sklearn.manifold import TSNE
    
    set_plot_style()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Normalize for visualization
    x_min, x_max = features_2d.min(0), features_2d.max(0)
    features_norm = (features_2d - x_min) / (x_max - x_min + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in unique_labels]
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            features_norm[mask, 0],
            features_norm[mask, 1],
            c=[colors[i]],
            label=class_names[i] if i < len(class_names) else f"Class {label}",
            alpha=0.7,
            s=50,
            edgecolor='white',
            linewidth=0.5
        )
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {output_path}")
    
    return fig


def plot_attention_weights(
    attention: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Attention Weights",
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot attention weight heatmap.
    Inspired by EEG-Conformer CAT.py visualization.
    
    Args:
        attention: Attention weight array (sequence_length, sequence_length) 
                   or (batch, heads, seq, seq)
        output_path: Path to save the figure
        title: Figure title
        cmap: Colormap
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_plot_style()
    
    # Handle different attention shapes
    if attention.ndim == 4:
        # Average over batch and heads
        attention = attention.mean(axis=(0, 1))
    elif attention.ndim == 3:
        # Average over first dimension (batch or heads)
        attention = attention.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(attention, cmap=cmap, aspect='auto')
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Attention plot saved to {output_path}")
    
    return fig


def plot_channel_activation_map(
    activations: np.ndarray,
    channel_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: str = "Channel Activation Map",
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot channel-wise activation heatmap.
    
    Args:
        activations: Activation array (n_channels, n_timepoints) or (batch, channels, time)
        channel_names: List of channel names
        output_path: Path to save the figure
        title: Figure title
        cmap: Colormap
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_plot_style()
    
    # Handle batch dimension
    if activations.ndim == 3:
        activations = activations.mean(axis=0)
    
    n_channels, n_timepoints = activations.shape
    
    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(n_channels)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(activations, cmap=cmap, aspect='auto', interpolation='bilinear')
    
    ax.set_xlabel("Time Points")
    ax.set_ylabel("Channels")
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(channel_names, fontsize=8)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Activation")
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Channel activation map saved to {output_path}")
    
    return fig


def plot_sr_comparison(
    results: Dict[str, pd.DataFrame],
    metric: str = "accuracy",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Compare multiple SR configurations on a single plot.
    
    Args:
        results: Dictionary mapping configuration name to results DataFrame
        metric: Metric column to plot
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for (name, df), color in zip(results.items(), colors):
        ax.plot(
            df["noise_intensity"], 
            df[metric],
            marker='o',
            markersize=4,
            linewidth=2,
            label=name,
            color=color,
            alpha=0.8
        )
    
    ax.set_xlabel("Noise Intensity (D)")
    ax.set_ylabel(metric.replace("_", " ").title())
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Noise Intensity", 
                     fontsize=14, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
    
    return fig


def create_summary_figure(
    df: pd.DataFrame,
    conf_matrix: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    suptitle: str = "SR-EEG Analysis Summary",
    training_info: Optional[Dict[str, Any]] = None,
    show_annotation: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary figure with multiple panels.
    
    Args:
        df: Results DataFrame with noise_intensity and metrics
        conf_matrix: Confusion matrix at optimal noise intensity
        class_names: List of class names
        output_path: Path to save figure
        suptitle: Super title
        training_info: 训练信息字典，用于在图表中显示实验参数
        show_annotation: 是否显示底部注释
    
    Returns:
        matplotlib Figure object
    """
    set_plot_style()
    
    if conf_matrix is not None:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Metrics plots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, :2])
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Individual metric plots
        for ax, (col, label, color) in zip(
            [ax1, ax2, ax3],
            [("accuracy", "Accuracy", "tab:blue"),
             ("f1_score", "F1 Score", "tab:green"),
             ("kappa", "Kappa", "tab:orange")]
        ):
            ax.plot(df["noise_intensity"], df[col], marker='o', markersize=3, 
                    linewidth=2, color=color)
            ax.fill_between(df["noise_intensity"], df[col], alpha=0.2, color=color)
            
            optimal_idx = df[col].idxmax()
            optimal_d = df.loc[optimal_idx, "noise_intensity"]
            optimal_val = df.loc[optimal_idx, col]
            ax.axvline(x=optimal_d, color='red', linestyle='--', alpha=0.7)
            ax.scatter([optimal_d], [optimal_val], color='red', s=80, zorder=5)
            
            ax.set_xlabel("D")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        
        # Combined metrics plot
        for col, label, color in [
            ("accuracy", "Accuracy", "tab:blue"),
            ("f1_score", "F1 Score", "tab:green"),
            ("kappa", "Kappa", "tab:orange")
        ]:
            ax4.plot(df["noise_intensity"], df[col], marker='o', markersize=3,
                     linewidth=2, color=color, label=label)
        ax4.set_xlabel("Noise Intensity (D)")
        ax4.set_ylabel("Score")
        ax4.set_title("All Metrics Comparison")
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        # Confusion matrix
        n_classes = conf_matrix.shape[0]
        if class_names is None:
            labels = [f"C{i}" for i in range(n_classes)]
        else:
            labels = class_names
        
        conf_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
        sns.heatmap(conf_norm, annot=True, fmt='.1%', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax5,
                    cbar=True, square=True)
        ax5.set_xlabel("Predicted")
        ax5.set_ylabel("True")
        ax5.set_title("Confusion Matrix (Optimal D)")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Metric plots
        for ax, (col, label, color) in zip(
            axes.flat[:3],
            [("accuracy", "Accuracy", "tab:blue"),
             ("f1_score", "F1 Score", "tab:green"),
             ("kappa", "Kappa", "tab:orange")]
        ):
            ax.plot(df["noise_intensity"], df[col], marker='o', markersize=3,
                    linewidth=2, color=color)
            ax.fill_between(df["noise_intensity"], df[col], alpha=0.2, color=color)
            ax.set_xlabel("Noise Intensity (D)")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        
        # Combined plot
        ax_combined = axes.flat[3]
        for col, label, color in [
            ("accuracy", "Accuracy", "tab:blue"),
            ("f1_score", "F1 Score", "tab:green"),
            ("kappa", "Kappa", "tab:orange")
        ]:
            ax_combined.plot(df["noise_intensity"], df[col], marker='o', 
                            markersize=3, linewidth=2, color=color, label=label)
        ax_combined.set_xlabel("Noise Intensity (D)")
        ax_combined.set_ylabel("Score")
        ax_combined.set_title("All Metrics Comparison")
        ax_combined.legend()
        ax_combined.grid(True, alpha=0.3)
    
    # 使用增强标题
    enhanced_title = create_enhanced_title(suptitle, training_info)
    fig.suptitle(enhanced_title, fontsize=18, fontweight='bold', y=1.02)
    
    # 添加底部注释
    if show_annotation and training_info:
        add_figure_annotation(fig, training_info, position="bottom")
    
    plt.tight_layout()
    
    # 调整底部边距以容纳注释
    if show_annotation and training_info:
        plt.subplots_adjust(bottom=0.08)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Summary figure saved to {output_path}")
    
    return fig


def plot_training_curves(
    metrics_csv_path: str,
    output_path: Optional[str] = None,
    training_info_path: Optional[str] = None,
    title: str = "Training Convergence Curves",
    figsize: Tuple[int, int] = (14, 6),
    show_annotation: bool = True
) -> plt.Figure:
    """
    绘制训练收敛曲线，包括 loss 和 accuracy 随 epoch 的变化。
    
    Args:
        metrics_csv_path: PyTorch Lightning 生成的 metrics.csv 文件路径
        output_path: 保存图像的路径（可选）
        training_info_path: training_info.json 文件路径，用于标注早停点（可选）
        title: 图表标题
        figsize: 图像尺寸
        show_annotation: 是否显示底部注释
    
    Returns:
        matplotlib Figure 对象
    """
    set_plot_style()
    
    # 读取 metrics CSV
    if not Path(metrics_csv_path).exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_csv_path}")
    
    df = pd.read_csv(metrics_csv_path)
    
    # 读取训练信息（如果有）
    early_stopped = False
    stopped_epoch = None
    training_info = None
    if training_info_path and Path(training_info_path).exists():
        try:
            with open(training_info_path, 'r') as f:
                training_info = json.load(f)
                early_stopped = training_info.get('early_stopped', False)
                stopped_epoch = training_info.get('stopped_epoch', None)
        except Exception:
            pass
    
    # PyTorch Lightning metrics.csv 的列名格式
    # 可能的列: step, epoch, train_loss, train_acc, val_loss, val_acc 等
    
    # 按 epoch 分组获取最后一个值（因为一个 epoch 可能有多个 step 的记录）
    # 先确定有哪些指标列
    metric_cols = [col for col in df.columns if col not in ['step', 'epoch']]
    
    # 对于每个 epoch，取最后一个非 NaN 值
    epochs_data = {}
    
    if 'epoch' in df.columns:
        for epoch in df['epoch'].dropna().unique():
            epoch_df = df[df['epoch'] == epoch]
            epochs_data[int(epoch)] = {}
            for col in metric_cols:
                valid_values = epoch_df[col].dropna()
                if len(valid_values) > 0:
                    epochs_data[int(epoch)][col] = valid_values.iloc[-1]
    else:
        # 如果没有 epoch 列，使用 step 作为 x 轴
        for col in metric_cols:
            epochs_data[col] = df[col].dropna().values
    
    # 转换为 DataFrame
    if epochs_data and isinstance(list(epochs_data.values())[0], dict):
        plot_df = pd.DataFrame(epochs_data).T
        plot_df.index.name = 'epoch'
        plot_df = plot_df.reset_index()
    else:
        plot_df = df
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 确定可用的列
    train_loss_col = None
    val_loss_col = None
    train_acc_col = None
    val_acc_col = None
    
    for col in plot_df.columns:
        col_lower = col.lower()
        if 'train' in col_lower and 'loss' in col_lower:
            train_loss_col = col
        elif 'val' in col_lower and 'loss' in col_lower:
            val_loss_col = col
        elif 'train' in col_lower and 'acc' in col_lower:
            train_acc_col = col
        elif 'val' in col_lower and 'acc' in col_lower:
            val_acc_col = col
    
    x_col = 'epoch' if 'epoch' in plot_df.columns else 'step'
    x_values = plot_df[x_col].values if x_col in plot_df.columns else np.arange(len(plot_df))
    
    # 图1: Loss 曲线
    ax1 = axes[0]
    has_loss_data = False
    
    if train_loss_col and train_loss_col in plot_df.columns:
        train_loss = plot_df[train_loss_col].values
        valid_mask = ~np.isnan(train_loss)
        if np.any(valid_mask):
            ax1.plot(x_values[valid_mask], train_loss[valid_mask],
                    'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
            has_loss_data = True
    
    if val_loss_col and val_loss_col in plot_df.columns:
        val_loss = plot_df[val_loss_col].values
        valid_mask = ~np.isnan(val_loss)
        if np.any(valid_mask):
            ax1.plot(x_values[valid_mask], val_loss[valid_mask],
                    'r-', linewidth=2, label='Val Loss', marker='s', markersize=3)
            has_loss_data = True
            
            # 标注早停点
            if early_stopped and stopped_epoch is not None:
                # 找到早停 epoch 对应的 val_loss
                stop_idx = np.where(x_values == stopped_epoch)[0]
                if len(stop_idx) > 0 and not np.isnan(val_loss[stop_idx[0]]):
                    ax1.axvline(x=stopped_epoch, color='green', linestyle='--',
                               linewidth=2, alpha=0.7, label=f'Early Stop (epoch {stopped_epoch})')
                    ax1.scatter([stopped_epoch], [val_loss[stop_idx[0]]],
                               color='green', s=100, zorder=5, marker='*')
    
    if has_loss_data:
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No loss data available',
                ha='center', va='center', fontsize=12)
        ax1.set_title('Loss (No Data)', fontsize=14)
    
    # 图2: Accuracy 曲线
    ax2 = axes[1]
    has_acc_data = False
    
    if train_acc_col and train_acc_col in plot_df.columns:
        train_acc = plot_df[train_acc_col].values
        valid_mask = ~np.isnan(train_acc)
        if np.any(valid_mask):
            ax2.plot(x_values[valid_mask], train_acc[valid_mask],
                    'b-', linewidth=2, label='Train Acc', marker='o', markersize=3)
            has_acc_data = True
    
    if val_acc_col and val_acc_col in plot_df.columns:
        val_acc = plot_df[val_acc_col].values
        valid_mask = ~np.isnan(val_acc)
        if np.any(valid_mask):
            ax2.plot(x_values[valid_mask], val_acc[valid_mask],
                    'r-', linewidth=2, label='Val Acc', marker='s', markersize=3)
            has_acc_data = True
            
            # 标注早停点
            if early_stopped and stopped_epoch is not None:
                stop_idx = np.where(x_values == stopped_epoch)[0]
                if len(stop_idx) > 0 and not np.isnan(val_acc[stop_idx[0]]):
                    ax2.axvline(x=stopped_epoch, color='green', linestyle='--',
                               linewidth=2, alpha=0.7, label=f'Early Stop (epoch {stopped_epoch})')
                    ax2.scatter([stopped_epoch], [val_acc[stop_idx[0]]],
                               color='green', s=100, zorder=5, marker='*')
    
    if has_acc_data:
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
    else:
        ax2.text(0.5, 0.5, 'No accuracy data available',
                ha='center', va='center', fontsize=12)
        ax2.set_title('Accuracy (No Data)', fontsize=14)
    
    # 添加总标题（增强版）
    early_stop_text = " (Early Stopped)" if early_stopped else ""
    enhanced_title = create_enhanced_title(f"{title}{early_stop_text}", training_info)
    fig.suptitle(enhanced_title, fontsize=16, fontweight='bold', y=1.02)
    
    # 添加底部注释
    if show_annotation and training_info:
        add_figure_annotation(fig, training_info, position="bottom")
    
    plt.tight_layout()
    
    # 调整底部边距以容纳注释
    if show_annotation and training_info:
        plt.subplots_adjust(bottom=0.12)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {output_path}")
    
    return fig


def find_latest_metrics_csv(lightning_logs_dir: str = "lightning_logs") -> Optional[Tuple[str, str]]:
    """
    查找最新的 metrics.csv 文件。
    
    Args:
        lightning_logs_dir: lightning_logs 目录路径
    
    Returns:
        (metrics_csv_path, training_info_path) 或 None
    """
    logs_dir = Path(lightning_logs_dir)
    if not logs_dir.exists():
        return None
    
    # 获取所有版本目录，按修改时间排序
    versions = sorted(logs_dir.glob("version_*"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    
    for version_dir in versions:
        metrics_path = version_dir / "metrics.csv"
        if metrics_path.exists():
            training_info_path = version_dir / "training_info.json"
            return (
                str(metrics_path),
                str(training_info_path) if training_info_path.exists() else None
            )
    
    return None