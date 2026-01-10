"""
Utility modules for SR-EEG project.
"""
from .plotting import (
    set_plot_style,
    plot_noise_vs_metrics,
    plot_confusion_matrix,
    plot_multi_confusion_matrices,
    plot_tsne_features,
    plot_attention_weights,
    plot_channel_activation_map,
    plot_sr_comparison,
    create_summary_figure
)

__all__ = [
    "set_plot_style",
    "plot_noise_vs_metrics",
    "plot_confusion_matrix",
    "plot_multi_confusion_matrices",
    "plot_tsne_features",
    "plot_attention_weights",
    "plot_channel_activation_map",
    "plot_sr_comparison",
    "create_summary_figure"
]