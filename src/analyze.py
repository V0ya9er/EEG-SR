"""
SR-EEG Analysis Script
Analyzes model performance under varying noise intensities.
"""
import os
import sys
import logging

# Allow imports from the root directory - must be done before other imports
# Use absolute path to handle Hydra's working directory changes
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from typing import List, Dict, Any, Optional

from src.models.lit_model import LitEEGModel
from src.modules.sr_layer import (
    AdditiveSR, BistableSR, TristableSR,
    GaussianNoise, UniformNoise, AlphaStableNoise, PoissonNoise, ColoredNoise
)

log = logging.getLogger(__name__)


def create_noise_source(noise_type: str, **kwargs):
    """Create a noise source based on type."""
    if noise_type == "gaussian":
        return GaussianNoise()
    elif noise_type == "uniform":
        return UniformNoise()
    elif noise_type == "alpha_stable":
        alpha = kwargs.get("alpha", 1.5)
        return AlphaStableNoise(alpha=alpha)
    elif noise_type == "poisson":
        lam = kwargs.get("lam", 1.0)
        return PoissonNoise(lam=lam)
    elif noise_type == "colored":
        beta = kwargs.get("beta", 1.0)
        return ColoredNoise(beta=beta)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def create_sr_layer(mechanism_type: str, noise_source, intensity: float, **kwargs):
    """Create an SR layer based on mechanism type."""
    if mechanism_type == "additive":
        return AdditiveSR(noise_source=noise_source, intensity=intensity)
    elif mechanism_type == "bistable":
        a = kwargs.get("a", 1.0)
        b = kwargs.get("b", 1.0)
        dt = kwargs.get("dt", 0.01)
        return BistableSR(noise_source=noise_source, intensity=intensity, a=a, b=b, dt=dt)
    elif mechanism_type == "tristable":
        a = kwargs.get("a", 1.0)
        b = kwargs.get("b", 1.0)
        c = kwargs.get("c", 1.0)
        dt = kwargs.get("dt", 0.01)
        return TristableSR(noise_source=noise_source, intensity=intensity, a=a, b=b, c=c, dt=dt)
    else:
        raise ValueError(f"Unknown SR mechanism: {mechanism_type}")


def evaluate_model(
    model: LitEEGModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate model and return metrics.
    
    Returns:
        Dictionary containing accuracy, f1_score, kappa, confusion_matrix, 
        predictions, and true labels.
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 获取类别数（从配置或推断）
    # 对于 BCI IV 2a，有 4 个类别 (0, 1, 2, 3)
    # 对于 BCI IV 2b，有 2 个类别 (0, 1)
    n_classes = len(np.unique(np.concatenate([all_labels, all_preds])))
    if n_classes <= 2:
        all_labels_list = list(range(2))
    else:
        all_labels_list = list(range(4))
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', labels=all_labels_list)
    kappa = cohen_kappa_score(all_labels, all_preds)
    # 使用 labels 参数确保混淆矩阵包含所有类别，即使某类别不存在
    conf_mat = confusion_matrix(all_labels, all_preds, labels=all_labels_list)
    
    return {
        "accuracy": acc,
        "f1_score": f1,
        "kappa": kappa,
        "confusion_matrix": conf_mat,
        "predictions": all_preds,
        "labels": all_labels
    }


def analyze_noise_sweep(
    cfg: DictConfig,
    model: LitEEGModel,
    datamodule: pl.LightningDataModule,
    noise_intensities: List[float],
    noise_type: str = "gaussian",
    mechanism_type: str = "additive",
    device: torch.device = None,
    **sr_kwargs
) -> pd.DataFrame:
    """
    Sweep through noise intensities and evaluate model performance.
    
    Args:
        cfg: Hydra configuration
        model: Trained LitEEGModel
        datamodule: DataModule with test dataloader
        noise_intensities: List of noise intensity values (D)
        noise_type: Type of noise ('gaussian', 'uniform', 'alpha_stable')
        mechanism_type: SR mechanism ('additive', 'bistable', 'tristable')
        device: Torch device
        **sr_kwargs: Additional arguments for SR layer
    
    Returns:
        DataFrame with columns: noise_intensity, accuracy, f1_score, kappa
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    
    # Get the base network (without SR layer)
    base_net = model.net
    
    # Ensure datamodule is set up
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    
    log.info(f"Starting noise intensity sweep: {len(noise_intensities)} levels")
    log.info(f"Noise type: {noise_type}, Mechanism: {mechanism_type}")
    
    for intensity in tqdm(noise_intensities, desc="Noise Sweep"):
        # Create new noise source and SR layer for each intensity
        noise_source = create_noise_source(noise_type, **sr_kwargs)
        sr_layer = create_sr_layer(mechanism_type, noise_source, intensity, **sr_kwargs)
        
        # Create a new model with updated SR layer
        # We need to temporarily set the SR layer
        test_model = LitEEGModel(
            net=base_net, 
            sr_layer=sr_layer,
            lr=model.lr,
            weight_decay=model.weight_decay
        )
        
        # Evaluate
        metrics = evaluate_model(test_model, test_loader, device)
        
        results.append({
            "noise_intensity": intensity,
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "kappa": metrics["kappa"]
        })
        
        log.debug(f"D={intensity:.2f}: Acc={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1_score']:.4f}, Kappa={metrics['kappa']:.4f}")
    
    df = pd.DataFrame(results)
    return df


def save_confusion_matrices(
    cfg: DictConfig,
    model: LitEEGModel,
    datamodule: pl.LightningDataModule,
    noise_intensities: List[float],
    output_dir: str,
    noise_type: str = "gaussian",
    mechanism_type: str = "additive",
    device: torch.device = None,
    **sr_kwargs
) -> Dict[float, np.ndarray]:
    """
    Save confusion matrices for specific noise intensities.
    
    Returns:
        Dictionary mapping noise intensity to confusion matrix
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_net = model.net
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    
    conf_matrices = {}
    
    for intensity in tqdm(noise_intensities, desc="Computing Confusion Matrices"):
        noise_source = create_noise_source(noise_type, **sr_kwargs)
        sr_layer = create_sr_layer(mechanism_type, noise_source, intensity, **sr_kwargs)
        
        test_model = LitEEGModel(
            net=base_net,
            sr_layer=sr_layer,
            lr=model.lr,
            weight_decay=model.weight_decay
        )
        
        metrics = evaluate_model(test_model, test_loader, device)
        conf_matrices[intensity] = metrics["confusion_matrix"]
        
        # Save to numpy file
        np.save(
            os.path.join(output_dir, f"confusion_matrix_D{intensity:.2f}.npy"),
            metrics["confusion_matrix"]
        )
    
    return conf_matrices


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main analysis function."""
    # Set seed
    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Instantiate DataModule
    log.info(f"Instantiating DataModule <{cfg.dataset._target_}>")
    datamodule = hydra.utils.instantiate(cfg.dataset)
    
    # Instantiate Model (Network)
    log.info(f"Instantiating Network <{cfg.model._target_}>")
    net = hydra.utils.instantiate(cfg.model)
    
    # Checkpoint path
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        log.error("Please provide a checkpoint path using +ckpt_path=/path/to/checkpoint.ckpt")
        return
    
    # Load model from checkpoint
    log.info(f"Loading model from checkpoint: {ckpt_path}")
    # Create a dummy SR layer for loading (will be replaced during analysis)
    noise_source = hydra.utils.instantiate(cfg.sr.noise)
    sr_layer = hydra.utils.instantiate(cfg.sr.mechanism, noise_source=noise_source)
    model = LitEEGModel.load_from_checkpoint(
        ckpt_path, net=net, sr_layer=sr_layer, weights_only=False
    )
    
    # Analysis parameters
    analysis_cfg = cfg.get("analysis", {})
    d_min = analysis_cfg.get("d_min", 0.0)
    d_max = analysis_cfg.get("d_max", 5.0)
    d_step = analysis_cfg.get("d_step", 0.1)
    
    noise_intensities = np.arange(d_min, d_max + d_step, d_step).tolist()
    
    # Get noise and mechanism types from config
    noise_type = cfg.sr.noise.get("_target_", "").split(".")[-1].lower()
    if "gaussian" in noise_type:
        noise_type = "gaussian"
    elif "uniform" in noise_type:
        noise_type = "uniform"
    elif "alpha" in noise_type:
        noise_type = "alpha_stable"
    elif "poisson" in noise_type:
        noise_type = "poisson"
    elif "colored" in noise_type:
        noise_type = "colored"
    else:
        noise_type = "gaussian"  # default
    
    mechanism_type = cfg.sr.mechanism.get("_target_", "").split(".")[-1].lower()
    if "additive" in mechanism_type:
        mechanism_type = "additive"
    elif "bistable" in mechanism_type:
        mechanism_type = "bistable"
    elif "tristable" in mechanism_type:
        mechanism_type = "tristable"
    else:
        mechanism_type = "additive"  # default
    
    log.info(f"Analysis configuration: D from {d_min} to {d_max}, step {d_step}")
    log.info(f"Noise type: {noise_type}, Mechanism: {mechanism_type}")
    
    # Output directory
    output_dir = analysis_cfg.get("output_dir", "./results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run noise sweep analysis
    log.info("Starting noise intensity sweep analysis...")
    df_results = analyze_noise_sweep(
        cfg=cfg,
        model=model,
        datamodule=datamodule,
        noise_intensities=noise_intensities,
        noise_type=noise_type,
        mechanism_type=mechanism_type,
        device=device
    )
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, "noise_sweep_results.csv")
    df_results.to_csv(csv_path, index=False)
    log.info(f"Results saved to {csv_path}")
    
    # Find optimal noise intensity
    optimal_idx = df_results["accuracy"].idxmax()
    optimal_d = df_results.loc[optimal_idx, "noise_intensity"]
    optimal_acc = df_results.loc[optimal_idx, "accuracy"]
    optimal_f1 = df_results.loc[optimal_idx, "f1_score"]
    optimal_kappa = df_results.loc[optimal_idx, "kappa"]
    
    log.info(f"Optimal noise intensity: D={optimal_d:.2f}")
    log.info(f"  Accuracy: {optimal_acc:.4f}")
    log.info(f"  F1 Score: {optimal_f1:.4f}")
    log.info(f"  Kappa: {optimal_kappa:.4f}")
    
    # Save confusion matrices for key intensities
    key_intensities = [0.0, optimal_d, d_max]
    # Add some intermediate points
    key_intensities.extend([d_max * 0.25, d_max * 0.5, d_max * 0.75])
    key_intensities = sorted(list(set(key_intensities)))
    
    log.info(f"Computing confusion matrices for D = {key_intensities}")
    conf_mat_dir = os.path.join(output_dir, "confusion_matrices")
    save_confusion_matrices(
        cfg=cfg,
        model=model,
        datamodule=datamodule,
        noise_intensities=key_intensities,
        output_dir=conf_mat_dir,
        noise_type=noise_type,
        mechanism_type=mechanism_type,
        device=device
    )
    
    # Save analysis summary
    summary = {
        "checkpoint": ckpt_path,
        "noise_type": noise_type,
        "mechanism": mechanism_type,
        "d_range": [d_min, d_max],
        "d_step": d_step,
        "optimal_d": float(optimal_d),
        "optimal_accuracy": float(optimal_acc),
        "optimal_f1": float(optimal_f1),
        "optimal_kappa": float(optimal_kappa)
    }
    
    summary_path = os.path.join(output_dir, "analysis_summary.yaml")
    with open(summary_path, "w") as f:
        import yaml
        yaml.dump(summary, f, default_flow_style=False)
    
    log.info(f"Analysis complete. Summary saved to {summary_path}")
    
    return df_results


if __name__ == "__main__":
    main()