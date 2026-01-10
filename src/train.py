import os
import sys
import logging
from typing import List, Optional

# Allow imports from the root directory - must be done before other imports
# Use absolute path to handle Hydra's working directory changes
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import omegaconf
import omegaconf.base  # 需要导入这个以访问 ContainerMetadata
# 设置 CUDA 矩阵乘法精度，避免警告并提升性能
torch.set_float32_matmul_precision('medium')

# 允许加载 OmegaConf 配置对象（PyTorch 2.6+ 默认 weights_only=True）
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata  # 新增：解决 WeightsUnpickler error
])

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback

from src.models.lit_model import LitEEGModel

log = logging.getLogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    从配置中实例化所有 callbacks。
    
    Args:
        callbacks_cfg: callbacks 配置字典
        
    Returns:
        Callback 对象列表
    """
    callbacks: List[Callback] = []
    
    if not callbacks_cfg:
        log.warning("No callbacks configuration found!")
        return callbacks
    
    for cb_name, cb_conf in callbacks_cfg.items():
        if cb_conf is not None and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            try:
                callback = hydra.utils.instantiate(cb_conf)
                callbacks.append(callback)
            except Exception as e:
                log.error(f"Error instantiating callback {cb_name}: {e}")
                raise
    
    return callbacks


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Set seed
    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)

    # Instantiate DataModule
    log.info(f"Instantiating DataModule <{cfg.dataset._target_}>")
    datamodule = hydra.utils.instantiate(cfg.dataset)

    # Instantiate Model (Network)
    log.info(f"Instantiating Network <{cfg.model._target_}>")
    net = hydra.utils.instantiate(cfg.model)

    # Instantiate SR Layer
    log.info(f"Instantiating SR Layer Mechanism <{cfg.sr.mechanism._target_}> with Noise <{cfg.sr.noise._target_}>")
    # 1. Noise Source
    noise_source = hydra.utils.instantiate(cfg.sr.noise)
    # 2. Mechanism (inject noise source)
    sr_layer = hydra.utils.instantiate(cfg.sr.mechanism, noise_source=noise_source)

    # Instantiate Lightning Module
    log.info("Instantiating Lightning Module")
    # 提取配置名称用于保存到 hparams
    model_name = cfg.model.get("_target_", "").split(".")[-1]
    datamodule_name = cfg.dataset.get("_target_", "").split(".")[-1]
    mechanism_name = cfg.sr.mechanism.get("_target_", "").split(".")[-1]
    noise_name = cfg.sr.noise.get("_target_", "").split(".")[-1]
    
    model = LitEEGModel(
        net=net,
        sr_layer=sr_layer,
        model_name=model_name,
        datamodule_name=datamodule_name,  # 使用 datamodule_name 避免与 DataModule.dataset_name 冲突
        mechanism_name=mechanism_name,
        noise_name=noise_name
    )

    # Instantiate Callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        log.info("Instantiating Callbacks...")
        callbacks = instantiate_callbacks(cfg.callbacks)
        log.info(f"Loaded {len(callbacks)} callbacks: {[type(cb).__name__ for cb in callbacks]}")
    
    # Instantiate Trainer with callbacks
    log.info("Instantiating Trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

    # Train
    log.info("Starting Training...")
    trainer.fit(model=model, datamodule=datamodule)
    
    # 获取训练结果信息
    early_stopped = False
    stopped_epoch = trainer.current_epoch
    if callbacks:
        for cb in callbacks:
            if hasattr(cb, 'stopped_epoch') and cb.stopped_epoch > 0:
                early_stopped = True
                stopped_epoch = cb.stopped_epoch
                log.info(f"Early stopping triggered at epoch {stopped_epoch}")
                break
    
    # 保存训练元信息用于可视化
    log_dir = trainer.log_dir if trainer.log_dir else "lightning_logs"
    training_info = {
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "max_epochs": cfg.trainer.max_epochs,
        "model_name": model_name,
        "datamodule_name": datamodule_name,
        "mechanism_name": mechanism_name,
        "noise_name": noise_name
    }
    
    # 保存训练信息到日志目录
    import json
    info_path = os.path.join(log_dir, "training_info.json")
    try:
        with open(info_path, "w") as f:
            json.dump(training_info, f, indent=2)
        log.info(f"Training info saved to {info_path}")
    except Exception as e:
        log.warning(f"Could not save training info: {e}")

    # Test
    log.info("Starting Testing...")
    # Use the best checkpoint from training
    # 使用 weights_only=False 解决 PyTorch 2.6+ checkpoint 加载问题
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best", weights_only=False)
    
    # 返回日志目录路径供后续使用
    return log_dir


if __name__ == "__main__":
    main()