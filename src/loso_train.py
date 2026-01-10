"""
LOSO 交叉验证训练脚本

使用方式:
  # 单次训练（默认 fold 1）
  python src/loso_train.py dataset=bci2a_loso
  
  # 指定特定折
  python src/loso_train.py dataset=bci2a_loso dataset.fold_id=2
  
  # 多折批量运行（Hydra multirun）
  python src/loso_train.py --multirun dataset=bci2a_loso dataset.fold_id=1,2,3
  
  # 结合噪声强度遍历
  python src/loso_train.py --multirun \
      dataset=bci2a_loso \
      dataset.fold_id=1,2,3 \
      sr.mechanism.intensity=0.1,0.5,1.0
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 项目根目录
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
    """LOSO 训练主函数"""
    
    # 1. 设置随机种子
    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)
    
    # 2. 实例化 DataModule（会打印折数分配）
    log.info(f"Instantiating LOSO DataModule <{cfg.dataset._target_}>")
    log.info(f"  Fold: {cfg.dataset.fold_id} / {cfg.dataset.n_folds}")
    datamodule = hydra.utils.instantiate(cfg.dataset)
    
    # 3. 实例化模型
    log.info(f"Instantiating Network <{cfg.model._target_}>")
    net = hydra.utils.instantiate(cfg.model)
    
    # 4. 实例化 SR 层
    log.info(f"Instantiating SR Layer: {cfg.sr.mechanism._target_}")
    noise_source = hydra.utils.instantiate(cfg.sr.noise)
    sr_layer = hydra.utils.instantiate(cfg.sr.mechanism, noise_source=noise_source)
    
    # 5. 创建 Lightning Module
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
    
    # 6. 实例化 Callbacks
    callbacks = []
    if "callbacks" in cfg:
        log.info("Instantiating Callbacks...")
        callbacks = instantiate_callbacks(cfg.callbacks)
        log.info(f"Loaded {len(callbacks)} callbacks: {[type(cb).__name__ for cb in callbacks]}")
    
    # 7. 实例化 Trainer
    log.info("Instantiating Trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    
    # 8. 训练
    log.info("Starting LOSO Training...")
    trainer.fit(model=model, datamodule=datamodule)
    
    # 9. 保存训练信息
    fold_info = datamodule.get_fold_info()
    training_info = {
        # 数据集信息
        "dataset_name": cfg.dataset.name,
        "n_folds": cfg.dataset.n_folds,
        "fold_id": cfg.dataset.fold_id,
        "test_subjects": fold_info["test_subjects"],
        "train_subjects": fold_info["train_subjects"],
        
        # 模型信息
        "model_name": cfg.model.name,
        
        # SR 配置
        "mechanism_name": cfg.sr.mechanism.name,
        "noise_name": cfg.sr.noise.name,
        "intensity": cfg.sr.mechanism.intensity,
        
        # 训练配置
        "max_epochs": cfg.trainer.max_epochs,
        "precision": cfg.trainer.get("precision", "32"),
        "batch_size": cfg.dataset.batch_size,
        
        # 训练状态
        "early_stopped": False,
        "stopped_epoch": trainer.current_epoch,
        
        # 时间戳
        "timestamp": datetime.now().isoformat(),
        
        # 输出路径
        "output_dir": str(Path.cwd()),
    }
    
    # 检查 Early Stopping
    for cb in callbacks:
        if hasattr(cb, 'stopped_epoch') and cb.stopped_epoch > 0:
            training_info["early_stopped"] = True
            training_info["stopped_epoch"] = cb.stopped_epoch
            log.info(f"Early stopping triggered at epoch {cb.stopped_epoch}")
            break
    
    # 保存到日志目录
    log_dir = trainer.log_dir or "lightning_logs"
    info_path = os.path.join(log_dir, "training_info.json")
    try:
        with open(info_path, "w") as f:
            json.dump(training_info, f, indent=2)
        log.info(f"Training info saved to {info_path}")
    except Exception as e:
        log.warning(f"Could not save training info: {e}")
    
    # 10. 测试
    log.info("Starting Testing...")
    # 使用 weights_only=False 解决 PyTorch 2.6+ checkpoint 加载问题
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best", weights_only=False)
    
    return log_dir


if __name__ == "__main__":
    main()