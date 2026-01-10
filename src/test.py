import os
import sys
import logging

# Allow imports from the root directory - must be done before other imports
# Use absolute path to handle Hydra's working directory changes
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import omegaconf
import omegaconf.base  # 需要导入这个以访问 ContainerMetadata

# 允许加载 OmegaConf 配置对象（PyTorch 2.6+ 默认 weights_only=True）
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata  # 新增：解决 WeightsUnpickler error
])

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from src.models.lit_model import LitEEGModel

log = logging.getLogger(__name__)

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
    noise_source = hydra.utils.instantiate(cfg.sr.noise)
    sr_layer = hydra.utils.instantiate(cfg.sr.mechanism, noise_source=noise_source)

    # Checkpoint path
    ckpt_path = cfg.get("ckpt_path")
    
    if not ckpt_path:
        log.error("Please provide a checkpoint path using +ckpt_path=/path/to/checkpoint.ckpt")
        return

    log.info(f"Loading model from checkpoint: {ckpt_path}")
    # We need to pass net and sr_layer because they are not saved in hparams (ignored)
    # 使用 weights_only=False 解决 PyTorch 2.6+ checkpoint 加载问题
    model = LitEEGModel.load_from_checkpoint(ckpt_path, net=net, sr_layer=sr_layer, weights_only=False)

    # Instantiate Trainer
    log.info("Instantiating Trainer")
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Test
    log.info("Starting Testing...")
    # 注意：这里模型已经加载完毕，不需要 ckpt_path，所以不需要 weights_only 参数
    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()