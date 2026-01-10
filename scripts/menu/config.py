#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验配置管理模块

提供 ExperimentConfig 数据类和配置持久化功能。
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# ============================================================================
# 配置选项定义
# ============================================================================

MODELS = ["eegnet", "conformer"]
DATASETS = ["bci2a", "bci2b"]
DATASETS_LOSO = ["bci2a_loso", "bci2b_loso"]
MECHANISMS = ["additive", "bistable", "tristable"]
NOISES = ["gaussian", "uniform", "alpha_stable", "poisson", "colored"]

# 显示名称映射
MODEL_NAMES = {
    "eegnet": "EEGNet",
    "conformer": "EEG Conformer"
}

DATASET_NAMES = {
    "bci2a": "BCI Competition IV 2a (4类运动想象)",
    "bci2b": "BCI Competition IV 2b (2类运动想象)"
}

DATASET_LOSO_NAMES = {
    "bci2a_loso": "BCI Competition IV 2a LOSO (4类，9折)",
    "bci2b_loso": "BCI Competition IV 2b LOSO (2类，9折)"
}

MECHANISM_NAMES = {
    "additive": "Additive (加性 SR)",
    "bistable": "Bistable (双稳态 SR)",
    "tristable": "Tristable (三稳态 SR)"
}

NOISE_NAMES = {
    "gaussian": "Gaussian (高斯噪声)",
    "uniform": "Uniform (均匀噪声)",
    "alpha_stable": "Alpha-Stable (Alpha稳定噪声)",
    "poisson": "Poisson (泊松噪声)",
    "colored": "Colored (有色噪声/粉噪声)"
}

# 可配置参数定义
CONFIGURABLE_PARAMS = {
    "epochs": {
        "display_name": "Epochs",
        "description": "最大训练轮数",
        "type": int,
        "default": 50,
        "range": (1, 1000)
    },
    "batch_size": {
        "display_name": "Batch Size",
        "description": "每批样本数",
        "type": int,
        "default": 32,
        "choices": [8, 16, 32, 64, 128, 256]
    },
    "learning_rate": {
        "display_name": "Learning Rate",
        "description": "初始学习率",
        "type": float,
        "default": 0.001,
        "range": (1e-6, 1.0)
    },
    "early_stopping_patience": {
        "display_name": "Early Stopping",
        "description": "提前停止耐心值",
        "type": int,
        "default": 10,
        "range": (1, 100)
    }
}


# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    
    # 模型与数据 - LOSO 优先设计
    model: str = "conformer"
    dataset: str = "bci2a_loso"  # 默认使用 LOSO 数据集
    
    # 随机共振设置
    mechanism: str = "additive"
    noise_type: str = "gaussian"
    
    # 训练参数
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    
    # 硬件设置
    gpu_id: int = 0
    use_cpu: bool = False
    
    # LOSO 设置
    loso_dataset: str = "bci2a_loso"  # LOSO 数据集
    loso_n_folds: int = 9              # 折数 (1-9)
    loso_fold_id: int = 1              # 当前运行的折 (1-n_folds)
    loso_run_all: bool = False         # 是否运行所有折
    
    # 配置文件路径
    _config_file: Path = field(default_factory=lambda: Path("config.json"), repr=False)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保配置文件路径是 Path 对象
        if isinstance(self._config_file, str):
            self._config_file = Path(self._config_file)
    
    @classmethod
    def load(cls, config_file: Optional[Union[str, Path]] = None) -> "ExperimentConfig":
        """
        从文件加载配置
        
        Args:
            config_file: 配置文件路径，默认为 config.json
            
        Returns:
            ExperimentConfig 实例
        """
        if config_file is None:
            config_file = Path("config.json")
        elif isinstance(config_file, str):
            config_file = Path(config_file)
            
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 移除私有字段
                data.pop("_config_file", None)
                config = cls(**data)
                config._config_file = config_file
                return config
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"警告: 加载配置文件失败 ({e})，使用默认配置")
        
        config = cls()
        config._config_file = config_file
        return config
    
    def save(self, config_file: Optional[Union[str, Path]] = None) -> bool:
        """
        保存配置到文件
        
        Args:
            config_file: 配置文件路径，默认使用加载时的路径
            
        Returns:
            是否保存成功
        """
        if config_file is None:
            config_file = self._config_file
        elif isinstance(config_file, str):
            config_file = Path(config_file)
            
        try:
            # 转换为字典，排除私有字段
            data = asdict(self)
            data.pop("_config_file", None)
            
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except (IOError, OSError) as e:
            print(f"警告: 保存配置文件失败 ({e})")
            return False
    
    def reset_to_defaults(self):
        """重置为默认值 - LOSO 优先"""
        self.model = "conformer"
        self.dataset = "bci2a_loso"  # 默认使用 LOSO 数据集
        self.mechanism = "additive"
        self.noise_type = "gaussian"
        self.epochs = 50
        self.batch_size = 32
        self.learning_rate = 0.001
        self.early_stopping_patience = 10
        self.gpu_id = 0
        self.use_cpu = False
        self.loso_dataset = "bci2a_loso"
        self.loso_n_folds = 9
        self.loso_fold_id = 1
        self.loso_run_all = False
    
    def get_experiment_name(self) -> str:
        """获取实验名称"""
        return f"{self.model}_{self.dataset}_{self.mechanism}_{self.noise_type}"
    
    def get_output_dir(self) -> str:
        """获取输出目录"""
        return f"./results/{self.get_experiment_name()}"
    
    def get_model_display_name(self) -> str:
        """获取模型显示名称"""
        return MODEL_NAMES.get(self.model, self.model)
    
    def get_dataset_display_name(self) -> str:
        """获取数据集显示名称"""
        return DATASET_NAMES.get(self.dataset, 
               DATASET_LOSO_NAMES.get(self.dataset, self.dataset))
    
    def get_mechanism_display_name(self) -> str:
        """获取机制显示名称"""
        return MECHANISM_NAMES.get(self.mechanism, self.mechanism)
    
    def get_noise_display_name(self) -> str:
        """获取噪声显示名称"""
        return NOISE_NAMES.get(self.noise_type, self.noise_type)
    
    def validate(self) -> List[str]:
        """
        验证配置是否有效
        
        Returns:
            错误信息列表，空列表表示配置有效
        """
        errors = []
        
        if self.model not in MODELS:
            errors.append(f"无效的模型: {self.model}")
        
        if self.dataset not in DATASETS + DATASETS_LOSO:
            errors.append(f"无效的数据集: {self.dataset}")
        
        if self.mechanism not in MECHANISMS:
            errors.append(f"无效的机制: {self.mechanism}")
        
        if self.noise_type not in NOISES:
            errors.append(f"无效的噪声类型: {self.noise_type}")
        
        if not (1 <= self.epochs <= 1000):
            errors.append(f"Epochs 必须在 1-1000 之间: {self.epochs}")
        
        if self.batch_size not in [8, 16, 32, 64, 128, 256]:
            errors.append(f"无效的 Batch Size: {self.batch_size}")
        
        if not (1e-6 <= self.learning_rate <= 1.0):
            errors.append(f"Learning Rate 必须在 1e-6 到 1.0 之间: {self.learning_rate}")
        
        if not (1 <= self.early_stopping_patience <= 100):
            errors.append(f"Early Stopping Patience 必须在 1-100 之间: {self.early_stopping_patience}")
        
        if not self.use_cpu and self.gpu_id < 0:
            errors.append(f"无效的 GPU ID: {self.gpu_id}")
        
        return errors
    
    def to_hydra_overrides(self) -> List[str]:
        """
        转换为 Hydra 命令行覆盖参数
        
        Returns:
            Hydra 覆盖参数列表
        """
        overrides = [
            f"model={self.model}",
            f"dataset={self.dataset}",
            f"sr/mechanism={self.mechanism}",
            f"sr/noise={self.noise_type}",
            f"trainer.max_epochs={self.epochs}",
        ]
        
        # 添加可选参数（如果支持）
        # overrides.append(f"datamodule.batch_size={self.batch_size}")
        # overrides.append(f"optimizer.lr={self.learning_rate}")
        
        return overrides
    
    def summary(self) -> Dict[str, str]:
        """
        获取配置摘要
        
        Returns:
            配置摘要字典
        """
        return {
            "模型": self.get_model_display_name(),
            "数据集": self.get_dataset_display_name(),
            "SR 机制": self.get_mechanism_display_name(),
            "噪声类型": self.get_noise_display_name(),
            "训练轮数": str(self.epochs),
            "批大小": str(self.batch_size),
            "学习率": str(self.learning_rate),
            "GPU": "CPU 模式" if self.use_cpu else f"#{self.gpu_id}",
        }
    
    def loso_summary(self) -> Dict[str, str]:
        """
        获取 LOSO 配置摘要
        
        Returns:
            LOSO 配置摘要字典
        """
        # 计算每折留出的被试数
        subjects_per_fold = 9 // self.loso_n_folds
        remainder = 9 % self.loso_n_folds
        
        if self.loso_run_all:
            fold_info = f"全部 {self.loso_n_folds} 折"
        else:
            fold_info = f"第 {self.loso_fold_id} 折 (共 {self.loso_n_folds} 折)"
        
        return {
            "LOSO 数据集": DATASET_LOSO_NAMES.get(self.loso_dataset, self.loso_dataset),
            "折数 (n_folds)": str(self.loso_n_folds),
            "每折留出被试": f"约 {subjects_per_fold}" + (f"-{subjects_per_fold+1}" if remainder > 0 else "") + " 个",
            "运行折": fold_info,
            "模型": self.get_model_display_name(),
            "SR 机制": self.get_mechanism_display_name(),
            "噪声类型": self.get_noise_display_name(),
            "训练轮数": str(self.epochs),
            "GPU": "CPU 模式" if self.use_cpu else f"#{self.gpu_id}",
        }


def validate_param_value(param_name: str, value: Any) -> tuple:
    """
    验证参数值
    
    Args:
        param_name: 参数名
        value: 参数值
        
    Returns:
        (is_valid, error_message or converted_value)
    """
    if param_name not in CONFIGURABLE_PARAMS:
        return False, f"未知参数: {param_name}"
    
    param_def = CONFIGURABLE_PARAMS[param_name]
    param_type = param_def["type"]
    
    # 类型转换
    try:
        if param_type == int:
            converted = int(value)
        elif param_type == float:
            converted = float(value)
        else:
            converted = value
    except (ValueError, TypeError):
        return False, f"无法转换为 {param_type.__name__} 类型"
    
    # 范围检查
    if "range" in param_def:
        min_val, max_val = param_def["range"]
        if not (min_val <= converted <= max_val):
            return False, f"值必须在 {min_val} 到 {max_val} 之间"
    
    # 选项检查
    if "choices" in param_def:
        if converted not in param_def["choices"]:
            return False, f"值必须是以下选项之一: {param_def['choices']}"
    
    return True, converted