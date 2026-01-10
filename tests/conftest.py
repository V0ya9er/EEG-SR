"""
pytest 配置和共享 fixtures
"""
import os
import sys
from pathlib import Path

import pytest

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@pytest.fixture
def project_root():
    """返回项目根目录"""
    return PROJECT_ROOT


@pytest.fixture
def mock_gpu_count(monkeypatch):
    """模拟 GPU 数量的 fixture 工厂"""
    def _mock(count: int):
        import torch
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    return _mock


@pytest.fixture
def temp_config_file(tmp_path):
    """创建临时配置文件"""
    config_file = tmp_path / "config.json"
    return config_file


@pytest.fixture
def sample_experiment_config():
    """创建示例实验配置"""
    from menu.config import ExperimentConfig
    return ExperimentConfig(
        model="eegnet",
        dataset="bci2a_loso",
        mechanism="additive",
        noise_type="gaussian",
        epochs=10,
        batch_size=32,
        loso_n_folds=9,
        loso_fold_id=1
    )


@pytest.fixture
def mock_training_info():
    """模拟 training_info.json 内容"""
    return {
        "dataset_name": "bci2a",
        "n_folds": 9,
        "fold_id": 1,
        "test_subjects": [1],
        "train_subjects": [2, 3, 4, 5, 6, 7, 8, 9],
        "model_name": "Conformer",
        "mechanism_name": "AdditiveSR",
        "noise_name": "GaussianNoise",
        "intensity": 0.5,
        "max_epochs": 50,
        "batch_size": 128,
        "early_stopped": False,
        "stopped_epoch": 50,
    }