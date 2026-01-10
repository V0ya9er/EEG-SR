"""
测试 scripts/menu/config.py 模块
"""
import json
import pytest
from pathlib import Path

from menu.config import (
    ExperimentConfig,
    validate_param_value,
    MODELS, DATASETS, DATASETS_LOSO, MECHANISMS, NOISES
)


class TestExperimentConfig:
    """ExperimentConfig 类测试"""
    
    def test_default_values(self):
        """测试默认值是否符合 LOSO 优先设计"""
        config = ExperimentConfig()
        
        # 核心：默认应该是 LOSO 数据集
        assert config.dataset == "bci2a_loso" or config.dataset == "bci2a"
        assert config.model == "conformer"
        assert config.mechanism == "additive"
        assert config.noise_type == "gaussian"
        assert config.loso_n_folds == 9
        assert config.loso_fold_id == 1
    
    def test_save_and_load(self, temp_config_file):
        """测试配置保存和加载"""
        # 创建配置
        config = ExperimentConfig(
            model="eegnet",
            dataset="bci2b_loso",
            mechanism="bistable",
            noise_type="uniform",
            epochs=100,
            loso_n_folds=3,
            loso_fold_id=2
        )
        config._config_file = temp_config_file
        
        # 保存
        assert config.save() is True
        assert temp_config_file.exists()
        
        # 加载
        loaded_config = ExperimentConfig.load(temp_config_file)
        assert loaded_config.model == "eegnet"
        assert loaded_config.dataset == "bci2b_loso"
        assert loaded_config.mechanism == "bistable"
        assert loaded_config.noise_type == "uniform"
        assert loaded_config.epochs == 100
        assert loaded_config.loso_n_folds == 3
        assert loaded_config.loso_fold_id == 2
    
    def test_reset_to_defaults(self):
        """测试重置为默认值"""
        config = ExperimentConfig(
            model="eegnet",
            epochs=999,
            loso_n_folds=3
        )
        
        config.reset_to_defaults()
        
        assert config.model == "conformer"
        assert config.epochs == 50
        assert config.loso_n_folds == 9
    
    def test_validate_valid_config(self):
        """测试有效配置的验证"""
        config = ExperimentConfig(
            model="eegnet",
            dataset="bci2a",
            mechanism="additive",
            noise_type="gaussian",
            epochs=50,
            batch_size=32,
            learning_rate=0.001
        )
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validate_invalid_model(self):
        """测试无效模型的验证"""
        config = ExperimentConfig(model="invalid_model")
        errors = config.validate()
        assert any("无效的模型" in e for e in errors)
    
    def test_validate_invalid_epochs(self):
        """测试无效 epochs 的验证"""
        config = ExperimentConfig(epochs=9999)
        errors = config.validate()
        assert any("Epochs" in e for e in errors)
    
    def test_validate_invalid_batch_size(self):
        """测试无效 batch_size 的验证"""
        config = ExperimentConfig(batch_size=100)  # 不在允许列表中
        errors = config.validate()
        assert any("Batch Size" in e for e in errors)
    
    def test_get_experiment_name(self):
        """测试实验名称生成"""
        config = ExperimentConfig(
            model="conformer",
            dataset="bci2a_loso",
            mechanism="additive",
            noise_type="gaussian"
        )
        
        name = config.get_experiment_name()
        assert name == "conformer_bci2a_loso_additive_gaussian"
    
    def test_get_output_dir(self):
        """测试输出目录生成"""
        config = ExperimentConfig(
            model="eegnet",
            dataset="bci2a",
            mechanism="bistable",
            noise_type="uniform"
        )
        
        output_dir = config.get_output_dir()
        assert output_dir == "./results/eegnet_bci2a_bistable_uniform"
    
    def test_to_hydra_overrides(self):
        """测试 Hydra 覆盖参数生成"""
        config = ExperimentConfig(
            model="conformer",
            dataset="bci2a_loso",
            mechanism="additive",
            noise_type="gaussian",
            epochs=100
        )
        
        overrides = config.to_hydra_overrides()
        
        assert "model=conformer" in overrides
        assert "dataset=bci2a_loso" in overrides
        assert "sr/mechanism=additive" in overrides
        assert "sr/noise=gaussian" in overrides
        assert "trainer.max_epochs=100" in overrides
    
    def test_summary(self):
        """测试配置摘要"""
        config = ExperimentConfig()
        summary = config.summary()
        
        assert "模型" in summary
        assert "数据集" in summary
        assert "SR 机制" in summary
        assert "噪声类型" in summary
    
    def test_loso_summary(self):
        """测试 LOSO 配置摘要"""
        config = ExperimentConfig(
            loso_dataset="bci2a_loso",
            loso_n_folds=9,
            loso_fold_id=3,
            loso_run_all=False
        )
        
        summary = config.loso_summary()
        
        assert "LOSO 数据集" in summary
        # 键名包含 "折数"，可能是 "折数 (n_folds)" 或类似格式
        assert any("折数" in key for key in summary.keys())
        assert any("运行" in key for key in summary.keys())


class TestValidateParamValue:
    """validate_param_value 函数测试"""
    
    def test_valid_epochs(self):
        """测试有效 epochs 验证"""
        valid, result = validate_param_value("epochs", "100")
        assert valid is True
        assert result == 100
    
    def test_invalid_epochs_range(self):
        """测试超出范围的 epochs"""
        valid, result = validate_param_value("epochs", "9999")
        assert valid is False
        assert "范围" in result or "之间" in result
    
    def test_invalid_epochs_type(self):
        """测试无效类型的 epochs"""
        valid, result = validate_param_value("epochs", "abc")
        assert valid is False
    
    def test_valid_learning_rate(self):
        """测试有效学习率"""
        valid, result = validate_param_value("learning_rate", "0.001")
        assert valid is True
        assert result == 0.001
    
    def test_valid_batch_size(self):
        """测试有效 batch_size"""
        valid, result = validate_param_value("batch_size", "64")
        assert valid is True
        assert result == 64
    
    def test_invalid_batch_size_choice(self):
        """测试不在选项中的 batch_size"""
        valid, result = validate_param_value("batch_size", "100")
        assert valid is False
    
    def test_unknown_param(self):
        """测试未知参数"""
        valid, result = validate_param_value("unknown_param", "value")
        assert valid is False
        assert "未知参数" in result


class TestConfigConstants:
    """配置常量测试"""
    
    def test_models_not_empty(self):
        """测试模型列表非空"""
        assert len(MODELS) > 0
        assert "eegnet" in MODELS
        assert "conformer" in MODELS
    
    def test_datasets_not_empty(self):
        """测试数据集列表非空"""
        assert len(DATASETS) > 0
        assert "bci2a" in DATASETS
        assert "bci2b" in DATASETS
    
    def test_datasets_loso_not_empty(self):
        """测试 LOSO 数据集列表非空"""
        assert len(DATASETS_LOSO) > 0
        assert "bci2a_loso" in DATASETS_LOSO
        assert "bci2b_loso" in DATASETS_LOSO
    
    def test_mechanisms_not_empty(self):
        """测试机制列表非空"""
        assert len(MECHANISMS) > 0
        assert "additive" in MECHANISMS
    
    def test_noises_not_empty(self):
        """测试噪声类型列表非空"""
        assert len(NOISES) > 0
        assert "gaussian" in NOISES