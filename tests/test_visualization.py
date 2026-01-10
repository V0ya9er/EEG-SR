"""
测试可视化模块
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json
import os


class TestLoadTrainingInfo:
    """测试 load_training_info 函数"""
    
    def test_load_from_direct_path(self, tmp_path, mock_training_info):
        """测试从直接路径加载"""
        from src.utils.plotting import load_training_info
        
        # 创建 training_info.json
        info_path = tmp_path / "training_info.json"
        with open(info_path, "w") as f:
            json.dump(mock_training_info, f)
        
        # 加载
        result = load_training_info(str(tmp_path))
        
        assert result is not None
        assert result["model_name"] == "Conformer"
        assert result["n_folds"] == 9
        assert result["fold_id"] == 1
    
    def test_load_from_subdirectory(self, tmp_path, mock_training_info):
        """测试从子目录加载"""
        from src.utils.plotting import load_training_info
        
        # 在子目录创建 training_info.json
        sub_dir = tmp_path / "fold_1"
        sub_dir.mkdir()
        info_path = sub_dir / "training_info.json"
        with open(info_path, "w") as f:
            json.dump(mock_training_info, f)
        
        # 加载
        result = load_training_info(str(tmp_path))
        
        assert result is not None
    
    def test_load_nonexistent(self, tmp_path, monkeypatch):
        """测试加载不存在的文件"""
        from src.utils.plotting import load_training_info
        
        # 阻止函数回退到全局 lightning_logs 目录
        # 通过将当前工作目录临时改为 tmp_path，使其找不到 lightning_logs
        nonexistent_dir = tmp_path / "nonexistent_subdir"
        monkeypatch.chdir(tmp_path)
        
        result = load_training_info(str(nonexistent_dir))
        assert result is None


class TestFormatTrainingInfoText:
    """测试 format_training_info_text 函数"""
    
    def test_format_full_info(self, mock_training_info):
        """测试完整信息格式化"""
        from src.utils.plotting import format_training_info_text
        
        result = format_training_info_text(mock_training_info)
        
        assert "Conformer" in result
        assert "bci2a" in result
        assert "1/9" in result
    
    def test_format_minimal_info(self):
        """测试最小信息格式化"""
        from src.utils.plotting import format_training_info_text
        
        minimal_info = {"model_name": "EEGNet"}
        result = format_training_info_text(minimal_info)
        
        assert "EEGNet" in result
    
    def test_format_with_early_stopping(self, mock_training_info):
        """测试包含早停信息"""
        from src.utils.plotting import format_training_info_text
        
        mock_training_info["early_stopped"] = True
        mock_training_info["stopped_epoch"] = 35
        
        result = format_training_info_text(mock_training_info)
        
        assert "Early Stopped" in result
        assert "35" in result


class TestCreateEnhancedTitle:
    """测试 create_enhanced_title 函数"""
    
    def test_enhanced_title_with_info(self, mock_training_info):
        """测试带信息的增强标题"""
        from src.utils.plotting import create_enhanced_title
        
        result = create_enhanced_title("Base Title", mock_training_info)
        
        assert "Base Title" in result
        assert "1/9" in result or "Fold" in result
    
    def test_enhanced_title_without_info(self):
        """测试不带信息的增强标题"""
        from src.utils.plotting import create_enhanced_title
        
        result = create_enhanced_title("Base Title", None)
        
        assert result == "Base Title"
    
    def test_enhanced_title_no_fold(self, mock_training_info):
        """测试不包含折数的标题"""
        from src.utils.plotting import create_enhanced_title
        
        result = create_enhanced_title("Base Title", mock_training_info, include_fold=False)
        
        assert "Base Title" in result


class TestPlotNoiseVsMetrics:
    """测试 plot_noise_vs_metrics 函数"""
    
    def test_plot_basic(self, tmp_path):
        """测试基本绘图"""
        from src.utils.plotting import plot_noise_vs_metrics
        
        # 创建测试数据
        df = pd.DataFrame({
            "noise_intensity": np.linspace(0, 5, 10),
            "accuracy": np.random.uniform(0.5, 0.9, 10),
            "f1_score": np.random.uniform(0.5, 0.9, 10),
            "kappa": np.random.uniform(0.3, 0.8, 10)
        })
        
        output_path = tmp_path / "test_plot.png"
        fig = plot_noise_vs_metrics(df, output_path=str(output_path))
        
        assert output_path.exists()
        assert fig is not None
    
    def test_plot_with_training_info(self, tmp_path, mock_training_info):
        """测试带训练信息的绘图"""
        from src.utils.plotting import plot_noise_vs_metrics
        
        df = pd.DataFrame({
            "noise_intensity": np.linspace(0, 5, 10),
            "accuracy": np.random.uniform(0.5, 0.9, 10),
            "f1_score": np.random.uniform(0.5, 0.9, 10),
            "kappa": np.random.uniform(0.3, 0.8, 10)
        })
        
        output_path = tmp_path / "test_plot_with_info.png"
        fig = plot_noise_vs_metrics(
            df, 
            output_path=str(output_path),
            training_info=mock_training_info
        )
        
        assert output_path.exists()
        assert fig is not None


class TestPlotConfusionMatrix:
    """测试 plot_confusion_matrix 函数"""
    
    def test_plot_basic(self, tmp_path):
        """测试基本混淆矩阵绘图"""
        from src.utils.plotting import plot_confusion_matrix
        
        conf_matrix = np.array([
            [50, 10, 5, 2],
            [8, 45, 7, 3],
            [6, 5, 48, 4],
            [3, 4, 6, 52]
        ])
        
        output_path = tmp_path / "test_confusion.png"
        fig = plot_confusion_matrix(
            conf_matrix,
            class_names=["Left", "Right", "Foot", "Tongue"],
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert fig is not None
    
    def test_plot_normalized(self, tmp_path):
        """测试归一化混淆矩阵"""
        from src.utils.plotting import plot_confusion_matrix
        
        conf_matrix = np.array([[80, 20], [15, 85]])
        
        output_path = tmp_path / "test_confusion_norm.png"
        fig = plot_confusion_matrix(
            conf_matrix,
            normalize=True,
            output_path=str(output_path)
        )
        
        assert output_path.exists()


class TestPlotTrainingCurves:
    """测试 plot_training_curves 函数"""
    
    def test_plot_with_metrics_csv(self, tmp_path):
        """测试从 metrics.csv 绘制训练曲线"""
        from src.utils.plotting import plot_training_curves
        
        # 创建模拟的 metrics.csv
        metrics_data = {
            "epoch": list(range(10)) * 2,
            "step": list(range(20)),
            "train_loss": [1.5 - i*0.1 for i in range(10)] + [None] * 10,
            "val_loss": [None] * 10 + [1.4 - i*0.08 for i in range(10)],
            "train_acc": [0.3 + i*0.05 for i in range(10)] + [None] * 10,
            "val_acc": [None] * 10 + [0.35 + i*0.04 for i in range(10)]
        }
        df = pd.DataFrame(metrics_data)
        
        metrics_path = tmp_path / "metrics.csv"
        df.to_csv(metrics_path, index=False)
        
        output_path = tmp_path / "training_curves.png"
        fig = plot_training_curves(
            str(metrics_path),
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert fig is not None
    
    def test_plot_with_training_info(self, tmp_path, mock_training_info):
        """测试带训练信息的训练曲线"""
        from src.utils.plotting import plot_training_curves
        
        # 创建 metrics.csv
        metrics_data = {
            "epoch": list(range(5)),
            "train_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
            "val_loss": [1.1, 0.9, 0.7, 0.6, 0.55]
        }
        df = pd.DataFrame(metrics_data)
        
        metrics_path = tmp_path / "metrics.csv"
        df.to_csv(metrics_path, index=False)
        
        # 创建 training_info.json
        info_path = tmp_path / "training_info.json"
        with open(info_path, "w") as f:
            json.dump(mock_training_info, f)
        
        output_path = tmp_path / "training_curves_with_info.png"
        fig = plot_training_curves(
            str(metrics_path),
            output_path=str(output_path),
            training_info_path=str(info_path)
        )
        
        assert output_path.exists()


class TestFindLatestMetricsCsv:
    """测试 find_latest_metrics_csv 函数"""
    
    def test_find_metrics(self, tmp_path):
        """测试查找 metrics.csv"""
        from src.utils.plotting import find_latest_metrics_csv
        
        # 创建模拟的 lightning_logs 结构
        version_dir = tmp_path / "lightning_logs" / "version_0"
        version_dir.mkdir(parents=True)
        
        metrics_path = version_dir / "metrics.csv"
        metrics_path.write_text("epoch,loss\n0,1.0\n")
        
        result = find_latest_metrics_csv(str(tmp_path / "lightning_logs"))
        
        assert result is not None
        assert result[0] == str(metrics_path)
    
    def test_find_nonexistent(self, tmp_path):
        """测试查找不存在的目录"""
        from src.utils.plotting import find_latest_metrics_csv
        
        result = find_latest_metrics_csv(str(tmp_path / "nonexistent"))
        
        assert result is None