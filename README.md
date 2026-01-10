# EEG-SR: Stochastic Resonance for EEG Classification

研究随机共振 (Stochastic Resonance) 对脑电图 (EEG) 信号分类性能影响的深度学习项目。这是一个用于学习 EEG 模型和随机共振原理及效果的小项目。

## ✨ 主要特性

- ✅ **LOSO 交叉验证** 作为标准评估方法
- ✅ 支持 **BCI IV 2a** 和 **2b** 数据集
- ✅ **EEGNet** 和 **Conformer** 两种模型
- ✅ 多种 **SR 机制**（加性、双稳态、三稳态）
- ✅ 多种 **噪声类型**（高斯、有色、泊松等）
- ✅ **智能多 GPU 并行** 批量实验
- ✅ **交互式实验菜单**
- ✅ **自动分析与可视化**

## 📦 安装

```bash
git clone https://github.com/your-username/EEG-SR.git
cd EEG-SR
pip install -r requirements.txt
```

### 环境要求

- Python 3.9+
- CUDA 11.7+（如需 GPU 加速）
- 8GB+ 内存

## 🚀 快速开始

### 交互式菜单（推荐）

```bash
python -m scripts.menu
```

菜单功能包括：
- 🚀 **快速开始**：一键运行单次实验
- ⚙️ **实验配置**：修改模型、数据集、SR 参数、训练参数
- 🔬 **运行实验**：标准训练、LOSO 交叉验证、批量实验
- 📊 **分析与可视化**：分析已训练模型，生成图表

### 命令行 LOSO 训练

```bash
# 单折 LOSO 训练（默认 fold 1）
python src/loso_train.py dataset=bci2a_loso

# 指定特定折（fold 2 = 留出第 2 个被试）
python src/loso_train.py dataset=bci2a_loso dataset.fold_id=2

# 运行多个折（使用 Hydra multirun）
python src/loso_train.py --multirun dataset=bci2a_loso dataset.fold_id=1,2,3,4,5,6,7,8,9

# 结合模型和 SR 配置
python src/loso_train.py \
    model=conformer \
    dataset=bci2a_loso \
    sr/mechanism=bistable \
    sr/noise=gaussian \
    sr.mechanism.intensity=0.5
```

### 批量实验

```bash
# 使用菜单的批量实验功能
python -m scripts.menu
# 选择 "运行实验" -> "批量实验（多 GPU 并行 LOSO）"

# 或使用实验脚本
./run_experiment.sh  # Linux/Mac
run_experiment.bat   # Windows
```

### 分析与可视化

```bash
# 分析 LOSO 结果
python scripts/loso_analyze.py --results-dir ./outputs

# 可视化
python src/visualize.py \
    --results-dir ./results/analysis \
    --output-dir ./results/figures
```

## 📁 项目结构

```
EEG-SR/
├── conf/                    # Hydra 配置文件
│   ├── config.yaml          # 主配置
│   ├── dataset/             # 数据集配置（含 LOSO）
│   ├── model/               # 模型配置
│   └── sr/                  # SR 机制和噪声配置
├── src/                     # 源代码
│   ├── train.py             # 标准训练入口
│   ├── loso_train.py        # LOSO 训练入口
│   ├── test.py              # 测试入口
│   ├── visualize.py         # 可视化入口
│   ├── models/              # 模型定义
│   ├── modules/             # SR 层等功能模块
│   ├── data/                # 数据处理和 DataModule
│   └── utils/               # 工具函数
├── scripts/                 # 辅助脚本
│   ├── menu.py              # 交互式菜单
│   └── loso_analyze.py      # LOSO 结果分析
├── tests/                   # 测试文件
├── requirements.txt         # Python 依赖
└── README.md                # 项目文档
```

## 🔊 随机共振机制

| 机制 | 配置名称 | 说明 |
|------|---------|------|
| 加性 SR | `additive` | 直接将噪声叠加到输入信号 |
| 双稳态 SR | `bistable` | 基于双稳态势函数的非线性 SR |
| 三稳态 SR | `tristable` | 基于六次势函数，具有三个稳定态 |

## 📈 噪声类型

| 噪声类型 | 配置名称 | 说明 |
|---------|---------|------|
| 高斯噪声 | `gaussian` | 标准正态分布白噪声 |
| 均匀噪声 | `uniform` | 均匀分布 |
| Alpha 稳定噪声 | `alpha_stable` | 重尾分布 |
| 泊松噪声 | `poisson` | 离散脉冲噪声 |
| 有色噪声 | `colored` | 频率相关噪声 (1/f^β) |

## ☁️ AutoDL 部署

如需在 AutoDL 云平台部署，请参考 [`README_AutoDL.md`](README_AutoDL.md)。

## 📚 参考文献

### 模型
- **EEGNet**: Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*.
- **EEG Conformer**: Song, Y., et al. (2022). EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization. *IEEE TNSRE*.

### 随机共振
- Gammaitoni, L., et al. (1998). Stochastic resonance. *Reviews of Modern Physics*.
- McDonnell, M. D., & Abbott, D. (2009). What is stochastic resonance? *PLoS Computational Biology*.

### 数据集
- Tangermann, M., et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*.

## 📄 许可证

MIT License

## 🙏 致谢

感谢以下开源项目：
- [PyTorch Lightning](https://lightning.ai/)
- [Braindecode](https://braindecode.org/)
- [MNE-Python](https://mne.tools/)
- [MOABB](https://moabb.neurotechx.com/)
- [Hydra](https://hydra.cc/)