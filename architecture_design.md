# EEG模型随机共振(SR)机制研究 - 架构设计方案

## 1. 项目概述
本项目旨在探究随机共振（Stochastic Resonance, SR）机制对 EEGNet 和 EEG Conformer 模型在 BCIIV 数据集上性能的影响。项目采用模块化设计，支持多种噪声注入方式，并提供自动化的多GPU训练与全面的性能分析。

## 2. 技术选型
- **编程语言**: Python 3.9+
- **深度学习框架**: PyTorch 2.0+
- **训练流程管理**: PyTorch Lightning (自动化训练循环、多GPU支持、Logging)
- **配置管理**: Hydra (分层配置，支持多实验管理)
- **数据处理**: 在线实时处理 (Online Processing) + 性能计时器

## 3. 目录结构设计

```
SRTest/
├── conf/                       # Hydra 配置文件目录
│   ├── config.yaml             # 主配置文件
│   ├── model/                  # 模型配置
│   │   ├── eegnet.yaml
│   │   └── conformer.yaml
│   ├── sr/                     # 随机共振模块配置
│   │   ├── mechanism/          # 共振机制 (加性、双稳态等)
│   │   │   ├── additive.yaml
│   │   │   ├── bistable.yaml
│   │   │   └── tristable.yaml
│   │   └── noise/              # 噪声类型
│   │       ├── gaussian.yaml
│   │       ├── uniform.yaml
│   │       ├── alpha_stable.yaml
│   │       ├── poisson.yaml
│   │       └── colored.yaml
│   ├── dataset/                # 数据集配置
│   │   ├── bciiv2a.yaml
│   │   └── bciiv2b.yaml
│   └── train.yaml              # 训练参数 (batch_size, epochs等)
├── src/
│   ├── models/
│   │   ├── components/         # 原始模型代码 (EEGNet.py, conformer.py)
│   │   └── lit_module.py       # PyTorch Lightning Module 包装器
│   ├── modules/
│   │   └── sr_layer.py         # 随机共振注入层 (核心模块)
│   ├── data/
│   │   ├── datamodule.py       # Lightning DataModule
│   │   ├── transforms.py       # 数据预处理与增强
│   │   └── timer.py            # 耗时统计工具
│   └── utils/
│   │   └── metrics.py          # 自定义评价指标
├── scripts/                    # 辅助脚本
│   └── analyze_results.py      # 结果分析与绘图
├── train.py                    # 训练入口脚本
├── test.py                     # 测试入口脚本
└── requirements.txt            # 依赖列表
```

## 4. 核心模块接口设计

### 4.1 SR Mechanism (随机共振机制)
采用策略模式，将“噪声生成”与“共振机制”解耦，支持更复杂的 SR 类型。

#### 4.1.1 Noise Source (噪声源)
```python
class NoiseSource(nn.Module):
    """噪声生成基类"""
    def forward(self, shape: torch.Size) -> torch.Tensor:
        raise NotImplementedError

class GaussianNoise(NoiseSource): ...
class UniformNoise(NoiseSource): ...
class AlphaStableNoise(NoiseSource): ... # Alpha稳定分布
class PoissonNoise(NoiseSource): ...     # 泊松噪声
class ColoredNoise(NoiseSource): ...     # 有色噪声 (Pink/Red)
```

#### 4.1.2 SR Layer (共振层)
```python
class BaseSRLayer(nn.Module):
    def __init__(self, noise_source: NoiseSource, intensity: float):
        super().__init__()
        self.noise_source = noise_source
        self.intensity = intensity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class AdditiveSR(BaseSRLayer):
    """经典加性随机共振: x = x + D * noise"""
    def forward(self, x):
        noise = self.noise_source(x.shape)
        return x + self.intensity * noise

class BistableSR(BaseSRLayer):
    """双稳态随机共振 (Bistable Stochastic Resonance)
    模拟朗之万方程: dx/dt = ax - bx^3 + Input + Noise
    使用欧拉-马拉山法 (Euler-Maruyama) 进行离散化求解。
    """
    def __init__(self, noise_source, intensity, a=1.0, b=1.0, dt=0.01):
        super().__init__(noise_source, intensity)
        self.a = a
        self.b = b
        self.dt = dt

    def forward(self, x):
        # 模拟双稳态系统动力学演化
        # x_out[t] = x_out[t-1] + (a*x_out[t-1] - b*x_out[t-1]^3 + input[t] + noise[t]) * dt
        pass

class TristableSR(BaseSRLayer):
    """三稳态随机共振 (Tristable Stochastic Resonance)
    基于六次多项式势函数: U(x) = (a/2)x^2 - (b/4)x^4 + (c/6)x^6
    动力学方程: dx/dt = -ax + bx^3 - cx^5 + Input + Noise
    """
    def __init__(self, noise_source, intensity, a=1.0, b=1.0, c=1.0, dt=0.01):
        super().__init__(noise_source, intensity)
        self.a = a
        self.b = b
        self.c = c
        self.dt = dt

    def forward(self, x):
        # 模拟三稳态系统动力学演化
        # x_out[t] = x_out[t-1] + (-a*x_out[t-1] + b*x_out[t-1]^3 - c*x_out[t-1]^5 + input[t] + noise[t]) * dt
        pass
```

### 4.2 Lightning Module Wrapper
将原始 PyTorch 模型与 SR 层结合，并定义训练流程。

```python
class LitEEGModel(LightningModule):
    def __init__(self, model_cfg, sr_cfg, optimizer_cfg):
        super().__init__()
        self.sr_layer = SRLayer(**sr_cfg)
        self.model = self._build_model(model_cfg) # 实例化 EEGNet 或 Conformer
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x_noisy = self.sr_layer(x) # 注入噪声
        return self.model(x_noisy)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss
    
    # validation_step, test_step 类似...
```

### 4.3 Timed Data Processing
实现数据加载时的实时处理与耗时统计。

```python
class TimedTransform:
    def __init__(self, transform, timer_registry):
        self.transform = transform
        self.timer_registry = timer_registry

    def __call__(self, x):
        start = time.perf_counter()
        x = self.transform(x)
        end = time.perf_counter()
        self.timer_registry.record(self.transform.__name__, end - start)
        return x
```

## 5. 硬件兼容性与多GPU训练策略
利用 PyTorch Lightning 的 `Trainer` 实现硬件自适应，无需修改模型代码即可兼容本地单卡（CPU/GPU/MPS）和服务器多卡环境。

- **策略**:
  - **Auto**: Lightning 会自动检测硬件并选择最佳策略（如单卡使用默认策略，多卡自动切换为 DDP）。
  - **DDP**: 显式指定分布式数据并行 (Distributed Data Parallel)。

- **配置**: 在 `conf/config.yaml` 或命令行中指定。
  ```yaml
  trainer:
    accelerator: auto # 自动检测 (gpu, cpu, tpu, mps 等)
    devices: auto     # 自动选择可用设备数量 (例如本地为1，服务器为8)
    strategy: auto    # 自动选择策略 (或显式指定 'ddp' 以强制多卡并行)
  ```

## 6. 分析与可视化规划
1.  **性能指标**: Accuracy, F1-Score, Kappa 系数。
2.  **SR 效应曲线**: 绘制 `噪声强度 (X轴)` vs `模型性能 (Y轴)` 曲线，寻找最佳共振点。
3.  **耗时分析**: 统计不同预处理步骤和SR注入的平均耗时。
4.  **特征可视化**: 使用 t-SNE 展示 SR 注入前后特征分布的变化。

## 7. 实施步骤 (Todo List)

1.  **环境搭建**: 创建虚拟环境，安装 PyTorch, Lightning, Hydra 等依赖。
2.  **数据准备**: 编写 `DataModule`，实现 BCIIV 数据的读取、切分和在线预处理。
3.  **模型集成**: 将现有的 `EEGNet.py` 和 `conformer.py` 复制/链接到项目中，并编写 `LitEEGModel` 包装器。
4.  **SR模块开发**: 实现 `SRLayer`，支持高斯噪声等基础类型。
5.  **配置系统**: 建立 Hydra 配置文件结构。
6.  **训练脚本**: 编写 `train.py`，打通训练流程。
7.  **测试与验证**: 运行基准测试（无噪声），确保模型复现原性能。
8.  **SR实验**: 编写脚本批量运行不同噪声强度的实验。
9.  **分析工具**: 编写分析脚本，生成图表。