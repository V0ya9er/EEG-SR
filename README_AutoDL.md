# AutoDL 部署指南

本文档提供在 AutoDL 云平台上部署和运行 EEG 随机共振 (SR) 实验的完整指南。

## 🚀 快速开始

### 1. 创建实例

1. 登录 [AutoDL](https://www.autodl.com/)
2. 选择镜像：**PyTorch 2.0+ / CUDA 12.x**
3. 选择 GPU（参见下方推荐）

### GPU 选型推荐

| GPU 型号 | 显存 | 单实验时间 | 性价比 | 推荐场景 |
|---------|------|----------|--------|---------|
| **RTX 4080** | 16GB | ~60s | ⭐⭐⭐⭐⭐ | 预算有限 |
| **RTX 4090** | 24GB | ~45s | ⭐⭐⭐⭐ | 最佳性价比 |
| **RTX 4090D** | 24GB | ~45s | ⭐⭐⭐⭐ | 国内版 4090 |
| **L20** | 48GB | ~40s | ⭐⭐⭐ | 大 batch |
| **L40** | 48GB | ~35s | ⭐⭐⭐ | 专业级 |
| **H20** | 96GB | ~30s | ⭐⭐ | 土豪专属 |
| **H800** | 80GB | ~25s | ⭐ | 壕无人性 |
| **RTX 5090** | 32GB | ~35s | ⭐⭐⭐⭐ | 新一代 |
| **RTX 5090D** | 32GB | ~35s | ⭐⭐⭐⭐ | 新一代国内版 |
| **RTX Pro 6000** | 48GB | ~30s | ⭐⭐⭐ | 专业工作站 |

**推荐配置：4 × RTX 4090** - 约 ¥7.6/小时，性价比最高

### 2. 上传代码

```bash
# 方式 1：Git 克隆 (代码更新方便，但需要下载数据)
cd /root/autodl-tmp
git clone <your-repo-url> SRTest
cd SRTest

# 方式 2：上传压缩包 (推荐，包含离线数据)
# 详见下方"离线数据打包"章节
```

### 3. 初始化环境

```bash
chmod +x scripts/autodl_setup.sh
./scripts/autodl_setup.sh
```

初始化脚本会：
- ✅ 配置国内镜像源（清华 TUNA）
- ✅ 检查 GPU 配置
- ✅ 安装 Python 依赖
- ✅ 检测并使用离线数据（如有）
- ✅ 验证安装

**如果已包含离线数据：**
```bash
./scripts/autodl_setup.sh --skip-download
```

---

## 📊 运行实验

### 单折训练

```bash
# 默认配置 (BCI IV 2a, EEGNet, Additive SR, Gaussian 噪声)
python src/loso_train.py dataset=bci2a_loso

# 指定折
python src/loso_train.py dataset=bci2a_loso dataset.fold_id=2

# 指定模型和 SR 配置
python src/loso_train.py \
    dataset=bci2a_loso \
    model=conformer \
    sr/mechanism=bistable \
    sr/noise=colored \
    sr.mechanism.intensity=0.5
```

### 多折批量运行 (Hydra Multirun)

```bash
# 遍历 3 折
python src/loso_train.py --multirun dataset.fold_id=1,2,3

# 遍历折数 + 噪声强度
python src/loso_train.py --multirun \
    dataset.fold_id=1,2,3 \
    sr.mechanism.intensity=0.1,0.5,1.0,1.5,2.0

# 并行运行 (4 个进程)
python src/loso_train.py --multirun \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=4 \
    dataset.fold_id=1,2,3 \
    sr.mechanism.intensity=0.1,0.5,1.0
```

### 多 GPU 并行实验 (推荐)

```bash
# 使用所有 GPU，运行完整实验网格
python scripts/run_sweep.py --gpus 0,1,2,3

# 指定部分配置
python scripts/run_sweep.py \
    --gpus 0,1,2,3 \
    --datasets bci2a_loso \
    --models eegnet conformer \
    --mechanisms additive bistable

# 从断点恢复
python scripts/run_sweep.py --gpus 0,1,2,3 --resume

# 仅显示实验列表 (不运行)
python scripts/run_sweep.py --dry-run
```

---

## 📁 输出目录结构

运行实验后，结果保存在 `outputs/` 目录，使用语义化命名：

```
outputs/
├── bci2a_eegnet_add_gauss_fold1_D0.5/
│   ├── training_info.json        # 实验配置和结果
│   ├── .hydra/
│   │   └── config.yaml           # 完整 Hydra 配置
│   └── lightning_logs/
│       └── version_0/
│           ├── checkpoints/
│           │   ├── best-*.ckpt   # 最佳模型
│           │   └── last.ckpt     # 最后模型
│           └── metrics.csv       # 训练指标
├── bci2a_eegnet_add_gauss_fold2_D0.5/
├── bci2a_conformer_bi_color_fold1_D1.0/
└── ...

experiment_state.json             # 实验状态 (断点续跑)
```

**命名格式：** `{dataset}_{model}_{mechanism}_{noise}_fold{id}_D{intensity}`

---

## 📈 结果分析

```bash
# 基本分析 - 显示汇总统计
python scripts/loso_analyze.py --results-dir .

# 详细分析 - 显示每个折的结果
python scripts/loso_analyze.py --results-dir . --verbose

# 导出 CSV
python scripts/loso_analyze.py \
    --results-dir . \
    --output results/all_folds.csv \
    --summary-output results/summary.csv \
    --verbose

# JSON 格式输出 (便于后续处理)
python scripts/loso_analyze.py --results-dir . --format json > results.json
```

---

## ⚙️ 配置说明

### 折数配置

```yaml
# conf/dataset/bci2a_loso.yaml
n_folds: 3        # 默认 3 折 (可选 1-9)
fold_id: 1        # 当前折 ID
```

命令行覆盖：
```bash
# 使用 9 折 (完整 LOSO)
python src/loso_train.py dataset.n_folds=9 dataset.fold_id=1
```

### 噪声强度

```yaml
# conf/sr/mechanism/additive.yaml
intensity: 1.0    # 默认强度
```

实验需要遍历 0.1-2.0，步长 0.1：
```bash
python src/loso_train.py --multirun \
    sr.mechanism.intensity=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0
```

### 混合精度

已默认启用，在 `conf/config.yaml`：
```yaml
trainer:
  precision: "16-mixed"
```

---

## 💡 性能优化建议

1. **数据放 SSD**：`autodl_setup.sh` 自动处理
2. **使用混合精度**：已默认启用
3. **增大 batch_size**：LOSO 默认 128，可尝试 256
4. **多 GPU 并行**：使用 `run_sweep.py`
5. **关闭不必要的日志**：`trainer.enable_progress_bar=false`

---

## 🔧 常见问题

### Q: 实验中断如何恢复？

```bash
# 查看中断的实验
cat experiment_state.json | python -m json.tool

# 从断点恢复
python scripts/run_sweep.py --gpus 0,1,2,3 --resume
```

### Q: 如何查看实验进度？

```bash
# 查看状态文件
python -c "
import json
with open('experiment_state.json') as f:
    state = json.load(f)
total = len(state['experiments'])
completed = sum(1 for e in state['experiments'].values() if e['status'] == 'completed')
print(f'进度: {completed}/{total} ({100*completed/total:.1f}%)')
"
```

### Q: 显存不足 (OOM) 怎么办？

```bash
# 减小 batch_size
python src/loso_train.py dataset.batch_size=64

# 或使用梯度累积 (等效增大 batch)
python src/loso_train.py trainer.accumulate_grad_batches=2
```

### Q: 如何只运行部分实验？

```bash
# 使用 run_sweep.py 的过滤参数
python scripts/run_sweep.py \
    --gpus 0,1,2,3 \
    --datasets bci2a_loso \
    --models eegnet \
    --mechanisms additive \
    --noises gaussian colored
```

---

## 📞 实验规模参考

| 配置 | 实验数 | 4×4090 时间 | 费用估算 |
|------|--------|------------|---------|
| 3折 × 20强度 × 2模型 × 3机制 × 5噪声 | 1,800 | ~27 小时 | ~¥205 |
| 3折 × 10强度 × 1模型 × 1机制 × 1噪声 | 30 | ~0.5 小时 | ~¥4 |
| 9折 × 20强度 × 2模型 × 3机制 × 5噪声 | 5,400 | ~81 小时 | ~¥616 |

**推荐：** 先用小配置验证，再跑完整实验。

---

## 📦 离线数据打包

为避免在服务器上下载数据（BNCI 服务器较慢），可以在本地准备好数据后一起打包上传。

### 1. 本地准备数据

```bash
# 检查现有数据完整性
python scripts/download_data.py --check

# 如有缺失，下载补全
python scripts/download_data.py --download

# 复制数据到项目目录
python scripts/download_data.py --copy-to-project

# 或者一步完成
python scripts/download_data.py --all
```

### 2. 打包项目

**Windows (PowerShell):**
```powershell
# 使用 7-Zip (推荐)
7z a -xr!lightning_logs -xr!outputs -xr!__pycache__ -xr!.git -xr!*.ckpt `
    eeg-sr-with-data.7z .

# 或使用 tar (需要 Git Bash 或 WSL)
tar --exclude='lightning_logs' --exclude='outputs' --exclude='__pycache__' \
    --exclude='.git' --exclude='*.ckpt' -czvf eeg-sr-with-data.tar.gz .
```

**Linux/Mac:**
```bash
tar --exclude='lightning_logs' --exclude='outputs' --exclude='__pycache__' \
    --exclude='.git' --exclude='*.ckpt' -czvf eeg-sr-with-data.tar.gz .
```

预期大小：约 **150-200 MB**（含 BCI IV 2a + 2b 数据）

### 3. 上传到 AutoDL

1. 登录 AutoDL 控制台
2. 进入实例 → 文件管理
3. 上传 `eeg-sr-with-data.7z` 到 `/root/autodl-tmp/`
4. 解压并运行：

```bash
cd /root/autodl-tmp
7z x eeg-sr-with-data.7z -o./SRTest
# 或
tar -xzvf eeg-sr-with-data.tar.gz -C ./SRTest

cd SRTest
chmod +x scripts/autodl_setup.sh
./scripts/autodl_setup.sh --skip-download
```

安装脚本会自动检测 `data/MNE-bnci-data` 目录并使用离线数据。

### 4. 验证数据

```bash
python -c "
from src.data.loso_datamodule import LOSODataModule
dm = LOSODataModule(dataset_name='BNCI2014_001', n_folds=3, fold_id=1)
dm.setup()
print('✅ 数据加载成功')
"
```