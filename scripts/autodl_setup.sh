#!/bin/bash
# ============================================
# AutoDL 环境初始化脚本 (优化版)
# ============================================
#
# 使用方式:
#   chmod +x scripts/autodl_setup.sh
#   ./scripts/autodl_setup.sh [--skip-download]
#
# 选项:
#   --skip-download  跳过数据下载（使用本地数据时）
#
# 该脚本会:
#   1. 配置国内镜像源
#   2. 检查 GPU 配置
#   3. 安装 Python 依赖
#   4. 配置离线数据或下载数据
#   5. 验证安装
#

set -e  # 遇到错误立即退出

# 解析参数
SKIP_DOWNLOAD=false
for arg in "$@"; do
    case $arg in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
    esac
done

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              AutoDL 环境初始化                             ║"
echo "║              EEG-SR 随机共振实验                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================
# 1. 配置国内镜像源
# ============================================
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ [1/6] 配置国内镜像源                                        │"
echo "└──────────────────────────────────────────────────────────────┘"

# pip 清华镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn 2>/dev/null || true

# conda 清华镜像 (如果使用 conda)
if command -v conda &> /dev/null; then
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ 2>/dev/null || true
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 2>/dev/null || true
    conda config --set show_channel_urls yes 2>/dev/null || true
fi

echo "✓ 镜像源配置完成 (清华 TUNA)"

# ============================================
# 2. 安装系统工具
# ============================================
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ [2/7] 安装系统工具 (7z, unrar)                              │"
echo "└──────────────────────────────────────────────────────────────┘"

# 检查是否需要安装
NEED_INSTALL=false
if ! command -v 7z &> /dev/null; then
    echo "  7z 未安装"
    NEED_INSTALL=true
fi
if ! command -v unrar &> /dev/null; then
    echo "  unrar 未安装"
    NEED_INSTALL=true
fi

if [ "$NEED_INSTALL" = true ]; then
    echo "安装解压工具..."
    apt-get update -qq
    apt-get install -y -qq p7zip-full unrar 2>/dev/null || {
        echo "  ⚠ apt 安装失败，尝试其他方式..."
        # 尝试使用 conda 安装
        if command -v conda &> /dev/null; then
            conda install -y -c conda-forge p7zip unrar 2>/dev/null || true
        fi
    }
fi

# 验证安装
if command -v 7z &> /dev/null; then
    echo "✓ 7z 已安装: $(7z | head -2 | tail -1)"
else
    echo "⚠ 7z 安装失败，请手动安装: apt-get install p7zip-full"
fi

if command -v unrar &> /dev/null; then
    echo "✓ unrar 已安装"
else
    echo "⚠ unrar 安装失败，请手动安装: apt-get install unrar"
fi

# ============================================
# 3. 检查 GPU
# ============================================
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ [3/7] 检查 GPU 配置                                         │"
echo "└──────────────────────────────────────────────────────────────┘"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo ""
echo "✓ 检测到 ${GPU_COUNT} 个 GPU"

# ============================================
# 4. 创建数据目录（使用 SSD）
# ============================================
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ [4/7] 配置数据目录 (SSD)                                    │"
echo "└──────────────────────────────────────────────────────────────┘"

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_DATA="${PROJECT_ROOT}/data"

# AutoDL SSD 路径
SSD_PATH="/root/autodl-tmp"
MNE_PATH="${SSD_PATH}/mne_data"

mkdir -p "$MNE_PATH"

# 检查项目是否包含离线数据
if [ -d "${PROJECT_DATA}/MNE-bnci-data" ]; then
    echo "✓ 检测到项目内离线数据: ${PROJECT_DATA}/MNE-bnci-data"
    # 复制离线数据到 SSD 加速访问
    if [ ! -d "${MNE_PATH}/MNE-bnci-data" ]; then
        echo "  复制数据到 SSD..."
        cp -r "${PROJECT_DATA}/MNE-bnci-data" "${MNE_PATH}/"
        echo "  ✓ 数据已复制到 SSD"
    else
        echo "  ✓ SSD 中已存在数据"
    fi
    SKIP_DOWNLOAD=true
fi

# 设置 MNE 数据目录环境变量
export MNE_DATA="$MNE_PATH"
if ! grep -q "export MNE_DATA=" ~/.bashrc; then
    echo "export MNE_DATA=$MNE_PATH" >> ~/.bashrc
fi

# 创建符号链接（如果需要）
if [ -d "$HOME/mne_data" ] && [ ! -L "$HOME/mne_data" ]; then
    echo "移动现有数据到 SSD..."
    mv ~/mne_data/* "$MNE_PATH/" 2>/dev/null || true
    rm -rf ~/mne_data
fi
ln -sf "$MNE_PATH" ~/mne_data 2>/dev/null || true

echo "✓ MNE 数据目录: $MNE_PATH"

# ============================================
# 5. 安装依赖
# ============================================
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ [5/7] 安装 Python 依赖                                      │"
echo "└──────────────────────────────────────────────────────────────┘"

# 升级 pip
pip install --upgrade pip -q

# 安装项目依赖 (使用清华镜像)
echo "安装依赖中..."
pip install -r "${PROJECT_ROOT}/requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

echo "✓ Python 依赖安装完成"

# ============================================
# 6. 数据集准备
# ============================================
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ [6/7] EEG 数据集准备                                        │"
echo "└──────────────────────────────────────────────────────────────┘"

if [ "$SKIP_DOWNLOAD" = true ]; then
    echo "✓ 使用离线数据，跳过下载"
else
    echo "下载 EEG 数据集 (这可能需要 5-10 分钟)..."
    echo ""

    python -c "
import os
os.environ['MNE_DATA'] = '$MNE_PATH'

from braindecode.datasets import MOABBDataset

print('📥 下载 BCI IV 2a 数据集 (9 被试)...')
try:
    ds = MOABBDataset('BNCI2014_001', subject_ids=[1,2,3,4,5,6,7,8,9])
    print('   ✓ BCI IV 2a 下载完成')
except Exception as e:
    print(f'   ⚠ 下载失败: {e}')

print('')
print('📥 下载 BCI IV 2b 数据集 (9 被试)...')
try:
    ds = MOABBDataset('BNCI2014_004', subject_ids=[1,2,3,4,5,6,7,8,9])
    print('   ✓ BCI IV 2b 下载完成')
except Exception as e:
    print(f'   ⚠ 下载失败: {e}')
"
fi

echo ""
echo "✓ 数据集准备完成"

# ============================================
# 7. 验证安装
# ============================================
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ [7/7] 验证安装                                              │"
echo "└──────────────────────────────────────────────────────────────┘"

python -c "
import torch
import pytorch_lightning as pl
import hydra

print(f'PyTorch 版本:     {torch.__version__}')
print(f'CUDA 可用:        {torch.cuda.is_available()}')
print(f'CUDA 设备数:      {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')

print(f'Lightning 版本:   {pl.__version__}')
print(f'Hydra 版本:       {hydra.__version__}')

# 测试项目模块
from src.data.fold_utils import FoldSplitter
from src.data.loso_datamodule import LOSODataModule

print('')
print('📋 FoldSplitter 测试 (3 折):')
splitter = FoldSplitter([1,2,3,4,5,6,7,8,9], n_folds=3)
splitter.print_allocation()

print('')
print('✅ 所有依赖安装正确!')
"

# ============================================
# 完成
# ============================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ 初始化完成!                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📖 使用示例:"
echo ""
echo "  # 单折训练 (默认配置)"
echo "  python src/loso_train.py dataset=bci2a_loso"
echo ""
echo "  # 指定折和噪声强度"
echo "  python src/loso_train.py dataset=bci2a_loso dataset.fold_id=2 sr.mechanism.intensity=0.5"
echo ""
echo "  # 多折批量运行 (Hydra multirun)"
echo "  python src/loso_train.py --multirun dataset.fold_id=1,2,3"
echo ""
echo "  # 多 GPU 并行实验 (推荐)"
echo "  python scripts/run_sweep.py --gpus 0,1,2,3"
echo ""
echo "  # 结果分析"
echo "  python scripts/loso_analyze.py --results-dir . --verbose"
echo ""