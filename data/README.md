# 数据目录

此目录用于存放 EEG 数据集，支持离线运行。

## 目录结构

```
data/
└── MNE-bnci-data/
    └── database/
        └── data-sets/
            ├── 001-2014/    # BCI IV 2a 数据集
            │   ├── A01T.mat
            │   ├── A01E.mat
            │   └── ...
            └── 004-2014/    # BCI IV 2b 数据集
                ├── B01T.mat
                ├── B01E.mat
                └── ...
```

## 获取数据

### 方法 1：运行下载脚本（推荐）

```bash
# 检查数据完整性
python scripts/download_data.py --check

# 下载缺失的数据
python scripts/download_data.py --download

# 复制数据到项目目录
python scripts/download_data.py --copy-to-project

# 或者一键完成所有操作
python scripts/download_data.py --all
```

### 方法 2：手动复制

如果您已经有 MNE 数据，可以手动复制：

```bash
# Windows
xcopy /E /I "%USERPROFILE%\mne_data\MNE-bnci-data" "data\MNE-bnci-data"

# Linux/Mac
cp -r ~/mne_data/MNE-bnci-data data/
```

## 数据集信息

| 数据集 | 代码名称 | 被试数 | 类别数 | Session |
|-------|---------|--------|--------|---------|
| BCI IV 2a | BNCI2014_001 | 9 | 4 (左手/右手/足/舌) | 2 |
| BCI IV 2b | BNCI2014_004 | 9 | 2 (左手/右手) | 5 |

## 离线运行

将整个项目目录（包含 `data/MNE-bnci-data`）打包上传到服务器：

```bash
# 压缩（排除不需要的文件）
tar --exclude='*.ckpt' --exclude='lightning_logs' --exclude='__pycache__' \
    -czvf eeg-sr-with-data.tar.gz .

# 或使用 7z
7z a -xr!*.ckpt -xr!lightning_logs -xr!__pycache__ eeg-sr-with-data.7z .
```

上传后，安装脚本会自动检测并使用本地数据：

```bash
./scripts/autodl_setup.sh --skip-download