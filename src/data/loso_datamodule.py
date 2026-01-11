"""Leave-One-Subject-Out (LOSO) 交叉验证 DataModule

支持可配置折数的跨被试交叉验证，用于评估模型的泛化能力。
"""
import os
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
from braindecode.util import set_random_seeds
import torch
import numpy as np
import mne
from typing import Optional, List

from .fold_utils import FoldSplitter

# 抑制 MNE 滤波器日志，只显示警告及以上级别
mne.set_log_level('WARNING')


def setup_data_path(data_dir: str = "./data"):
    """
    设置 MNE_DATA 环境变量，让 MOABB 从项目目录加载数据
    
    如果项目目录中存在 MNE-bnci-data 文件夹，则使用项目目录作为数据源。
    这样可以实现离线运行，无需每次从网络下载数据。
    
    Args:
        data_dir: 项目数据目录路径
        
    Returns:
        实际使用的数据目录路径
    """
    # 获取项目根目录
    # 假设此文件位于 src/data/loso_datamodule.py
    # 项目根目录是 ../../
    module_path = Path(__file__).resolve()
    project_root = module_path.parent.parent.parent
    
    # 构建数据目录的绝对路径
    if Path(data_dir).is_absolute():
        abs_data_dir = Path(data_dir)
    else:
        abs_data_dir = project_root / data_dir
    
    # 检查项目数据目录是否存在 MNE-bnci-data
    bnci_data_path = abs_data_dir / "MNE-bnci-data"
    
    if bnci_data_path.exists():
        # 使用项目目录
        mne.set_config('MNE_DATA', str(abs_data_dir), set_env=True)
        os.environ['MNE_DATA'] = str(abs_data_dir)
        print(f"[数据路径] 使用本地数据: {abs_data_dir}")
        return str(abs_data_dir)
    else:
        # 检查默认 MNE_DATA 是否已设置
        current_mne_data = mne.get_config('MNE_DATA')
        if current_mne_data:
            print(f"[数据路径] 使用 MNE 默认路径: {current_mne_data}")
            return current_mne_data
        else:
            # 没有本地数据，将从网络下载到项目目录
            abs_data_dir.mkdir(parents=True, exist_ok=True)
            mne.set_config('MNE_DATA', str(abs_data_dir), set_env=True)
            os.environ['MNE_DATA'] = str(abs_data_dir)
            print(f"[数据路径] 数据将下载到: {abs_data_dir}")
            return str(abs_data_dir)


def scale_to_uv(data):
    """
    将 EEG 数据从 V 转换为 µV
    
    使用顶层函数而非 lambda，以支持 braindecode 的序列化。
    
    注意：Preprocessor 在调用自定义函数时，可能传入的是：
    - numpy.ndarray (通过 raw.apply_function 调用时)
    - MNE Raw 对象 (直接作为 fn 参数时)
    
    Args:
        data: numpy 数组 (EEG 数据) 或 MNE Raw 对象
        
    Returns:
        缩放后的数据
    """
    return data * 1e6


class LOSODataModule(pl.LightningDataModule):
    """
    支持 LOSO 交叉验证的 EEG DataModule
    
    与原 EEGDataModule 的区别:
    - 支持 n_folds 参数（默认 3 折）
    - 支持 fold_id 参数（当前折）
    - 训练集使用 n-1 组被试
    - 测试集使用当前折的被试
    - 启动时显示折数分配方案
    
    使用方法:
        >>> dm = LOSODataModule(
        ...     dataset_name="BNCI2014_001",
        ...     all_subject_ids=[1,2,3,4,5,6,7,8,9],
        ...     n_folds=3,
        ...     fold_id=1,
        ... )
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
    """
    
    def __init__(
        self,
        dataset_name: str = "BNCI2014_001",
        all_subject_ids: List[int] = None,  # 默认 [1,2,3,4,5,6,7,8,9]
        n_folds: int = 3,                    # 折数
        fold_id: int = 1,                    # 当前折 (1-based)
        n_classes: int = 4,
        tmin: float = 0.0,
        tmax: float = 4.0,
        fmin: float = 4.0,
        fmax: float = 38.0,
        batch_size: int = 128,               # 比原来大，因为数据更多
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        data_dir: str = "./data",
        show_fold_info: bool = True,         # 是否显示折数分配
        val_split: float = 0.1,              # 从训练集中划分验证集的比例
        **kwargs,  # 接受 Hydra 配置中的额外字段（如 name）
    ):
        """
        初始化 LOSODataModule
        
        Args:
            dataset_name: MOABB 数据集名称
            all_subject_ids: 所有被试 ID 列表，默认 [1,2,3,4,5,6,7,8,9]
            n_folds: 交叉验证折数
            fold_id: 当前折 ID (1-based)
            n_classes: 分类类别数
            tmin: 试次开始时间（相对于事件）
            tmax: 试次结束时间
            fmin: 带通滤波最低频率
            fmax: 带通滤波最高频率
            batch_size: 批次大小
            num_workers: DataLoader 工作进程数
            pin_memory: 是否使用 pinned memory
            persistent_workers: 是否保持工作进程存活
            seed: 随机种子
            data_dir: 数据存储目录
            show_fold_info: 是否在 setup 时打印折数分配信息
            val_split: 验证集占训练数据的比例
        """
        super().__init__()
        
        # 数据集配置
        self.dataset_name = dataset_name
        self.all_subject_ids = (
            list(all_subject_ids) if all_subject_ids is not None 
            else [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        self.n_folds = n_folds
        self.fold_id = fold_id
        self.n_classes = n_classes
        
        # 预处理配置
        self.tmin = tmin
        self.tmax = tmax
        self.fmin = fmin
        self.fmax = fmax
        
        # DataLoader 配置
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and num_workers > 0
        
        # 其他配置
        self.seed = seed
        self.data_dir = data_dir
        self.show_fold_info = show_fold_info
        self.val_split = val_split
        
        # 设置数据路径（优先使用项目目录中的离线数据）
        self._actual_data_dir = setup_data_path(data_dir)
        
        # 验证 fold_id
        if fold_id < 1 or fold_id > n_folds:
            raise ValueError(
                f"fold_id 必须在 1 到 {n_folds} 之间，当前值: {fold_id}"
            )
        
        # 创建 FoldSplitter
        self.fold_splitter = FoldSplitter(
            subject_ids=self.all_subject_ids,
            n_folds=self.n_folds
        )
        
        # 获取当前折的分配
        self.current_fold = self.fold_splitter.get_fold(self.fold_id)
        self.train_subjects = self.current_fold['train_subjects']
        self.test_subjects = self.current_fold['test_subjects']
        
        # 数据集占位符
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
        # 保存超参数 (用于 Lightning 日志)
        self.save_hyperparameters()

    def prepare_data(self):
        """
        下载数据（如果需要）
        
        MOABBDataset 会自动处理下载。此方法在单进程中调用。
        """
        pass

    def _load_and_preprocess_subjects(self, subject_ids: List[int]):
        """
        加载并预处理指定被试的数据
        
        Args:
            subject_ids: 要加载的被试 ID 列表
        
        Returns:
            预处理后的 WindowsDataset
        """
        if not subject_ids:
            return None
        
        # 1. 加载数据集
        dataset = MOABBDataset(
            dataset_name=self.dataset_name, 
            subject_ids=subject_ids
        )

        # 2. 预处理
        preprocessors = [
            Preprocessor("pick_types", eeg=True, meg=False, stim=False),
            Preprocessor(scale_to_uv),  # V -> uV (使用顶层函数避免序列化警告)
            Preprocessor("filter", l_freq=self.fmin, h_freq=self.fmax),
        ]
        
        preprocess(dataset, preprocessors, n_jobs=1)

        # 3. 获取采样率并计算窗口参数
        sfreq = dataset.datasets[0].raw.info["sfreq"]
        window_size_samples = int((self.tmax - self.tmin) * sfreq)
        
        # 4. 定义类别映射
        # MOABB 返回的事件代码是 1, 2, 3, 4（不是 769-772）
        # 类名使用 'feet' 而不是 'foot'
        if self.dataset_name == "BNCI2014_001":
            mapping = {
                'left_hand': 1,
                'right_hand': 2,
                'feet': 3,
                'tongue': 4,
            }
            target_offset = 1
        elif self.dataset_name == "BNCI2014_004":
            mapping = {
                'left_hand': 1,
                'right_hand': 2,
            }
            target_offset = 1
        else:
            mapping = None
            target_offset = 0
        
        # 5. 创建窗口
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=int(self.tmin * sfreq),
            trial_stop_offset_samples=0,
            window_size_samples=window_size_samples,
            window_stride_samples=window_size_samples,
            drop_last_window=True,
            mapping=mapping,
            preload=True,
        )
        
        # 6. 将目标值映射到 0-based 索引
        if target_offset > 0:
            for ds in windows_dataset.datasets:
                if hasattr(ds, 'y') and ds.y is not None:
                    ds.y = np.array(ds.y) - target_offset

        return windows_dataset

    def setup(self, stage: Optional[str] = None):
        """
        数据准备流程
        
        1. 显示折数分配信息（如果启用）
        2. 分别加载训练被试和测试被试的数据
        3. 从训练数据中划分验证集
        
        Args:
            stage: 'fit', 'validate', 'test', 或 'predict'
        """
        # 避免重复加载
        if self.train_set is not None and self.test_set is not None:
            return

        # 设置随机种子
        set_random_seeds(seed=self.seed, cuda=torch.cuda.is_available())

        # 显示折数分配信息
        if self.show_fold_info:
            print("\n")
            self.fold_splitter.print_allocation()
            print(f"\n当前运行: Fold {self.fold_id}")
            print(f"  训练被试: {self.train_subjects}")
            print(f"  测试被试: {self.test_subjects}")
            print()

        # 加载训练被试数据
        print(f"加载训练被试数据: {self.train_subjects}")
        train_windows = self._load_and_preprocess_subjects(self.train_subjects)
        
        # 加载测试被试数据
        print(f"加载测试被试数据: {self.test_subjects}")
        test_windows = self._load_and_preprocess_subjects(self.test_subjects)

        # LOSO 模式下，我们合并所有 session 的数据
        # 因为不同被试间的数据差异比同一被试的 session 差异更大
        train_data_list = []
        if train_windows is not None:
            for ds in train_windows.datasets:
                train_data_list.append(ds)
        
        test_data_list = []
        if test_windows is not None:
            for ds in test_windows.datasets:
                test_data_list.append(ds)
        
        # 合并所有训练数据
        if train_data_list:
            combined_train = ConcatDataset(train_data_list)
        else:
            raise ValueError("训练数据为空！请检查被试 ID 配置。")
        
        # 合并所有测试数据
        if test_data_list:
            self.test_set = ConcatDataset(test_data_list)
        else:
            raise ValueError("测试数据为空！请检查被试 ID 配置。")

        # 从训练数据中划分验证集
        total_train = len(combined_train)
        val_size = int(total_train * self.val_split)
        train_size = total_train - val_size
        
        # 使用固定种子进行划分，确保可重复性
        generator = torch.Generator().manual_seed(self.seed)
        self.train_set, self.val_set = torch.utils.data.random_split(
            combined_train, 
            [train_size, val_size],
            generator=generator
        )

        # 打印数据统计
        print(f"\n数据加载完成:")
        print(f"  训练集: {len(self.train_set)} 样本")
        print(f"  验证集: {len(self.val_set)} 样本")
        print(f"  测试集: {len(self.test_set)} 样本")
        print()

    def train_dataloader(self):
        """返回训练数据加载器"""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        """返回验证数据加载器"""
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        """返回测试数据加载器"""
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def get_fold_info(self) -> dict:
        """
        获取当前折的信息
        
        Returns:
            包含折信息的字典
        """
        return {
            'fold_id': self.fold_id,
            'n_folds': self.n_folds,
            'train_subjects': self.train_subjects,
            'test_subjects': self.test_subjects,
            'all_subjects': self.all_subject_ids,
        }

    def __repr__(self) -> str:
        return (
            f"LOSODataModule("
            f"dataset={self.dataset_name}, "
            f"fold={self.fold_id}/{self.n_folds}, "
            f"train_subjects={self.train_subjects}, "
            f"test_subjects={self.test_subjects})"
        )