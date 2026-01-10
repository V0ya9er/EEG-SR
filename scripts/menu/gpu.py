#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 管理模块

提供 GPU 检测、信息解析和选择功能。
"""

import re
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GPUInfo:
    """GPU 信息数据类"""
    id: int
    name: str
    memory_total: float  # GB
    memory_used: float   # GB
    memory_free: float   # GB
    utilization: int     # 百分比
    temperature: int     # 摄氏度
    
    @property
    def memory_usage_percent(self) -> float:
        """内存使用百分比"""
        if self.memory_total <= 0:
            return 0.0
        return (self.memory_used / self.memory_total) * 100
    
    def format_memory(self) -> str:
        """格式化内存信息"""
        return f"{self.memory_used:.1f}/{self.memory_total:.1f} GB"
    
    def format_summary(self) -> str:
        """格式化摘要信息"""
        return (f"GPU {self.id}: {self.name} | "
                f"{self.format_memory()} | "
                f"利用率 {self.utilization}%")


class GPUManager:
    """GPU 管理器"""
    
    def __init__(self):
        self._cache: Optional[List[GPUInfo]] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 30  # 缓存有效期（秒）
    
    def get_gpu_list(self, refresh: bool = False) -> List[GPUInfo]:
        """
        获取 GPU 列表
        
        Args:
            refresh: 是否强制刷新
            
        Returns:
            GPUInfo 列表
        """
        now = time.time()
        if not refresh and self._cache and (now - self._cache_time < self._cache_ttl):
            return self._cache
        
        gpus = self._detect_gpus()
        self._cache = gpus
        self._cache_time = now
        return gpus
    
    def _detect_gpus(self) -> List[GPUInfo]:
        """检测系统中的 GPU"""
        gpus = []
        
        # 方法1: 尝试使用 pynvml
        gpus = self._detect_with_pynvml()
        if gpus:
            return gpus
        
        # 方法2: 解析 nvidia-smi 输出
        gpus = self._detect_with_nvidia_smi()
        if gpus:
            return gpus
        
        # 方法3: 使用 PyTorch CUDA 检测
        gpus = self._detect_with_pytorch()
        if gpus:
            return gpus
        
        return []
    
    def _detect_with_pynvml(self) -> List[GPUInfo]:
        """使用 pynvml 库检测 GPU"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            gpus = []
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 获取名称
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # 获取内存信息
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = mem_info.total / (1024**3)
                memory_used = mem_info.used / (1024**3)
                memory_free = mem_info.free / (1024**3)
                
                # 获取利用率
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                except pynvml.NVMLError:
                    utilization = 0
                
                # 获取温度
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except pynvml.NVMLError:
                    temperature = 0
                
                gpus.append(GPUInfo(
                    id=i,
                    name=name,
                    memory_total=memory_total,
                    memory_used=memory_used,
                    memory_free=memory_free,
                    utilization=utilization,
                    temperature=temperature
                ))
            
            pynvml.nvmlShutdown()
            return gpus
            
        except (ImportError, Exception):
            return []
    
    def _detect_with_nvidia_smi(self) -> List[GPUInfo]:
        """通过解析 nvidia-smi 输出检测 GPU"""
        try:
            # 使用 CSV 格式获取结构化数据
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return []
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    try:
                        gpus.append(GPUInfo(
                            id=int(parts[0]),
                            name=parts[1],
                            memory_total=float(parts[2]) / 1024,  # MB -> GB
                            memory_used=float(parts[3]) / 1024,
                            memory_free=float(parts[4]) / 1024,
                            utilization=int(parts[5]) if parts[5] != '[N/A]' else 0,
                            temperature=int(parts[6]) if parts[6] != '[N/A]' else 0
                        ))
                    except (ValueError, IndexError):
                        continue
            
            return gpus
            
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return []
    
    def _detect_with_pytorch(self) -> List[GPUInfo]:
        """使用 PyTorch 检测 GPU（基本信息）"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return []
            
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                
                # 尝试获取当前内存使用
                try:
                    memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                except Exception:
                    memory_used = 0
                
                gpus.append(GPUInfo(
                    id=i,
                    name=props.name,
                    memory_total=props.total_memory / (1024**3),
                    memory_used=memory_used,
                    memory_free=(props.total_memory / (1024**3)) - memory_used,
                    utilization=0,  # PyTorch 无法直接获取利用率
                    temperature=0   # PyTorch 无法直接获取温度
                ))
            
            return gpus
            
        except ImportError:
            return []
    
    def get_raw_nvidia_smi(self) -> Tuple[bool, str]:
        """
        获取原始 nvidia-smi 输出
        
        Returns:
            (是否成功, 输出内容)
        """
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr or "nvidia-smi 执行失败"
        except FileNotFoundError:
            return False, "未找到 nvidia-smi，可能未安装 NVIDIA 驱动"
        except subprocess.TimeoutExpired:
            return False, "nvidia-smi 执行超时"
        except Exception as e:
            return False, f"执行 nvidia-smi 时出错: {e}"
    
    def is_gpu_available(self) -> bool:
        """检查是否有可用的 GPU"""
        return len(self.get_gpu_list()) > 0
    
    def get_gpu_by_id(self, gpu_id: int) -> Optional[GPUInfo]:
        """根据 ID 获取 GPU 信息"""
        gpus = self.get_gpu_list()
        for gpu in gpus:
            if gpu.id == gpu_id:
                return gpu
        return None
    
    def get_best_gpu(self) -> Optional[GPUInfo]:
        """
        获取最佳可用 GPU（基于空闲内存）
        
        Returns:
            最佳 GPU 或 None
        """
        gpus = self.get_gpu_list()
        if not gpus:
            return None
        
        # 按空闲内存排序，选择空闲最多的
        return max(gpus, key=lambda g: g.memory_free)
    
    def validate_gpu_id(self, gpu_id: int) -> bool:
        """验证 GPU ID 是否有效"""
        gpus = self.get_gpu_list()
        return any(g.id == gpu_id for g in gpus)
    
    def format_gpu_table(self, highlight_id: Optional[int] = None) -> str:
        """
        格式化 GPU 表格输出
        
        Args:
            highlight_id: 高亮显示的 GPU ID
            
        Returns:
            格式化的表格字符串
        """
        gpus = self.get_gpu_list()
        
        if not gpus:
            return "  未检测到 NVIDIA GPU\n"
        
        lines = []
        
        # 表头
        lines.append("  ID   名称                            显存           使用率   温度")
        lines.append("  " + "─" * 70)
        
        # 数据行
        for gpu in gpus:
            marker = "→" if gpu.id == highlight_id else " "
            mem_str = f"{gpu.memory_used:.1f}/{gpu.memory_total:.1f} GB"
            util_str = f"{gpu.utilization:3d}%"
            temp_str = f"{gpu.temperature}°C" if gpu.temperature > 0 else "N/A"
            
            # 截断过长的名称
            name = gpu.name[:30] if len(gpu.name) > 30 else gpu.name
            
            lines.append(
                f"{marker} [{gpu.id}]  {name:<30}  {mem_str:<14}  {util_str:<6}  {temp_str}"
            )
        
        # CPU 选项
        lines.append("  " + "─" * 70)
        lines.append("  [C]  CPU 模式（不使用 GPU）")
        
        return "\n".join(lines)


# 全局单例
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """获取全局 GPU 管理器实例"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager