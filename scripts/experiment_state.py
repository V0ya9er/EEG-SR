"""
实验状态管理，支持断点续跑

功能:
- 记录已完成/失败/进行中的实验
- 支持从中断处恢复
- 生成实验进度报告
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Set
from pathlib import Path
import hashlib

class ExperimentState:
    """实验状态管理器"""
    
    def __init__(self, state_file: str = "experiment_state.json"):
        self.state_file = Path(state_file)
        self.state = self._load_or_create()
    
    def _load_or_create(self) -> Dict:
        """加载或创建状态文件"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed": [],
            "failed": [],
            "running": []
        }
    
    def save(self):
        """保存状态到文件"""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    @staticmethod
    def generate_exp_id(config: Dict) -> str:
        """生成实验唯一 ID"""
        # 格式: dataset_model_mechanism_noise_Dintensity_foldfold_id
        parts = [
            config.get("dataset", "unknown"),
            config.get("model", "unknown"),
            config.get("mechanism", "unknown"),
            config.get("noise", "unknown"),
            f"D{config.get('intensity', 0):.2f}",
            f"fold{config.get('fold_id', 0)}"
        ]
        return "_".join(parts)
    
    def mark_running(self, exp_id: str):
        """标记实验正在运行"""
        if exp_id not in self.state["running"]:
            self.state["running"].append(exp_id)
        self.save()
    
    def mark_completed(self, exp_id: str, result_path: str = None):
        """标记实验完成"""
        if exp_id in self.state["running"]:
            self.state["running"].remove(exp_id)
        if exp_id not in self.state["completed"]:
            self.state["completed"].append(exp_id)
        self.save()
    
    def mark_failed(self, exp_id: str, error: str = None):
        """标记实验失败"""
        if exp_id in self.state["running"]:
            self.state["running"].remove(exp_id)
        if exp_id not in self.state["failed"]:
            self.state["failed"].append(exp_id)
        self.save()
    
    def is_completed(self, exp_id: str) -> bool:
        """检查实验是否已完成"""
        return exp_id in self.state["completed"]
    
    def get_pending(self, all_exp_ids: List[str]) -> List[str]:
        """获取待运行的实验（排除已完成和正在运行的）"""
        done = set(self.state["completed"]) | set(self.state["running"])
        return [e for e in all_exp_ids if e not in done]
    
    def get_progress(self) -> Dict:
        """获取进度统计"""
        return {
            "completed": len(self.state["completed"]),
            "failed": len(self.state["failed"]),
            "running": len(self.state["running"])
        }
    
    def print_status(self):
        """打印状态报告"""
        progress = self.get_progress()
        total = progress["completed"] + progress["failed"] + progress["running"]
        print(f"\n{'='*50}")
        print("实验进度报告")
        print(f"{'='*50}")
        print(f"已完成: {progress['completed']}")
        print(f"失败:   {progress['failed']}")
        print(f"运行中: {progress['running']}")
        if total > 0:
            pct = progress['completed'] / total * 100
            print(f"完成率: {pct:.1f}%")
        print(f"{'='*50}\n")