#!/usr/bin/env python
"""
依赖验证脚本
检查项目所需的所有依赖库是否已正确安装
"""
import sys
from importlib.metadata import version, PackageNotFoundError


def check_package(package_name: str, import_name: str = None, min_version: str = None):
    """
    检查单个包是否已安装并满足版本要求
    
    Args:
        package_name: pip/conda 包名
        import_name: 导入时使用的名称（如果与包名不同）
        min_version: 最低版本要求
    """
    import_name = import_name or package_name
    
    try:
        # 尝试导入模块
        module = __import__(import_name.split('.')[0])
        
        # 获取版本
        try:
            pkg_version = version(package_name)
        except PackageNotFoundError:
            # 尝试从模块获取版本
            pkg_version = getattr(module, '__version__', 'unknown')
        
        # 版本检查
        status = "✅"
        version_ok = True
        if min_version and pkg_version != 'unknown':
            from packaging.version import parse
            try:
                if parse(pkg_version) < parse(min_version):
                    status = "⚠️"
                    version_ok = False
            except:
                pass
        
        return {
            "name": package_name,
            "installed": True,
            "version": pkg_version,
            "min_version": min_version,
            "version_ok": version_ok,
            "status": status
        }
        
    except ImportError as e:
        return {
            "name": package_name,
            "installed": False,
            "version": None,
            "min_version": min_version,
            "version_ok": False,
            "status": "❌",
            "error": str(e)
        }


def main():
    print("=" * 60)
    print("SR-EEG 项目依赖检查")
    print("=" * 60)
    print()
    
    # 定义需要检查的依赖列表
    # (包名, 导入名, 最低版本)
    dependencies = [
        # 核心深度学习框架
        ("torch", "torch", "2.0.0"),
        ("torchvision", "torchvision", "0.15.0"),
        ("torchaudio", "torchaudio", "2.0.0"),
        ("pytorch-lightning", "pytorch_lightning", "2.0.0"),
        ("torchmetrics", "torchmetrics", "1.0.0"),
        
        # 配置管理
        ("hydra-core", "hydra", "1.3.0"),
        ("omegaconf", "omegaconf", None),
        
        # EEG 专用库
        ("braindecode", "braindecode", "0.8.0"),
        ("mne", "mne", "1.3.0"),
        ("moabb", "moabb", "1.0.0"),
        
        # 数据处理
        ("numpy", "numpy", None),
        ("pandas", "pandas", None),
        ("scikit-learn", "sklearn", None),
        
        # 可视化
        ("matplotlib", "matplotlib", None),
        ("seaborn", "seaborn", "0.12.0"),
        
        # 工具库
        ("tqdm", "tqdm", None),
        ("pyyaml", "yaml", None),
        ("einops", "einops", "0.6.0"),
        
        # 版本解析（用于本脚本）
        ("packaging", "packaging", None),
    ]
    
    results = []
    missing = []
    version_issues = []
    
    for dep in dependencies:
        if len(dep) == 3:
            pkg_name, import_name, min_ver = dep
        else:
            pkg_name, import_name = dep
            min_ver = None
            
        result = check_package(pkg_name, import_name, min_ver)
        results.append(result)
        
        if not result["installed"]:
            missing.append(result)
        elif not result["version_ok"]:
            version_issues.append(result)
    
    # 打印结果表格
    print(f"{'包名':<25} {'状态':<5} {'已安装版本':<15} {'要求版本':<15}")
    print("-" * 60)
    
    for r in results:
        ver_str = r["version"] if r["version"] else "未安装"
        min_ver_str = r["min_version"] if r["min_version"] else "-"
        print(f"{r['name']:<25} {r['status']:<5} {ver_str:<15} {min_ver_str:<15}")
    
    print()
    print("=" * 60)
    
    # 汇总
    installed_count = len([r for r in results if r["installed"]])
    total_count = len(results)
    
    print(f"总计: {installed_count}/{total_count} 个包已安装")
    print()
    
    if missing:
        print("❌ 缺失的依赖:")
        for m in missing:
            print(f"   - {m['name']}")
            if 'error' in m:
                print(f"     错误: {m['error']}")
        print()
        print("安装缺失依赖的命令:")
        missing_names = [m['name'] for m in missing]
        print(f"   pip install {' '.join(missing_names)}")
        print()
        
    if version_issues:
        print("⚠️ 版本不满足要求:")
        for v in version_issues:
            print(f"   - {v['name']}: 已安装 {v['version']}, 要求 >= {v['min_version']}")
        print()
        
    if not missing and not version_issues:
        print("✅ 所有依赖已正确安装!")
    
    print("=" * 60)
    
    # 额外检查 CUDA
    print()
    print("CUDA 状态检查:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  ✅ CUDA 可用")
            print(f"     CUDA 版本: {torch.version.cuda}")
            print(f"     GPU 设备: {torch.cuda.get_device_name(0)}")
            print(f"     GPU 数量: {torch.cuda.device_count()}")
        else:
            print(f"  ⚠️ CUDA 不可用 (将使用 CPU)")
    except Exception as e:
        print(f"  ❌ 无法检查 CUDA 状态: {e}")
    
    print()
    
    # 返回状态码
    if missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())