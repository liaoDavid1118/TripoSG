#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TripoSG环境验证脚本
检查所有必要的依赖是否正确安装
"""

import sys
import importlib
import torch

def check_package(package_name, import_name=None):
    """检查包是否可以导入"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', '未知版本')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {package_name}: 导入失败 - {e}")
        return False

def main():
    """主验证函数"""
    print("=" * 60)
    print("TripoSG环境验证")
    print("=" * 60)
    
    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python版本: {python_version}")
    
    # 必需的包列表
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('diffusers', 'diffusers'),
        ('transformers', 'transformers'),
        ('einops', 'einops'),
        ('huggingface_hub', 'huggingface_hub'),
        ('opencv-python', 'cv2'),
        ('trimesh', 'trimesh'),
        ('omegaconf', 'omegaconf'),
        ('scikit-image', 'skimage'),
        ('numpy', 'numpy'),
        ('peft', 'peft'),
        ('jaxtyping', 'jaxtyping'),
        ('typeguard', 'typeguard'),
        ('diso', 'diso'),
        ('pymeshlab', 'pymeshlab'),
    ]
    
    print("\n检查依赖包:")
    print("-" * 40)
    
    failed_packages = []
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            failed_packages.append(package_name)
    
    # 检查PyTorch CUDA支持
    print("\n检查PyTorch CUDA支持:")
    print("-" * 40)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.version.cuda}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 测试GPU内存
        try:
            device = torch.device('cuda:0')
            test_tensor = torch.randn(100, 100).to(device)
            print(f"✓ GPU内存测试通过")
            print(f"✓ GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        except Exception as e:
            print(f"✗ GPU内存测试失败: {e}")
            failed_packages.append("CUDA内存测试")
    else:
        print("✗ CUDA不可用")
        failed_packages.append("CUDA")
    
    # 总结
    print("\n" + "=" * 60)
    if failed_packages:
        print("❌ 环境验证失败!")
        print("失败的组件:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\n请检查安装或重新运行安装脚本")
        return False
    else:
        print("✅ 环境验证成功!")
        print("所有依赖都已正确安装，TripoSG环境可以正常使用")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
