#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TripoSG模型下载脚本
支持断点续传和重试功能
"""

import os
import time
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_robust_session():
    """创建一个具有重试机制的requests会话"""
    session = requests.Session()
    
    # 配置重试策略
    retry_strategy = Retry(
        total=10,  # 总重试次数
        backoff_factor=2,  # 退避因子
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # 允许重试的HTTP方法
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def download_with_retry(repo_id, local_dir, max_retries=5):
    """带重试机制的模型下载"""
    for attempt in range(max_retries):
        try:
            print(f"尝试下载 {repo_id} (第 {attempt + 1}/{max_retries} 次)")
            
            # 使用自定义会话进行下载
            session = create_robust_session()
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                resume_download=True,  # 启用断点续传
                local_dir_use_symlinks=False,  # 不使用符号链接
            )
            
            print(f"✓ {repo_id} 下载成功!")
            return True
            
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # 递增等待时间
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"✗ {repo_id} 下载失败，已达到最大重试次数")
                return False
    
    return False

def main():
    """主函数"""
    print("=" * 60)
    print("TripoSG模型下载器")
    print("=" * 60)
    
    # 创建权重目录
    weights_dir = Path("pretrained_weights")
    weights_dir.mkdir(exist_ok=True)
    
    triposg_weights_dir = weights_dir / "TripoSG"
    rmbg_weights_dir = weights_dir / "RMBG-1.4"
    
    # 下载模型
    models_to_download = [
        ("VAST-AI/TripoSG", triposg_weights_dir),
        ("briaai/RMBG-1.4", rmbg_weights_dir)
    ]
    
    success_count = 0
    for repo_id, local_dir in models_to_download:
        print(f"\n下载模型: {repo_id}")
        print(f"保存到: {local_dir}")
        
        # 检查是否已经存在
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"模型已存在，跳过下载")
            success_count += 1
            continue
        
        if download_with_retry(repo_id, str(local_dir)):
            success_count += 1
    
    print("\n" + "=" * 60)
    if success_count == len(models_to_download):
        print("✅ 所有模型下载完成!")
        return True
    else:
        print(f"❌ 部分模型下载失败 ({success_count}/{len(models_to_download)})")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n用户中断下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n程序出错: {e}")
        sys.exit(1)
