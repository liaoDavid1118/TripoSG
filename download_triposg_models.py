#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TripoSG大模型文件下载脚本
专门处理大文件下载，支持断点续传
"""

import os
import time
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
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

def download_large_file(repo_id, filename, local_dir, max_retries=5):
    """下载大文件，支持重试"""
    for attempt in range(max_retries):
        try:
            print(f"尝试下载 {filename} (第 {attempt + 1}/{max_retries} 次)")
            
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                resume_download=True,  # 启用断点续传
                local_dir_use_symlinks=False,  # 不使用符号链接
            )
            
            print(f"✓ {filename} 下载成功!")
            return True
            
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15  # 递增等待时间
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"✗ {filename} 下载失败，已达到最大重试次数")
                return False
    
    return False

def main():
    """主函数"""
    print("=" * 60)
    print("TripoSG大模型文件下载器")
    print("=" * 60)
    
    repo_id = "VAST-AI/TripoSG"
    local_dir = "pretrained_weights/TripoSG"
    
    # 需要下载的大文件列表
    large_files = [
        "image_encoder_dinov2/model.safetensors",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors"
    ]
    
    success_count = 0
    for filename in large_files:
        print(f"\n下载文件: {filename}")
        
        # 检查文件是否已存在
        file_path = Path(local_dir) / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 大于100MB认为是完整的
                print(f"文件已存在且完整 ({file_size} bytes)，跳过下载")
                success_count += 1
                continue
        
        if download_large_file(repo_id, filename, local_dir):
            success_count += 1
    
    print("\n" + "=" * 60)
    if success_count == len(large_files):
        print("✅ 所有大文件下载完成!")
        return True
    else:
        print(f"❌ 部分文件下载失败 ({success_count}/{len(large_files)})")
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
