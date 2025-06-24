#!/usr/bin/env python3
"""
TripoSG UV坐标生成示例
演示如何使用新的UV坐标生成功能
"""

import os
import sys
import argparse
from pathlib import Path

def run_example(image_path, output_dir="examples_output"):
    """运行UV坐标生成示例"""
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 获取图像文件名（不含扩展名）
    image_name = Path(image_path).stem
    
    print(f"🎯 开始TripoSG UV坐标生成示例")
    print(f"📷 输入图像: {image_path}")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)
    
    # 示例1：不生成UV坐标（原始方式）
    print(f"\n1️⃣ 生成不带UV坐标的模型（原始方式）")
    output1 = output_path / f"{image_name}_no_uv.glb"
    cmd1 = f"C:\\Users\\david\\miniconda3\\envs\\TripoSG\\python.exe scripts/inference_triposg.py --image-input {image_path} --output-path {output1}"
    print(f"命令: {cmd1}")
    os.system(cmd1)
    
    # 示例2：智能UV映射（推荐）
    print(f"\n2️⃣ 生成带智能UV坐标的模型（推荐）")
    output2 = output_path / f"{image_name}_smart_uv.glb"
    cmd2 = f"C:\\Users\\david\\miniconda3\\envs\\TripoSG\\python.exe scripts/inference_triposg.py --image-input {image_path} --output-path {output2} --generate-uv --uv-method smart"
    print(f"命令: {cmd2}")
    os.system(cmd2)
    
    # 示例3：球面投影UV映射
    print(f"\n3️⃣ 生成带球面投影UV坐标的模型")
    output3 = output_path / f"{image_name}_spherical_uv.glb"
    cmd3 = f"C:\\Users\\david\\miniconda3\\envs\\TripoSG\\python.exe scripts/inference_triposg.py --image-input {image_path} --output-path {output3} --generate-uv --uv-method spherical"
    print(f"命令: {cmd3}")
    os.system(cmd3)
    
    # 示例4：柱面投影UV映射
    print(f"\n4️⃣ 生成带柱面投影UV坐标的模型")
    output4 = output_path / f"{image_name}_cylindrical_uv.glb"
    cmd4 = f"C:\\Users\\david\\miniconda3\\envs\\TripoSG\\python.exe scripts/inference_triposg.py --image-input {image_path} --output-path {output4} --generate-uv --uv-method cylindrical"
    print(f"命令: {cmd4}")
    os.system(cmd4)
    
    print(f"\n" + "=" * 60)
    print(f"✅ 所有示例生成完成!")
    print(f"📁 输出文件位置: {output_dir}")
    print(f"📋 生成的文件:")
    
    outputs = [output1, output2, output3, output4]
    descriptions = [
        "不带UV坐标（原始）",
        "智能UV映射（推荐）", 
        "球面投影UV映射",
        "柱面投影UV映射"
    ]
    
    for output, desc in zip(outputs, descriptions):
        if output.exists():
            size_mb = output.stat().st_size / (1024 * 1024)
            print(f"  ✅ {output.name} - {desc} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {output.name} - 生成失败")
    
    print(f"\n💡 使用建议:")
    print(f"  - 对于人物/动物: 使用智能UV映射或球面投影")
    print(f"  - 对于建筑/柱状物: 使用柱面投影")
    print(f"  - 对于扁平物体: 使用平面投影")
    print(f"  - 不确定时: 使用智能UV映射（自动选择）")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TripoSG UV坐标生成示例")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output-dir", type=str, default="examples_output", help="输出目录")
    
    args = parser.parse_args()
    
    # 检查输入图像是否存在
    if not os.path.exists(args.image):
        print(f"❌ 输入图像不存在: {args.image}")
        return False
    
    try:
        return run_example(args.image, args.output_dir)
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
