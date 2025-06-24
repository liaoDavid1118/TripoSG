#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TripoSG推理脚本 - 使用本地预训练模型
"""

import argparse
import os
import sys
from glob import glob
from typing import Any, Union
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

# 添加scripts目录到路径
scripts_dir = Path(__file__).parent / "scripts"
sys.path.append(str(scripts_dir))

from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from scripts.image_process import prepare_image
from scripts.briarmbg import BriaRMBG

import pymeshlab


@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_input: Union[str, Image.Image],
    rmbg_net: Any,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    faces: int = -1,
) -> trimesh.Scene:

    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)

    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).samples[0]
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    if faces > 0:
        mesh = simplify_mesh(mesh, faces)

    return mesh

def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    return trimesh.Trimesh(vertices=verts, faces=faces)

def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
    if mesh.faces.shape[0] > n_faces:
        ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum = n_faces)
        return pymesh_to_trimesh(ms.current_mesh())
    else:
        return mesh

def check_models():
    """检查预训练模型是否存在"""
    triposg_weights_dir = Path("pretrained_weights/TripoSG")
    rmbg_weights_dir = Path("pretrained_weights/RMBG-1.4")
    
    if not triposg_weights_dir.exists():
        print(f"❌ TripoSG模型不存在: {triposg_weights_dir}")
        print("请先运行: python download_models.py")
        return False
    
    if not rmbg_weights_dir.exists():
        print(f"❌ RMBG模型不存在: {rmbg_weights_dir}")
        print("请先运行: python download_models.py")
        return False
    
    print("✅ 预训练模型检查通过")
    return True

def main():
    """主函数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"使用设备: {device}")
    print(f"数据类型: {dtype}")

    parser = argparse.ArgumentParser(description="TripoSG推理脚本")
    parser.add_argument("--image-input", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output-path", type=str, default="./output.glb", help="输出GLB文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="引导尺度")
    parser.add_argument("--faces", type=int, default=-1, help="简化网格面数(-1表示不简化)")
    args = parser.parse_args()

    # 检查输入图像
    if not os.path.exists(args.image_input):
        print(f"❌ 输入图像不存在: {args.image_input}")
        return False

    # 检查模型
    if not check_models():
        return False

    # 模型路径
    triposg_weights_dir = "pretrained_weights/TripoSG"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"

    try:
        print("\n🔄 初始化RMBG模型...")
        # 尝试直接加载model.pth文件
        from pathlib import Path

        model_path = Path(rmbg_weights_dir) / "model.pth"
        if model_path.exists():
            print(f"从 {model_path} 加载模型权重")
            rmbg_net = BriaRMBG()
            state_dict = torch.load(model_path, map_location=device)
            rmbg_net.load_state_dict(state_dict)
            rmbg_net = rmbg_net.to(device)
        else:
            # 回退到from_pretrained方法
            rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)

        rmbg_net.eval()
        print("✅ RMBG模型加载成功")

        print("\n🔄 初始化TripoSG管道...")
        pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)
        print("✅ TripoSG管道加载成功")

        print(f"\n🔄 开始推理...")
        print(f"输入图像: {args.image_input}")
        print(f"输出路径: {args.output_path}")
        print(f"推理步数: {args.num_inference_steps}")
        print(f"引导尺度: {args.guidance_scale}")
        print(f"随机种子: {args.seed}")

        # 运行推理
        mesh = run_triposg(
            pipe,
            image_input=args.image_input,
            rmbg_net=rmbg_net,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            faces=args.faces,
        )

        # 保存结果
        print(f"\n💾 保存网格到: {args.output_path}")
        mesh.export(args.output_path)
        
        print(f"✅ 推理完成!")
        print(f"网格信息:")
        print(f"  - 顶点数: {len(mesh.vertices)}")
        print(f"  - 面数: {len(mesh.faces)}")
        print(f"  - 输出文件: {args.output_path}")
        
        return True

    except Exception as e:
        print(f"❌ 推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n程序出错: {e}")
        sys.exit(1)
