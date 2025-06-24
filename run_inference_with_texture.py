#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TripoSG推理脚本 - 增强版，支持纹理映射
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
import cv2

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

    return mesh, img_pil

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

def create_texture_mapping(mesh, original_image, texture_size=1024):
    """
    为3D网格创建纹理映射
    使用简单的球面投影或柱面投影
    """
    print(f"🎨 创建纹理映射 (分辨率: {texture_size}x{texture_size})")
    
    # 获取网格的边界框
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min
    
    # 将顶点标准化到[-1, 1]范围
    normalized_vertices = (mesh.vertices - bbox_center) / (bbox_size.max() / 2)
    
    # 使用球面投影计算UV坐标
    x, y, z = normalized_vertices[:, 0], normalized_vertices[:, 1], normalized_vertices[:, 2]
    
    # 球面坐标
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # 方位角
    phi = np.arccos(np.clip(z / (r + 1e-8), -1, 1))  # 极角
    
    # 转换为UV坐标 [0, 1]
    u = (theta + np.pi) / (2 * np.pi)
    v = phi / np.pi
    
    # 确保UV坐标在[0, 1]范围内
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    uv_coordinates = np.column_stack([u, v])
    
    # 创建纹理图像
    texture_image = create_texture_from_original(original_image, texture_size)
    
    return uv_coordinates, texture_image

def create_texture_from_original(original_image, texture_size):
    """
    基于原始图像创建纹理
    """
    # 将PIL图像转换为numpy数组
    if isinstance(original_image, Image.Image):
        img_array = np.array(original_image)
    else:
        img_array = original_image
    
    # 调整图像大小到纹理尺寸
    texture = cv2.resize(img_array, (texture_size, texture_size), interpolation=cv2.INTER_LANCZOS4)
    
    # 创建一个更丰富的纹理模式
    # 可以添加一些变化和细节
    texture = enhance_texture(texture)
    
    return texture

def enhance_texture(texture):
    """
    增强纹理，添加一些细节和变化
    """
    # 添加轻微的噪声来增加细节
    noise = np.random.normal(0, 5, texture.shape).astype(np.int16)
    enhanced = texture.astype(np.int16) + noise
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # 轻微的高斯模糊来平滑噪声
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
    
    return enhanced

def apply_texture_to_mesh(mesh, uv_coordinates, texture_image):
    """
    将纹理应用到网格
    """
    print("🖼️ 应用纹理到网格")
    
    # 创建纹理材质
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture_image),
        metallicFactor=0.1,
        roughnessFactor=0.8
    )
    
    # 创建纹理视觉对象
    texture_visual = trimesh.visual.TextureVisuals(
        uv=uv_coordinates,
        material=material
    )
    
    # 应用纹理到网格
    mesh.visual = texture_visual
    
    return mesh

def save_texture_files(mesh, output_path, texture_image):
    """
    保存纹理文件
    """
    output_path = Path(output_path)
    base_name = output_path.stem
    output_dir = output_path.parent
    
    # 保存纹理图像
    texture_path = output_dir / f"{base_name}_texture.png"
    Image.fromarray(texture_image).save(texture_path)
    print(f"💾 纹理图像保存到: {texture_path}")
    
    # 保存带纹理的GLB文件
    mesh.export(output_path)
    print(f"💾 带纹理的3D模型保存到: {output_path}")
    
    # 也保存OBJ格式（包含MTL文件）
    obj_path = output_dir / f"{base_name}.obj"
    mesh.export(obj_path)
    print(f"💾 OBJ格式保存到: {obj_path}")

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

    parser = argparse.ArgumentParser(description="TripoSG推理脚本 - 支持纹理映射")
    parser.add_argument("--image-input", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output-path", type=str, default="./output_textured.glb", help="输出GLB文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="引导尺度")
    parser.add_argument("--faces", type=int, default=-1, help="简化网格面数(-1表示不简化)")
    parser.add_argument("--texture-size", type=int, default=1024, help="纹理分辨率")
    parser.add_argument("--no-texture", action="store_true", help="不添加纹理，只生成几何")
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
        print(f"纹理分辨率: {args.texture_size}")

        # 运行推理
        mesh, original_image = run_triposg(
            pipe,
            image_input=args.image_input,
            rmbg_net=rmbg_net,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            faces=args.faces,
        )

        print(f"\n✅ 3D网格生成完成!")
        print(f"网格信息:")
        print(f"  - 顶点数: {len(mesh.vertices)}")
        print(f"  - 面数: {len(mesh.faces)}")

        if not args.no_texture:
            # 创建纹理映射
            uv_coordinates, texture_image = create_texture_mapping(
                mesh, original_image, args.texture_size
            )
            
            # 应用纹理到网格
            mesh = apply_texture_to_mesh(mesh, uv_coordinates, texture_image)
            
            # 保存带纹理的文件
            save_texture_files(mesh, args.output_path, texture_image)
        else:
            # 只保存几何
            print(f"\n💾 保存网格到: {args.output_path}")
            mesh.export(args.output_path)
        
        print(f"\n✅ 处理完成!")
        
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
