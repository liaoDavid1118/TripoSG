import argparse
import os
import sys
from glob import glob
from typing import Any, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
sys.path.append(script_dir)

from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from image_process import prepare_image
from briarmbg import BriaRMBG

import pymeshlab


def generate_uv_coordinates(vertices, method='smart'):
    """
    为3D网格生成UV坐标映射

    Args:
        vertices: 网格顶点坐标 (N, 3)
        method: UV映射方法 ('smart', 'spherical', 'cylindrical', 'planar')

    Returns:
        uv_coords: UV坐标 (N, 2)
    """
    print(f"🗺️ 生成UV坐标映射 (方法: {method})")

    if method == 'smart':
        return smart_uv_projection(vertices)
    elif method == 'spherical':
        return spherical_projection(vertices)
    elif method == 'cylindrical':
        return cylindrical_projection(vertices)
    elif method == 'planar':
        return planar_projection(vertices)
    else:
        print(f"⚠️ 未知的UV映射方法: {method}，使用智能投影")
        return smart_uv_projection(vertices)


def smart_uv_projection(vertices):
    """智能UV投影 - 自动选择最佳投影方法"""
    center = vertices.mean(axis=0)
    centered = vertices - center

    # 计算网格的形状特征
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列

    # 计算形状比率
    ratio1 = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 1e-8 else 0
    ratio2 = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 1e-8 else 0

    print(f"  形状分析 - 比率1: {ratio1:.3f}, 比率2: {ratio2:.3f}")

    # 根据形状特征选择投影方法
    if ratio1 > 0.7 and ratio2 > 0.7:
        # 接近球形 - 使用球面投影
        print("  检测到球形物体，使用球面投影")
        return spherical_projection(vertices)
    elif ratio1 > 0.3 and ratio2 < 0.3:
        # 柱形物体 - 使用柱面投影
        print("  检测到柱形物体，使用柱面投影")
        return cylindrical_projection(vertices)
    else:
        # 扁平物体 - 使用平面投影
        print("  检测到扁平物体，使用平面投影")
        return planar_projection(vertices)


def spherical_projection(vertices):
    """改进的球面投影UV映射"""
    # 计算网格的主轴方向
    center = vertices.mean(axis=0)
    centered = vertices - center

    # 使用PCA找到主轴方向
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 按特征值排序，最大的特征向量作为主轴
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # 将顶点转换到主轴坐标系
    aligned_vertices = centered @ eigenvectors

    # 标准化到单位球
    max_dist = np.linalg.norm(aligned_vertices, axis=1).max()
    normalized = aligned_vertices / (max_dist + 1e-8)

    x, y, z = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    # 改进的球面坐标计算
    theta = np.arctan2(y, x)  # 方位角 [-π, π]

    # 计算极角，避免数值不稳定
    r_xy = np.sqrt(x**2 + y**2)
    phi = np.arctan2(r_xy, z)  # 极角 [0, π]

    # 转换为UV坐标，处理接缝问题
    u = (theta + np.pi) / (2 * np.pi)
    v = phi / np.pi

    # 处理极点附近的奇异性
    pole_threshold = 0.01
    north_pole_mask = (phi < pole_threshold)
    south_pole_mask = (phi > (np.pi - pole_threshold))

    if np.any(north_pole_mask):
        u[north_pole_mask] = 0.5
    if np.any(south_pole_mask):
        u[south_pole_mask] = 0.5

    # 确保UV坐标在[0, 1]范围内
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    return np.column_stack([u, v])


def cylindrical_projection(vertices):
    """改进的柱面投影UV映射"""
    center = vertices.mean(axis=0)
    centered = vertices - center

    # 使用PCA找到主轴方向
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 按特征值排序，最大的特征向量作为柱轴
    idx = np.argsort(eigenvalues)[::-1]
    main_axis = eigenvectors[:, idx[0]]  # 主轴作为柱轴

    # 构建旋转矩阵，将主轴对齐到Z轴
    z_axis = np.array([0, 0, 1])
    if np.abs(np.dot(main_axis, z_axis)) < 0.999:
        # 使用罗德里格斯旋转公式
        v = np.cross(main_axis, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(main_axis, z_axis)

        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    else:
        R = np.eye(3)

    # 应用旋转
    aligned_vertices = centered @ R.T

    x, y, z = aligned_vertices[:, 0], aligned_vertices[:, 1], aligned_vertices[:, 2]

    # 改进的柱面坐标计算
    theta = np.arctan2(y, x)  # 方位角
    height = z  # 高度

    # 标准化UV坐标
    u = (theta + np.pi) / (2 * np.pi)

    # 高度标准化，使用更稳定的方法
    height_range = height.max() - height.min()
    if height_range > 1e-8:
        v = (height - height.min()) / height_range
    else:
        v = np.full_like(height, 0.5)

    # 确保UV坐标在[0, 1]范围内
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    return np.column_stack([u, v])


def planar_projection(vertices, axis='auto'):
    """改进的平面投影UV映射"""
    center = vertices.mean(axis=0)
    centered = vertices - center

    # 使用PCA找到最佳投影平面
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 按特征值排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    if axis == 'auto':
        # 自动选择最佳投影方向（最小特征值对应的方向）
        u_axis = eigenvectors[:, 0]  # 最大特征值对应的特征向量
        v_axis = eigenvectors[:, 1]  # 中等特征值对应的特征向量
    else:
        # 手动指定投影轴
        if axis == 'z':
            u_axis = np.array([1, 0, 0])
            v_axis = np.array([0, 1, 0])
        elif axis == 'y':
            u_axis = np.array([1, 0, 0])
            v_axis = np.array([0, 0, 1])
        else:  # x
            u_axis = np.array([0, 1, 0])
            v_axis = np.array([0, 0, 1])

    # 投影到选定的平面
    u_coord = np.dot(centered, u_axis)
    v_coord = np.dot(centered, v_axis)

    # 标准化到[0, 1]，处理边界情况
    u_range = u_coord.max() - u_coord.min()
    v_range = v_coord.max() - v_coord.min()

    if u_range > 1e-8:
        u = (u_coord - u_coord.min()) / u_range
    else:
        u = np.full_like(u_coord, 0.5)

    if v_range > 1e-8:
        v = (v_coord - v_coord.min()) / v_range
    else:
        v = np.full_like(v_coord, 0.5)

    # 确保UV坐标在[0, 1]范围内
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    return np.column_stack([u, v])


@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_input: Union[str, Image.Image],
    rmbg_net: Any,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    faces: int = -1,
    generate_uv: bool = False,
    uv_method: str = 'smart',
) -> trimesh.Scene:

    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)

    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).samples[0]

    # 创建基础网格
    vertices = outputs[0].astype(np.float32)
    faces = np.ascontiguousarray(outputs[1])

    # 生成UV坐标（如果需要）
    if generate_uv:
        print("🗺️ 为网格生成UV坐标映射...")
        uv_coords = generate_uv_coordinates(vertices, method=uv_method)

        # 创建带UV坐标的网格
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 添加UV坐标到网格的视觉属性
        # 创建一个简单的白色材质作为占位符
        material = trimesh.visual.material.SimpleMaterial(
            diffuse=[255, 255, 255, 255]  # 白色材质
        )

        # 创建纹理视觉对象
        texture_visual = trimesh.visual.TextureVisuals(
            uv=uv_coords,
            material=material
        )

        mesh.visual = texture_visual
        print(f"✅ UV坐标生成完成 - 顶点数: {len(vertices)}, UV坐标数: {len(uv_coords)}")
    else:
        # 创建不带UV坐标的网格
        mesh = trimesh.Trimesh(vertices, faces)

    # 网格简化（如果指定了目标面数）
    if isinstance(faces, int) and faces > 0:
        print(f"🔧 简化网格到 {faces} 个面...")
        mesh = simplify_mesh(mesh, faces)

    return mesh

def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()#.tolist()
    faces = mesh.face_matrix()#.tolist()
    return trimesh.Trimesh(vertices=verts, faces=faces)  #, vID, fID

def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
    """简化网格，保持UV坐标信息"""
    if mesh.faces.shape[0] > n_faces:
        # 保存原始的视觉属性
        original_visual = mesh.visual if hasattr(mesh, 'visual') else None

        # 简化网格几何
        ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum = n_faces)
        simplified_mesh = pymesh_to_trimesh(ms.current_mesh())

        # 如果原始网格有UV坐标，需要重新生成或插值
        if original_visual and hasattr(original_visual, 'uv'):
            print("⚠️ 网格简化后重新生成UV坐标...")
            # 重新生成UV坐标（因为顶点数量已改变）
            uv_coords = generate_uv_coordinates(simplified_mesh.vertices, method='smart')

            # 恢复纹理视觉属性
            if hasattr(original_visual, 'material'):
                material = original_visual.material
            else:
                material = trimesh.visual.material.SimpleMaterial(
                    diffuse=[255, 255, 255, 255]
                )

            texture_visual = trimesh.visual.TextureVisuals(
                uv=uv_coords,
                material=material
            )
            simplified_mesh.visual = texture_visual

        return simplified_mesh
    else:
        return mesh

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    parser = argparse.ArgumentParser(description="TripoSG 3D模型生成工具")
    parser.add_argument("--image-input", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output-path", type=str, default="./output.glb", help="输出GLB文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="引导尺度")
    parser.add_argument("--faces", type=int, default=-1, help="目标面数（-1表示不简化）")

    # UV坐标相关参数
    parser.add_argument("--generate-uv", action="store_true", help="生成UV坐标映射")
    parser.add_argument("--uv-method", type=str, default="smart",
                       choices=['smart', 'spherical', 'cylindrical', 'planar'],
                       help="UV映射方法")

    args = parser.parse_args()

    # download pretrained weights
    triposg_weights_dir = "pretrained_weights/TripoSG"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

    # init rmbg model for background removal
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval() 

    # init tripoSG pipeline
    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)

    # run inference
    print(f"\n🔄 开始3D模型生成...")
    print(f"输入图像: {args.image_input}")
    print(f"输出路径: {args.output_path}")
    print(f"推理步数: {args.num_inference_steps}")
    print(f"引导尺度: {args.guidance_scale}")
    print(f"随机种子: {args.seed}")
    if args.generate_uv:
        print(f"UV映射方法: {args.uv_method}")

    mesh = run_triposg(
        pipe,
        image_input=args.image_input,
        rmbg_net=rmbg_net,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        faces=args.faces,
        generate_uv=args.generate_uv,
        uv_method=args.uv_method,
    )

    # 导出网格
    mesh.export(args.output_path)

    print(f"\n✅ 3D模型生成完成!")
    print(f"📁 输出文件: {args.output_path}")
    print(f"📊 网格信息:")
    print(f"  - 顶点数: {len(mesh.vertices)}")
    print(f"  - 面数: {len(mesh.faces)}")

    if args.generate_uv and hasattr(mesh.visual, 'uv'):
        print(f"  - UV坐标: ✅ 已生成 ({len(mesh.visual.uv)} 个)")
        print(f"  - UV映射方法: {args.uv_method}")
    else:
        print(f"  - UV坐标: ❌ 未生成")
