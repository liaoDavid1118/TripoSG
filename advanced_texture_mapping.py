#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级纹理映射脚本
支持多种纹理映射方法和增强功能
"""

import argparse
import numpy as np
import trimesh
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from pathlib import Path
import sys

def load_mesh(mesh_path):
    """加载3D网格"""
    print(f"📂 加载3D网格: {mesh_path}")
    scene = trimesh.load(mesh_path)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.geometry[list(scene.geometry.keys())[0]]
    else:
        mesh = scene
    print(f"✅ 网格加载成功 - 顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}")
    return mesh

def spherical_projection(vertices):
    """改进的球面投影UV映射"""
    print("🌐 使用改进的球面投影")

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
    # 使用atan2确保连续性
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
    print("🏛️ 使用改进的柱面投影")

    center = vertices.mean(axis=0)
    centered = vertices - center

    # 使用PCA找到主轴方向
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 按特征值排序，最大的特征向量作为柱轴
    idx = np.argsort(eigenvalues)[::-1]
    main_axis = eigenvectors[:, idx[0]]  # 主轴作为柱轴

    # 将顶点转换到柱轴坐标系
    # 构建旋转矩阵，使主轴对齐到Z轴
    z_axis = np.array([0, 0, 1])
    if np.abs(np.dot(main_axis, z_axis)) < 0.99:
        # 计算旋转轴
        rotation_axis = np.cross(main_axis, z_axis)
        rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)

        # 计算旋转角度
        cos_angle = np.dot(main_axis, z_axis)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # 罗德里格斯旋转公式
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        aligned_vertices = centered @ R.T
    else:
        aligned_vertices = centered

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

def planar_projection(vertices, axis='z'):
    """改进的平面投影UV映射"""
    print(f"📐 使用改进的平面投影 (轴: {axis})")

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
        normal = eigenvectors[:, 2]  # 最小特征值对应的特征向量
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

def smart_uv_projection(vertices):
    """智能UV投影 - 自动选择最佳投影方法"""
    print("🧠 使用智能UV投影")

    center = vertices.mean(axis=0)
    centered = vertices - center

    # 计算网格的形状特征
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列

    # 计算形状比率
    ratio1 = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 1e-8 else 0
    ratio2 = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 1e-8 else 0

    print(f"形状分析 - 比率1: {ratio1:.3f}, 比率2: {ratio2:.3f}")

    # 根据形状特征选择投影方法
    if ratio1 > 0.7 and ratio2 > 0.7:
        # 接近球形 - 使用球面投影
        print("检测到球形物体，使用球面投影")
        return spherical_projection(vertices)
    elif ratio1 > 0.3 and ratio2 < 0.3:
        # 柱形物体 - 使用柱面投影
        print("检测到柱形物体，使用柱面投影")
        return cylindrical_projection(vertices)
    else:
        # 扁平物体 - 使用平面投影
        print("检测到扁平物体，使用平面投影")
        return planar_projection(vertices, 'auto')

def conformal_projection(vertices):
    """保角投影 - 减少角度失真"""
    print("📐 使用保角投影")

    # 使用复数平面进行保角映射
    center = vertices.mean(axis=0)
    centered = vertices - center

    # 将3D点投影到复平面
    # 使用立体投影
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]

    # 标准化
    max_dist = np.linalg.norm(centered, axis=1).max()
    x, y, z = x / max_dist, y / max_dist, z / max_dist

    # 立体投影到复平面
    denom = 1 - z + 1e-8
    w_real = x / denom
    w_imag = y / denom

    # 处理南极点附近的奇异性
    south_pole_mask = (z < -0.99)
    if np.any(south_pole_mask):
        w_real[south_pole_mask] = 0
        w_imag[south_pole_mask] = 0

    # 转换为UV坐标
    # 使用反正切函数映射到[0,1]
    u = (np.arctan(w_real) + np.pi/2) / np.pi
    v = (np.arctan(w_imag) + np.pi/2) / np.pi

    # 确保UV坐标在[0, 1]范围内
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    return np.column_stack([u, v])

def evaluate_uv_quality(vertices, faces, uv_coords, sample_size=1000):
    """快速评估UV映射质量（采样版本）"""
    print("📊 快速评估UV映射质量")

    # 对大网格进行采样以提高速度
    num_faces = len(faces)
    if num_faces > sample_size:
        sample_indices = np.random.choice(num_faces, sample_size, replace=False)
        sample_faces = faces[sample_indices]
        print(f"采样 {sample_size}/{num_faces} 个面进行质量评估")
    else:
        sample_faces = faces

    # 快速检测UV坐标的分布
    u_range = uv_coords[:, 0].max() - uv_coords[:, 0].min()
    v_range = uv_coords[:, 1].max() - uv_coords[:, 1].min()
    print(f"UV覆盖范围 - U: {u_range:.3f}, V: {v_range:.3f}")

    # 检测UV坐标的聚集程度
    u_std = np.std(uv_coords[:, 0])
    v_std = np.std(uv_coords[:, 1])
    print(f"UV分布标准差 - U: {u_std:.3f}, V: {v_std:.3f}")

    # 检测异常值
    u_outliers = np.sum((uv_coords[:, 0] < 0) | (uv_coords[:, 0] > 1))
    v_outliers = np.sum((uv_coords[:, 1] < 0) | (uv_coords[:, 1] > 1))
    if u_outliers > 0 or v_outliers > 0:
        print(f"⚠️ 检测到UV坐标超出范围: U异常值={u_outliers}, V异常值={v_outliers}")

    return {
        'uv_coverage': min(u_range, v_range),
        'uv_distribution': min(u_std, v_std),
        'u_outliers': u_outliers,
        'v_outliers': v_outliers
    }

def fix_uv_coordinates(uv_coords):
    """修复UV坐标问题"""
    print("🔧 修复UV坐标")

    fixed_uv = uv_coords.copy()

    # 将超出范围的UV坐标钳制到[0,1]
    fixed_uv[:, 0] = np.clip(fixed_uv[:, 0], 0, 1)
    fixed_uv[:, 1] = np.clip(fixed_uv[:, 1], 0, 1)

    # 检查是否有NaN或无穷大值
    nan_mask = np.isnan(fixed_uv) | np.isinf(fixed_uv)
    if np.any(nan_mask):
        print("⚠️ 检测到无效UV坐标，使用默认值替换")
        fixed_uv[nan_mask] = 0.5

    return fixed_uv

def analyze_mesh_structure(vertices, faces):
    """分析网格结构，识别主要部位"""
    print("🔍 分析网格结构和语义部位")

    center = vertices.mean(axis=0)
    centered = vertices - center

    # 计算每个顶点到中心的距离
    distances = np.linalg.norm(centered, axis=1)

    # 计算主轴方向
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    main_axes = eigenvectors[:, idx]

    # 定义主要方向
    primary_axis = main_axes[:, 0]  # 主轴（通常是身体长轴）
    secondary_axis = main_axes[:, 1]  # 次轴（通常是宽度方向）
    tertiary_axis = main_axes[:, 2]  # 第三轴（通常是厚度方向）

    # 计算顶点在各个轴上的投影
    primary_proj = np.dot(centered, primary_axis)
    secondary_proj = np.dot(centered, secondary_axis)
    tertiary_proj = np.dot(centered, tertiary_axis)

    # 识别前后、左右、上下
    front_back_axis = primary_axis
    left_right_axis = secondary_axis
    up_down_axis = tertiary_axis

    # 确定正方向（假设模型面向+X方向）
    if abs(primary_axis[0]) > abs(primary_axis[2]):
        front_back_axis = primary_axis
        up_down_axis = tertiary_axis
    else:
        front_back_axis = tertiary_axis
        up_down_axis = primary_axis

    return {
        'center': center,
        'main_axes': main_axes,
        'front_back_axis': front_back_axis,
        'left_right_axis': left_right_axis,
        'up_down_axis': up_down_axis,
        'primary_proj': primary_proj,
        'secondary_proj': secondary_proj,
        'tertiary_proj': tertiary_proj,
        'distances': distances
    }

def segment_mesh_by_regions(vertices, faces, structure_info):
    """将网格分割为语义区域"""
    print("🎯 分割网格为语义区域")

    center = structure_info['center']
    centered = vertices - center

    # 计算各轴投影
    front_back_proj = np.dot(centered, structure_info['front_back_axis'])
    left_right_proj = np.dot(centered, structure_info['left_right_axis'])
    up_down_proj = np.dot(centered, structure_info['up_down_axis'])

    # 标准化投影值
    fb_norm = (front_back_proj - front_back_proj.min()) / (front_back_proj.max() - front_back_proj.min() + 1e-8)
    lr_norm = (left_right_proj - left_right_proj.min()) / (left_right_proj.max() - left_right_proj.min() + 1e-8)
    ud_norm = (up_down_proj - up_down_proj.min()) / (up_down_proj.max() - up_down_proj.min() + 1e-8)

    # 定义区域
    regions = {}

    # 前面区域（脸部、胸部）
    front_mask = fb_norm > 0.6
    regions['front'] = front_mask

    # 后面区域（背部）
    back_mask = fb_norm < 0.4
    regions['back'] = back_mask

    # 侧面区域
    left_mask = (lr_norm < 0.3) & (~front_mask) & (~back_mask)
    right_mask = (lr_norm > 0.7) & (~front_mask) & (~back_mask)
    regions['left'] = left_mask
    regions['right'] = right_mask

    # 顶部区域（头部）
    top_mask = ud_norm > 0.7
    regions['top'] = top_mask

    # 底部区域（脚部）
    bottom_mask = ud_norm < 0.3
    regions['bottom'] = bottom_mask

    # 中间区域
    middle_mask = ~(front_mask | back_mask | left_mask | right_mask | top_mask | bottom_mask)
    regions['middle'] = middle_mask

    print(f"区域分割结果:")
    for region_name, mask in regions.items():
        count = np.sum(mask)
        percentage = count / len(vertices) * 100
        print(f"  {region_name}: {count} 顶点 ({percentage:.1f}%)")

    return regions

def semantic_uv_mapping(vertices, faces, original_image, structure_info, regions):
    """基于语义的UV映射"""
    print("🧠 执行语义感知UV映射")

    uv_coords = np.zeros((len(vertices), 2))

    # 为不同区域分配不同的UV空间
    uv_regions = {
        'front': (0.0, 0.5, 0.0, 1.0),    # 左半部分
        'back': (0.5, 1.0, 0.0, 1.0),     # 右半部分
        'top': (0.0, 0.5, 0.0, 0.5),      # 左上
        'bottom': (0.0, 0.5, 0.5, 1.0),   # 左下
        'left': (0.5, 0.75, 0.0, 0.5),    # 右上左
        'right': (0.75, 1.0, 0.0, 0.5),   # 右上右
        'middle': (0.5, 1.0, 0.5, 1.0)    # 右下
    }

    center = structure_info['center']
    centered = vertices - center

    for region_name, mask in regions.items():
        if not np.any(mask):
            continue

        region_vertices = vertices[mask]
        region_centered = region_vertices - center

        # 获取该区域的UV空间范围
        u_min, u_max, v_min, v_max = uv_regions.get(region_name, (0, 1, 0, 1))

        if region_name in ['front', 'back']:
            # 前后面使用改进的平面投影
            if region_name == 'front':
                # 前面：投影到YZ平面
                u_coord = np.dot(region_centered, structure_info['left_right_axis'])
                v_coord = np.dot(region_centered, structure_info['up_down_axis'])
            else:
                # 后面：投影到YZ平面，但翻转U坐标
                u_coord = -np.dot(region_centered, structure_info['left_right_axis'])
                v_coord = np.dot(region_centered, structure_info['up_down_axis'])

        elif region_name in ['left', 'right']:
            # 侧面使用柱面投影
            if region_name == 'left':
                u_coord = np.dot(region_centered, structure_info['front_back_axis'])
                v_coord = np.dot(region_centered, structure_info['up_down_axis'])
            else:
                u_coord = -np.dot(region_centered, structure_info['front_back_axis'])
                v_coord = np.dot(region_centered, structure_info['up_down_axis'])

        elif region_name in ['top', 'bottom']:
            # 顶部和底部使用平面投影
            u_coord = np.dot(region_centered, structure_info['left_right_axis'])
            v_coord = np.dot(region_centered, structure_info['front_back_axis'])

        else:  # middle
            # 中间区域使用球面投影
            x, y, z = region_centered[:, 0], region_centered[:, 1], region_centered[:, 2]
            r = np.linalg.norm(region_centered, axis=1) + 1e-8
            theta = np.arctan2(y, x)
            phi = np.arccos(np.clip(z / r, -1, 1))
            u_coord = (theta + np.pi) / (2 * np.pi)
            v_coord = phi / np.pi

        # 标准化到区域UV空间
        if len(u_coord) > 0:
            u_range = u_coord.max() - u_coord.min()
            v_range = v_coord.max() - v_coord.min()

            if u_range > 1e-8:
                u_norm = (u_coord - u_coord.min()) / u_range
            else:
                u_norm = np.full_like(u_coord, 0.5)

            if v_range > 1e-8:
                v_norm = (v_coord - v_coord.min()) / v_range
            else:
                v_norm = np.full_like(v_coord, 0.5)

            # 映射到分配的UV区域
            u_final = u_min + u_norm * (u_max - u_min)
            v_final = v_min + v_norm * (v_max - v_min)

            uv_coords[mask, 0] = u_final
            uv_coords[mask, 1] = v_final

    return uv_coords

def create_semantic_texture(original_image, regions_info, texture_size=1024):
    """创建语义感知的纹理布局"""
    print("🎨 创建语义感知纹理布局")

    # 创建纹理画布
    texture = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 128

    if isinstance(original_image, str):
        img = Image.open(original_image)
    else:
        img = original_image

    # 确保图像是RGB格式
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = np.array(img)

    # 定义不同区域的纹理处理
    region_configs = {
        'front': {
            'area': (0, texture_size//2, 0, texture_size),
            'source_area': (0.2, 0.8, 0.1, 0.9),  # 主要面部/胸部区域
            'enhancement': 'face'
        },
        'back': {
            'area': (texture_size//2, texture_size, 0, texture_size),
            'source_area': (0.1, 0.9, 0.2, 0.8),  # 背部区域
            'enhancement': 'body'
        },
        'top': {
            'area': (0, texture_size//2, 0, texture_size//2),
            'source_area': (0.3, 0.7, 0.0, 0.4),  # 头部区域
            'enhancement': 'head'
        },
        'bottom': {
            'area': (0, texture_size//2, texture_size//2, texture_size),
            'source_area': (0.3, 0.7, 0.6, 1.0),  # 脚部区域
            'enhancement': 'feet'
        }
    }

    for region_name, config in region_configs.items():
        # 获取目标区域
        y1, y2, x1, x2 = config['area']

        # 获取源图像区域
        sy1, sy2, sx1, sx2 = config['source_area']
        src_y1 = int(sy1 * img_array.shape[0])
        src_y2 = int(sy2 * img_array.shape[0])
        src_x1 = int(sx1 * img_array.shape[1])
        src_x2 = int(sx2 * img_array.shape[1])

        # 提取并调整源区域
        src_region = img_array[src_y1:src_y2, src_x1:src_x2]

        if src_region.size > 0:
            # 调整大小到目标区域
            target_h, target_w = y2 - y1, x2 - x1
            resized_region = cv2.resize(src_region, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            # 根据区域类型进行增强
            if config['enhancement'] == 'face':
                # 面部区域：增强细节
                resized_region = enhance_face_region(resized_region)
            elif config['enhancement'] == 'head':
                # 头部区域：保持原色调
                resized_region = enhance_head_region(resized_region)
            elif config['enhancement'] == 'body':
                # 身体区域：柔化处理
                resized_region = enhance_body_region(resized_region)

            # 应用到纹理
            texture[y1:y2, x1:x2] = resized_region

    return Image.fromarray(texture)

def enhance_face_region(region):
    """增强面部区域"""
    # 增强对比度和锐度
    enhanced = cv2.convertScaleAbs(region, alpha=1.1, beta=5)
    # 轻微锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel * 0.1 + np.eye(3) * 0.9)
    return enhanced

def enhance_head_region(region):
    """增强头部区域"""
    # 保持自然色调，轻微增强
    enhanced = cv2.convertScaleAbs(region, alpha=1.05, beta=2)
    return enhanced

def enhance_body_region(region):
    """增强身体区域"""
    # 柔化处理
    enhanced = cv2.GaussianBlur(region, (3, 3), 0.5)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=0.95, beta=0)
    return enhanced

def create_enhanced_texture(original_image, texture_size=1024, style='realistic'):
    """创建增强纹理"""
    print(f"🎨 创建增强纹理 (风格: {style}, 尺寸: {texture_size})")
    
    # 加载并调整原始图像大小
    if isinstance(original_image, str):
        img = Image.open(original_image)
    else:
        img = original_image
    
    # 调整大小
    img = img.resize((texture_size, texture_size), Image.Resampling.LANCZOS)
    
    if style == 'realistic':
        # 现实风格：增强细节和对比度
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
    elif style == 'artistic':
        # 艺术风格：添加滤镜效果
        img = img.filter(ImageFilter.EDGE_ENHANCE)
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.3)
        
    elif style == 'vintage':
        # 复古风格：降低饱和度，添加暖色调
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.8)
        
        # 添加暖色调滤镜
        img_array = np.array(img)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.1, 0, 255)  # 增强红色
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.9, 0, 255)  # 减少蓝色
        img = Image.fromarray(img_array.astype(np.uint8))
        
    elif style == 'cartoon':
        # 卡通风格：减少细节，增强颜色
        img_array = np.array(img)
        
        # 双边滤波减少细节
        img_array = cv2.bilateralFilter(img_array, 15, 80, 80)
        
        # 增强颜色
        enhancer = ImageEnhance.Color(Image.fromarray(img_array))
        img = enhancer.enhance(1.5)
    
    # 添加细微的纹理噪声
    img_array = np.array(img)
    noise = np.random.normal(0, 3, img_array.shape).astype(np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def create_procedural_texture(texture_size=1024, pattern='wood'):
    """创建程序化纹理"""
    print(f"🔧 创建程序化纹理 (模式: {pattern})")
    
    if pattern == 'wood':
        # 木纹纹理
        texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        
        # 基础颜色
        base_color = np.array([139, 69, 19])  # 棕色
        
        for i in range(texture_size):
            for j in range(texture_size):
                # 创建木纹图案
                distance = np.sqrt((i - texture_size//2)**2 + (j - texture_size//2)**2)
                ring = int(distance / 20) % 2
                noise = np.random.normal(0, 10)
                
                if ring == 0:
                    color = base_color + noise
                else:
                    color = base_color * 0.8 + noise
                
                texture[i, j] = np.clip(color, 0, 255)
        
    elif pattern == 'marble':
        # 大理石纹理
        texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        
        for i in range(texture_size):
            for j in range(texture_size):
                # 创建大理石图案
                x, y = i / texture_size, j / texture_size
                value = np.sin(x * 10 + y * 10) * 0.5 + 0.5
                value += np.random.normal(0, 0.1)
                value = np.clip(value, 0, 1)
                
                # 白色到灰色渐变
                color = np.array([255, 255, 255]) * value + np.array([100, 100, 100]) * (1 - value)
                texture[i, j] = np.clip(color, 0, 255)
    
    elif pattern == 'metal':
        # 金属纹理
        texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        
        for i in range(texture_size):
            for j in range(texture_size):
                # 创建金属图案
                noise = np.random.normal(0.5, 0.1)
                value = np.clip(noise, 0, 1)
                
                # 银色金属
                color = np.array([192, 192, 192]) * value + np.array([128, 128, 128]) * (1 - value)
                texture[i, j] = np.clip(color, 0, 255)
    
    return Image.fromarray(texture)

def apply_advanced_texture(mesh, uv_coordinates, texture_image, material_type='pbr'):
    """应用高级纹理材质"""
    print(f"🖼️ 应用高级纹理 (材质类型: {material_type})")
    
    if material_type == 'pbr':
        # PBR材质
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture_image,
            metallicFactor=0.1,
            roughnessFactor=0.7,
            emissiveFactor=[0.0, 0.0, 0.0]
        )
    elif material_type == 'metallic':
        # 金属材质
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture_image,
            metallicFactor=0.9,
            roughnessFactor=0.3
        )
    elif material_type == 'glossy':
        # 光泽材质
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture_image,
            metallicFactor=0.0,
            roughnessFactor=0.1
        )
    else:
        # 简单材质
        material = trimesh.visual.material.SimpleMaterial(
            image=texture_image
        )
    
    # 创建纹理视觉对象
    texture_visual = trimesh.visual.TextureVisuals(
        uv=uv_coordinates,
        material=material
    )
    
    mesh.visual = texture_visual
    return mesh

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高级纹理映射工具")
    parser.add_argument("--mesh", type=str, required=True, help="输入3D网格文件")
    parser.add_argument("--image", type=str, help="纹理图像文件")
    parser.add_argument("--output", type=str, default="textured_output.glb", help="输出文件")
    parser.add_argument("--projection", choices=['semantic', 'smart', 'spherical', 'cylindrical', 'planar_z', 'planar_y', 'planar_x', 'planar_auto', 'conformal'],
                       default='semantic', help="UV投影方法")
    parser.add_argument("--texture-size", type=int, default=1024, help="纹理分辨率")
    parser.add_argument("--style", choices=['realistic', 'artistic', 'vintage', 'cartoon'], 
                       default='realistic', help="纹理风格")
    parser.add_argument("--material", choices=['pbr', 'metallic', 'glossy', 'simple'], 
                       default='pbr', help="材质类型")
    parser.add_argument("--procedural", choices=['wood', 'marble', 'metal'], help="使用程序化纹理")
    
    args = parser.parse_args()
    
    try:
        # 加载网格
        mesh = load_mesh(args.mesh)
        
        # 选择UV投影方法
        if args.projection == 'semantic':
            # 语义感知UV映射
            structure_info = analyze_mesh_structure(mesh.vertices, mesh.faces)
            regions = segment_mesh_by_regions(mesh.vertices, mesh.faces, structure_info)
            uv_coords = semantic_uv_mapping(mesh.vertices, mesh.faces, args.image, structure_info, regions)

            # 使用语义感知纹理
            if args.image:
                texture_image = create_semantic_texture(args.image, regions, args.texture_size)
            else:
                print("⚠️ 语义映射需要输入图像")
                return False

        elif args.projection == 'smart':
            uv_coords = smart_uv_projection(mesh.vertices)
        elif args.projection == 'spherical':
            uv_coords = spherical_projection(mesh.vertices)
        elif args.projection == 'cylindrical':
            uv_coords = cylindrical_projection(mesh.vertices)
        elif args.projection == 'conformal':
            uv_coords = conformal_projection(mesh.vertices)
        elif args.projection.startswith('planar'):
            if args.projection == 'planar_auto':
                axis = 'auto'
            else:
                axis = args.projection.split('_')[1]
            uv_coords = planar_projection(mesh.vertices, axis)

        # 评估和修复UV映射质量
        quality_metrics = evaluate_uv_quality(mesh.vertices, mesh.faces, uv_coords)

        # 修复UV坐标问题
        if quality_metrics['u_outliers'] > 0 or quality_metrics['v_outliers'] > 0:
            print("⚠️ 检测到UV坐标超出范围，进行修复...")
            uv_coords = fix_uv_coordinates(uv_coords)
            print("✅ UV坐标修复完成")

        # 创建纹理（如果还没有创建）
        if args.projection != 'semantic':
            if args.procedural:
                texture_image = create_procedural_texture(args.texture_size, args.procedural)
            elif args.image:
                texture_image = create_enhanced_texture(args.image, args.texture_size, args.style)
            else:
                print("❌ 请提供纹理图像或选择程序化纹理")
                return False
        
        # 应用纹理
        textured_mesh = apply_advanced_texture(mesh, uv_coords, texture_image, args.material)
        
        # 保存结果
        output_path = Path(args.output)
        textured_mesh.export(args.output)
        
        # 保存纹理图像
        texture_path = output_path.parent / f"{output_path.stem}_texture.png"
        texture_image.save(texture_path)
        
        print(f"\n✅ 纹理映射完成!")
        print(f"📁 输出文件: {args.output}")
        print(f"🖼️ 纹理文件: {texture_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
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
