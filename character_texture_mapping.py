#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人物/动物专用纹理映射系统
解决眼睛贴到背部、牙齿贴到腹部等错位问题
"""

import argparse
import numpy as np
import trimesh
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import cv2
from pathlib import Path
import sys

def detect_character_orientation(vertices):
    """检测人物/动物的朝向和姿态"""
    print("🧭 检测角色朝向和姿态")
    
    center = vertices.mean(axis=0)
    centered = vertices - center
    
    # PCA分析主要方向
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    
    # 主轴（通常是身体长轴）
    primary_axis = eigenvectors[:, idx[0]]
    secondary_axis = eigenvectors[:, idx[1]]
    tertiary_axis = eigenvectors[:, idx[2]]
    
    # 判断哪个轴是垂直轴（身高方向）
    # 通常垂直轴的Y分量最大
    axes = [primary_axis, secondary_axis, tertiary_axis]
    y_components = [abs(axis[1]) for axis in axes]
    vertical_idx = np.argmax(y_components)
    
    if vertical_idx == 0:
        height_axis = primary_axis
        front_axis = secondary_axis
        side_axis = tertiary_axis
    elif vertical_idx == 1:
        height_axis = secondary_axis
        front_axis = primary_axis
        side_axis = tertiary_axis
    else:
        height_axis = tertiary_axis
        front_axis = primary_axis
        side_axis = secondary_axis
    
    # 确保高度轴指向上方
    if height_axis[1] < 0:
        height_axis = -height_axis
    
    print(f"检测结果:")
    print(f"  高度轴: {height_axis}")
    print(f"  前后轴: {front_axis}")
    print(f"  左右轴: {side_axis}")
    
    return {
        'center': center,
        'height_axis': height_axis,
        'front_axis': front_axis,
        'side_axis': side_axis,
        'eigenvalues': eigenvalues[idx],
        'aspect_ratios': eigenvalues[idx] / eigenvalues[idx[0]]
    }

def segment_character_parts(vertices, orientation_info):
    """分割角色的身体部位"""
    print("🎭 分割角色身体部位")
    
    center = orientation_info['center']
    centered = vertices - center
    
    # 计算在各个轴上的投影
    height_proj = np.dot(centered, orientation_info['height_axis'])
    front_proj = np.dot(centered, orientation_info['front_axis'])
    side_proj = np.dot(centered, orientation_info['side_axis'])
    
    # 标准化投影值到[0,1]
    height_norm = (height_proj - height_proj.min()) / (height_proj.max() - height_proj.min() + 1e-8)
    front_norm = (front_proj - front_proj.min()) / (front_proj.max() - front_proj.min() + 1e-8)
    side_norm = (side_proj - side_proj.min()) / (side_proj.max() - side_proj.min() + 1e-8)
    
    # 计算到中心的距离（用于识别四肢）
    distances = np.linalg.norm(centered, axis=1)
    dist_norm = distances / distances.max()
    
    parts = {}
    
    # 头部：顶部区域
    head_mask = height_norm > 0.8
    parts['head'] = head_mask
    
    # 脸部：头部的前面部分
    face_mask = head_mask & (front_norm > 0.6)
    parts['face'] = face_mask
    
    # 后脑：头部的后面部分
    back_head_mask = head_mask & (front_norm < 0.4)
    parts['back_head'] = back_head_mask
    
    # 躯干：中间高度区域
    torso_mask = (height_norm >= 0.3) & (height_norm <= 0.8) & (dist_norm < 0.7)
    parts['torso'] = torso_mask
    
    # 胸部：躯干的前面
    chest_mask = torso_mask & (front_norm > 0.6)
    parts['chest'] = chest_mask
    
    # 背部：躯干的后面
    back_mask = torso_mask & (front_norm < 0.4)
    parts['back'] = back_mask
    
    # 左臂：左侧突出部分
    left_arm_mask = (side_norm < 0.2) & (height_norm > 0.4) & (dist_norm > 0.5)
    parts['left_arm'] = left_arm_mask
    
    # 右臂：右侧突出部分
    right_arm_mask = (side_norm > 0.8) & (height_norm > 0.4) & (dist_norm > 0.5)
    parts['right_arm'] = right_arm_mask
    
    # 腿部：底部区域
    legs_mask = height_norm < 0.3
    parts['legs'] = legs_mask
    
    # 左腿
    left_leg_mask = legs_mask & (side_norm < 0.4)
    parts['left_leg'] = left_leg_mask
    
    # 右腿
    right_leg_mask = legs_mask & (side_norm > 0.6)
    parts['right_leg'] = right_leg_mask
    
    # 侧面（肩膀、腰部等）
    left_side_mask = (side_norm < 0.3) & (~left_arm_mask) & (~left_leg_mask) & (height_norm > 0.3)
    right_side_mask = (side_norm > 0.7) & (~right_arm_mask) & (~right_leg_mask) & (height_norm > 0.3)
    parts['left_side'] = left_side_mask
    parts['right_side'] = right_side_mask
    
    # 打印分割结果
    print("身体部位分割结果:")
    total_vertices = len(vertices)
    for part_name, mask in parts.items():
        count = np.sum(mask)
        percentage = count / total_vertices * 100
        print(f"  {part_name}: {count} 顶点 ({percentage:.1f}%)")
    
    return parts

def create_character_uv_layout(vertices, parts, orientation_info):
    """为角色创建专门的UV布局"""
    print("🗺️ 创建角色专用UV布局")
    
    uv_coords = np.zeros((len(vertices), 2))
    
    # 定义UV布局：将不同身体部位分配到纹理的不同区域
    uv_layout = {
        # 主要区域：脸部和胸部（最重要的部分）
        'face': (0.0, 0.5, 0.5, 1.0),      # 左上角 - 脸部
        'chest': (0.0, 0.5, 0.0, 0.5),     # 左下角 - 胸部
        
        # 次要区域：背部和头部
        'back': (0.5, 1.0, 0.0, 0.5),      # 右下角 - 背部
        'back_head': (0.5, 0.75, 0.5, 0.75), # 右上小块 - 后脑
        
        # 四肢区域
        'left_arm': (0.75, 1.0, 0.5, 0.75),   # 右上角小块
        'right_arm': (0.75, 1.0, 0.75, 1.0),  # 右上角小块
        'left_leg': (0.5, 0.75, 0.75, 1.0),   # 右上中块
        'right_leg': (0.75, 1.0, 0.25, 0.5),  # 右中块
        
        # 侧面和其他
        'left_side': (0.5, 0.625, 0.5, 0.75),
        'right_side': (0.625, 0.75, 0.5, 0.75),
        'torso': (0.25, 0.5, 0.25, 0.5),      # 中间小块
        'legs': (0.5, 0.75, 0.25, 0.5),       # 腿部总体
        'head': (0.25, 0.5, 0.5, 0.75),       # 头部总体
    }
    
    center = orientation_info['center']
    centered = vertices - center
    
    for part_name, mask in parts.items():
        if not np.any(mask) or part_name not in uv_layout:
            continue
            
        part_vertices = vertices[mask]
        part_centered = part_vertices - center
        
        # 获取UV区域
        u_min, u_max, v_min, v_max = uv_layout[part_name]
        
        # 根据身体部位选择最佳投影方法
        if part_name in ['face', 'chest']:
            # 脸部和胸部：使用前向平面投影
            u_coord = np.dot(part_centered, orientation_info['side_axis'])
            v_coord = np.dot(part_centered, orientation_info['height_axis'])
            
        elif part_name in ['back', 'back_head']:
            # 背部：使用后向平面投影（翻转左右）
            u_coord = -np.dot(part_centered, orientation_info['side_axis'])
            v_coord = np.dot(part_centered, orientation_info['height_axis'])
            
        elif part_name in ['left_side', 'right_side']:
            # 侧面：使用侧向投影
            u_coord = np.dot(part_centered, orientation_info['front_axis'])
            v_coord = np.dot(part_centered, orientation_info['height_axis'])
            
        elif part_name in ['left_arm', 'right_arm', 'left_leg', 'right_leg']:
            # 四肢：使用柱面投影
            if 'arm' in part_name:
                # 手臂：以肩膀为中心的柱面投影
                main_axis = orientation_info['height_axis']
            else:
                # 腿部：以髋部为中心的柱面投影
                main_axis = orientation_info['height_axis']
            
            # 计算柱面坐标
            perp_proj = part_centered - np.outer(np.dot(part_centered, main_axis), main_axis)
            theta = np.arctan2(np.dot(perp_proj, orientation_info['side_axis']), 
                              np.dot(perp_proj, orientation_info['front_axis']))
            
            u_coord = (theta + np.pi) / (2 * np.pi)
            v_coord = np.dot(part_centered, main_axis)
            
        else:
            # 其他部位：使用球面投影
            r = np.linalg.norm(part_centered, axis=1) + 1e-8
            theta = np.arctan2(np.dot(part_centered, orientation_info['side_axis']),
                              np.dot(part_centered, orientation_info['front_axis']))
            phi = np.arccos(np.clip(np.dot(part_centered, orientation_info['height_axis']) / r, -1, 1))
            
            u_coord = (theta + np.pi) / (2 * np.pi)
            v_coord = phi / np.pi
        
        # 标准化到[0,1]
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

def create_character_texture(original_image, parts, texture_size=1024):
    """为角色创建专门的纹理布局"""
    print("🎨 创建角色专用纹理")
    
    # 确保图像是RGB格式
    if isinstance(original_image, str):
        img = Image.open(original_image)
    else:
        img = original_image
    
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # 创建纹理画布
    texture = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 128
    
    # 定义身体部位对应的源图像区域
    source_regions = {
        'face': (0.25, 0.75, 0.0, 0.6),     # 脸部区域：中上部
        'chest': (0.2, 0.8, 0.3, 0.8),      # 胸部区域：中部
        'back': (0.1, 0.9, 0.2, 0.9),       # 背部：使用身体纹理
        'back_head': (0.3, 0.7, 0.0, 0.3),  # 后脑：头发区域
        'left_arm': (0.0, 0.3, 0.3, 0.8),   # 左臂：左侧
        'right_arm': (0.7, 1.0, 0.3, 0.8),  # 右臂：右侧
        'left_leg': (0.2, 0.5, 0.6, 1.0),   # 左腿：下部左侧
        'right_leg': (0.5, 0.8, 0.6, 1.0),  # 右腿：下部右侧
    }
    
    # UV布局（与create_character_uv_layout中的定义一致）
    uv_layout = {
        'face': (0.0, 0.5, 0.5, 1.0),
        'chest': (0.0, 0.5, 0.0, 0.5),
        'back': (0.5, 1.0, 0.0, 0.5),
        'back_head': (0.5, 0.75, 0.5, 0.75),
        'left_arm': (0.75, 1.0, 0.5, 0.75),
        'right_arm': (0.75, 1.0, 0.75, 1.0),
        'left_leg': (0.5, 0.75, 0.75, 1.0),
        'right_leg': (0.75, 1.0, 0.25, 0.5),
    }
    
    for part_name in source_regions.keys():
        if part_name in uv_layout:
            # 获取源区域
            sx1, sx2, sy1, sy2 = source_regions[part_name]
            src_x1, src_x2 = int(sx1 * w), int(sx2 * w)
            src_y1, src_y2 = int(sy1 * h), int(sy2 * h)
            
            # 获取目标区域
            u_min, u_max, v_min, v_max = uv_layout[part_name]
            dst_x1, dst_x2 = int(u_min * texture_size), int(u_max * texture_size)
            dst_y1, dst_y2 = int(v_min * texture_size), int(v_max * texture_size)
            
            # 提取源区域
            if src_x2 > src_x1 and src_y2 > src_y1:
                src_region = img_array[src_y1:src_y2, src_x1:src_x2]
                
                if src_region.size > 0:
                    # 调整大小
                    target_w, target_h = dst_x2 - dst_x1, dst_y2 - dst_y1
                    if target_w > 0 and target_h > 0:
                        resized = cv2.resize(src_region, (target_w, target_h), 
                                           interpolation=cv2.INTER_LANCZOS4)
                        
                        # 根据部位进行特殊处理
                        if part_name == 'face':
                            # 脸部：增强细节和对比度
                            resized = cv2.convertScaleAbs(resized, alpha=1.15, beta=10)
                        elif part_name in ['left_arm', 'right_arm', 'left_leg', 'right_leg']:
                            # 四肢：柔化处理
                            resized = cv2.GaussianBlur(resized, (3, 3), 0.5)
                        
                        # 应用到纹理
                        texture[dst_y1:dst_y2, dst_x1:dst_x2] = resized
    
    return Image.fromarray(texture)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="人物/动物专用纹理映射")
    parser.add_argument("--mesh", type=str, required=True, help="输入3D网格文件")
    parser.add_argument("--image", type=str, required=True, help="输入图像文件")
    parser.add_argument("--output", type=str, default="character_textured.glb", help="输出文件")
    parser.add_argument("--texture-size", type=int, default=1024, help="纹理分辨率")
    
    args = parser.parse_args()
    
    try:
        # 加载网格
        print(f"📂 加载3D网格: {args.mesh}")
        scene = trimesh.load(args.mesh)
        if isinstance(scene, trimesh.Scene):
            mesh = scene.geometry[list(scene.geometry.keys())[0]]
        else:
            mesh = scene
        print(f"✅ 网格加载成功 - 顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}")
        
        # 检测角色朝向
        orientation_info = detect_character_orientation(mesh.vertices)
        
        # 分割身体部位
        parts = segment_character_parts(mesh.vertices, orientation_info)
        
        # 创建UV布局
        uv_coords = create_character_uv_layout(mesh.vertices, parts, orientation_info)
        
        # 创建专用纹理
        texture_image = create_character_texture(args.image, parts, args.texture_size)
        
        # 应用纹理
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture_image,
            metallicFactor=0.1,
            roughnessFactor=0.8
        )
        
        texture_visual = trimesh.visual.TextureVisuals(
            uv=uv_coords,
            material=material
        )
        
        mesh.visual = texture_visual
        
        # 保存结果
        output_path = Path(args.output)
        mesh.export(args.output)
        
        # 保存纹理图像
        texture_path = output_path.parent / f"{output_path.stem}_texture.png"
        texture_image.save(texture_path)
        
        print(f"\n✅ 角色纹理映射完成!")
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
