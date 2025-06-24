#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于图像语义理解的纹理映射系统
解决眼睛贴到肚子、牙齿贴到腹部等错位问题
"""

import argparse
import numpy as np
import trimesh
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import cv2
from pathlib import Path
import sys

def analyze_image_semantics(image_path):
    """分析图像的语义内容，识别面部特征和身体部位"""
    print("🔍 分析图像语义内容")
    
    # 加载图像
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = image_path
    
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # 使用简单的颜色和位置分析来识别语义区域
    semantic_regions = {}
    
    # 1. 面部区域检测（基于位置和颜色特征）
    face_region = detect_face_region(img_array)
    semantic_regions['face'] = face_region
    
    # 2. 眼睛区域检测
    eye_regions = detect_eye_regions(img_array, face_region)
    semantic_regions['eyes'] = eye_regions
    
    # 3. 嘴巴区域检测
    mouth_region = detect_mouth_region(img_array, face_region)
    semantic_regions['mouth'] = mouth_region
    
    # 4. 头发区域检测
    hair_region = detect_hair_region(img_array, face_region)
    semantic_regions['hair'] = hair_region
    
    # 5. 身体区域检测
    body_region = detect_body_region(img_array, face_region)
    semantic_regions['body'] = body_region
    
    # 6. 手部区域检测
    hand_regions = detect_hand_regions(img_array)
    semantic_regions['hands'] = hand_regions
    
    print("语义区域检测结果:")
    for region_name, region_data in semantic_regions.items():
        if region_data is not None:
            print(f"  ✅ {region_name}: 已检测")
        else:
            print(f"  ❌ {region_name}: 未检测到")
    
    return semantic_regions

def detect_face_region(img_array):
    """检测面部区域"""
    h, w = img_array.shape[:2]
    
    # 简单的面部检测：假设面部在图像的中上部分
    # 基于肤色检测
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # 肤色范围（HSV）
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 形态学操作清理噪声
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # 找到最大的肤色区域
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_face, h_face = cv2.boundingRect(largest_contour)
        
        # 扩展面部区域
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w_face = min(w - x, w_face + 2 * margin)
        h_face = min(h - y, h_face + 2 * margin)
        
        return {
            'bbox': (x, y, w_face, h_face),
            'center': (x + w_face // 2, y + h_face // 2),
            'mask': skin_mask
        }
    
    # 如果肤色检测失败，使用默认位置
    default_x, default_y = w // 4, h // 6
    default_w, default_h = w // 2, h // 3
    
    return {
        'bbox': (default_x, default_y, default_w, default_h),
        'center': (default_x + default_w // 2, default_y + default_h // 2),
        'mask': None
    }

def detect_eye_regions(img_array, face_region):
    """检测眼睛区域"""
    if face_region is None:
        return None
    
    x, y, w, h = face_region['bbox']
    face_img = img_array[y:y+h, x:x+w]
    
    # 转换为灰度图
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    
    # 使用Haar级联检测眼睛（如果可用）
    # 这里使用简单的位置估计
    eye_y = h // 4  # 眼睛通常在面部上1/4处
    eye_h = h // 6  # 眼睛高度约为面部高度的1/6
    
    # 左眼
    left_eye_x = w // 6
    left_eye_w = w // 4
    
    # 右眼
    right_eye_x = w * 2 // 3
    right_eye_w = w // 4
    
    return {
        'left_eye': {
            'bbox': (x + left_eye_x, y + eye_y, left_eye_w, eye_h),
            'center': (x + left_eye_x + left_eye_w // 2, y + eye_y + eye_h // 2)
        },
        'right_eye': {
            'bbox': (x + right_eye_x, y + eye_y, right_eye_w, eye_h),
            'center': (x + right_eye_x + right_eye_w // 2, y + eye_y + eye_h // 2)
        }
    }

def detect_mouth_region(img_array, face_region):
    """检测嘴巴区域"""
    if face_region is None:
        return None
    
    x, y, w, h = face_region['bbox']
    
    # 嘴巴通常在面部下1/3处
    mouth_y = y + h * 2 // 3
    mouth_h = h // 6
    mouth_x = x + w // 4
    mouth_w = w // 2
    
    return {
        'bbox': (mouth_x, mouth_y, mouth_w, mouth_h),
        'center': (mouth_x + mouth_w // 2, mouth_y + mouth_h // 2)
    }

def detect_hair_region(img_array, face_region):
    """检测头发区域"""
    if face_region is None:
        return None
    
    h, w = img_array.shape[:2]
    face_x, face_y, face_w, face_h = face_region['bbox']
    
    # 头发区域：面部上方和两侧
    hair_x = max(0, face_x - face_w // 4)
    hair_y = 0
    hair_w = min(w - hair_x, face_w + face_w // 2)
    hair_h = face_y + face_h // 3
    
    return {
        'bbox': (hair_x, hair_y, hair_w, hair_h),
        'center': (hair_x + hair_w // 2, hair_y + hair_h // 2)
    }

def detect_body_region(img_array, face_region):
    """检测身体区域"""
    h, w = img_array.shape[:2]
    
    if face_region is None:
        # 默认身体区域
        body_x, body_y = 0, h // 3
        body_w, body_h = w, h * 2 // 3
    else:
        face_x, face_y, face_w, face_h = face_region['bbox']
        # 身体区域：面部下方
        body_x = 0
        body_y = face_y + face_h
        body_w = w
        body_h = h - body_y
    
    return {
        'bbox': (body_x, body_y, body_w, body_h),
        'center': (body_x + body_w // 2, body_y + body_h // 2)
    }

def detect_hand_regions(img_array):
    """检测手部区域"""
    h, w = img_array.shape[:2]
    
    # 简单估计：手部通常在图像的左下和右下角
    hand_size = min(w, h) // 6
    
    left_hand = {
        'bbox': (0, h - hand_size, hand_size, hand_size),
        'center': (hand_size // 2, h - hand_size // 2)
    }
    
    right_hand = {
        'bbox': (w - hand_size, h - hand_size, hand_size, hand_size),
        'center': (w - hand_size // 2, h - hand_size // 2)
    }
    
    return {
        'left_hand': left_hand,
        'right_hand': right_hand
    }

def create_semantic_texture_mapping(mesh_vertices, mesh_faces, semantic_regions, original_image, texture_size=1024):
    """基于语义理解创建纹理映射"""
    print("🧠 基于语义理解创建纹理映射")
    
    # 分析3D模型结构
    orientation_info = detect_character_orientation(mesh_vertices)
    parts = segment_character_parts(mesh_vertices, orientation_info)
    
    # 创建纹理画布
    texture = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 128
    
    # 加载原始图像
    if isinstance(original_image, str):
        img = Image.open(original_image)
    else:
        img = original_image
    
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    
    # 语义映射规则：将图像的语义区域映射到3D模型的对应部位
    semantic_mapping = {
        # 面部特征精确映射
        'face': {
            'source_region': semantic_regions.get('face'),
            'target_uv': (0.0, 0.5, 0.5, 1.0),  # 左上角
            'mesh_parts': ['face', 'head']
        },
        'eyes': {
            'source_region': semantic_regions.get('eyes'),
            'target_uv': (0.1, 0.4, 0.7, 0.9),  # 面部区域内的眼睛位置
            'mesh_parts': ['face']
        },
        'mouth': {
            'source_region': semantic_regions.get('mouth'),
            'target_uv': (0.15, 0.35, 0.5, 0.65),  # 面部区域内的嘴巴位置
            'mesh_parts': ['face']
        },
        'hair': {
            'source_region': semantic_regions.get('hair'),
            'target_uv': (0.0, 0.5, 0.8, 1.0),  # 面部上方
            'mesh_parts': ['head', 'back_head']
        },
        'body': {
            'source_region': semantic_regions.get('body'),
            'target_uv': (0.0, 0.5, 0.0, 0.5),  # 左下角
            'mesh_parts': ['chest', 'torso']
        },
        'hands': {
            'source_region': semantic_regions.get('hands'),
            'target_uv': (0.75, 1.0, 0.5, 1.0),  # 右上角
            'mesh_parts': ['left_arm', 'right_arm']
        }
    }
    
    # 应用语义映射
    for semantic_name, mapping_info in semantic_mapping.items():
        source_region = mapping_info['source_region']
        target_uv = mapping_info['target_uv']
        
        if source_region is None:
            continue
        
        # 提取源图像区域
        if semantic_name == 'face' and 'bbox' in source_region:
            x, y, w, h = source_region['bbox']
            src_region = img_array[y:y+h, x:x+w]
        elif semantic_name == 'eyes' and source_region:
            # 合并左右眼区域
            src_region = extract_eye_region(img_array, source_region)
        elif semantic_name == 'mouth' and 'bbox' in source_region:
            x, y, w, h = source_region['bbox']
            src_region = img_array[y:y+h, x:x+w]
        elif semantic_name == 'hair' and 'bbox' in source_region:
            x, y, w, h = source_region['bbox']
            src_region = img_array[y:y+h, x:x+w]
        elif semantic_name == 'body' and 'bbox' in source_region:
            x, y, w, h = source_region['bbox']
            src_region = img_array[y:y+h, x:x+w]
        elif semantic_name == 'hands' and source_region:
            # 合并左右手区域
            src_region = extract_hand_region(img_array, source_region)
        else:
            continue
        
        if src_region.size == 0:
            continue
        
        # 计算目标纹理区域
        u_min, u_max, v_min, v_max = target_uv
        dst_x1, dst_x2 = int(u_min * texture_size), int(u_max * texture_size)
        dst_y1, dst_y2 = int(v_min * texture_size), int(v_max * texture_size)
        
        # 调整大小并应用
        target_w, target_h = dst_x2 - dst_x1, dst_y2 - dst_y1
        if target_w > 0 and target_h > 0:
            resized = cv2.resize(src_region, (target_w, target_h), 
                               interpolation=cv2.INTER_LANCZOS4)
            
            # 特殊处理
            if semantic_name == 'face':
                resized = enhance_face_texture(resized)
            elif semantic_name == 'eyes':
                resized = enhance_eye_texture(resized)
            
            texture[dst_y1:dst_y2, dst_x1:dst_x2] = resized
    
    return Image.fromarray(texture)

def extract_eye_region(img_array, eye_regions):
    """提取眼睛区域"""
    if 'left_eye' in eye_regions and 'right_eye' in eye_regions:
        left_bbox = eye_regions['left_eye']['bbox']
        right_bbox = eye_regions['right_eye']['bbox']
        
        # 计算包含两只眼睛的区域
        min_x = min(left_bbox[0], right_bbox[0])
        min_y = min(left_bbox[1], right_bbox[1])
        max_x = max(left_bbox[0] + left_bbox[2], right_bbox[0] + right_bbox[2])
        max_y = max(left_bbox[1] + left_bbox[3], right_bbox[1] + right_bbox[3])
        
        return img_array[min_y:max_y, min_x:max_x]
    
    return np.array([])

def extract_hand_region(img_array, hand_regions):
    """提取手部区域"""
    if 'left_hand' in hand_regions:
        x, y, w, h = hand_regions['left_hand']['bbox']
        return img_array[y:y+h, x:x+w]
    
    return np.array([])

def enhance_face_texture(texture):
    """增强面部纹理"""
    # 增强对比度和细节
    enhanced = cv2.convertScaleAbs(texture, alpha=1.2, beta=10)
    return enhanced

def enhance_eye_texture(texture):
    """增强眼部纹理"""
    # 增强眼部细节
    enhanced = cv2.convertScaleAbs(texture, alpha=1.3, beta=5)
    return enhanced

# 导入之前定义的函数
def detect_character_orientation(vertices):
    """检测人物/动物的朝向和姿态"""
    center = vertices.mean(axis=0)
    centered = vertices - center
    
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    
    primary_axis = eigenvectors[:, idx[0]]
    secondary_axis = eigenvectors[:, idx[1]]
    tertiary_axis = eigenvectors[:, idx[2]]
    
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
    
    if height_axis[1] < 0:
        height_axis = -height_axis
    
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
    center = orientation_info['center']
    centered = vertices - center
    
    height_proj = np.dot(centered, orientation_info['height_axis'])
    front_proj = np.dot(centered, orientation_info['front_axis'])
    side_proj = np.dot(centered, orientation_info['side_axis'])
    
    height_norm = (height_proj - height_proj.min()) / (height_proj.max() - height_proj.min() + 1e-8)
    front_norm = (front_proj - front_proj.min()) / (front_proj.max() - front_proj.min() + 1e-8)
    side_norm = (side_proj - side_proj.min()) / (side_proj.max() - side_proj.min() + 1e-8)
    
    distances = np.linalg.norm(centered, axis=1)
    dist_norm = distances / distances.max()
    
    parts = {}
    
    # 基本部位分割
    parts['head'] = height_norm > 0.8
    parts['face'] = parts['head'] & (front_norm > 0.6)
    parts['back_head'] = parts['head'] & (front_norm < 0.4)
    
    torso_mask = (height_norm >= 0.3) & (height_norm <= 0.8) & (dist_norm < 0.7)
    parts['torso'] = torso_mask
    parts['chest'] = torso_mask & (front_norm > 0.6)
    parts['back'] = torso_mask & (front_norm < 0.4)
    
    parts['left_arm'] = (side_norm < 0.2) & (height_norm > 0.4) & (dist_norm > 0.5)
    parts['right_arm'] = (side_norm > 0.8) & (height_norm > 0.4) & (dist_norm > 0.5)
    
    parts['legs'] = height_norm < 0.3
    
    return parts

def create_semantic_uv_mapping(vertices, parts, semantic_regions, texture_size=1024):
    """基于语义理解创建UV映射"""
    print("🗺️ 创建语义感知UV映射")

    uv_coords = np.zeros((len(vertices), 2))

    # 精确的UV布局，确保面部特征正确对应
    uv_layout = {
        'face': (0.0, 0.5, 0.5, 1.0),      # 左上角 - 面部主区域
        'head': (0.0, 0.5, 0.5, 1.0),      # 与面部共享区域
        'back_head': (0.5, 0.75, 0.75, 1.0), # 右上小块
        'chest': (0.0, 0.5, 0.0, 0.5),     # 左下角 - 胸部
        'torso': (0.0, 0.5, 0.0, 0.5),     # 与胸部共享
        'back': (0.5, 1.0, 0.0, 0.5),      # 右下角 - 背部
        'left_arm': (0.75, 1.0, 0.5, 0.75), # 右上小块
        'right_arm': (0.75, 1.0, 0.25, 0.5), # 右中小块
        'legs': (0.5, 0.75, 0.5, 0.75),    # 右上中块
    }

    # 为每个身体部位分配UV坐标
    for part_name, mask in parts.items():
        if not np.any(mask) or part_name not in uv_layout:
            continue

        # 获取UV区域
        u_min, u_max, v_min, v_max = uv_layout[part_name]

        # 简单的平面投影到UV空间
        part_vertices = vertices[mask]
        if len(part_vertices) > 0:
            # 计算部位的边界框
            min_coords = part_vertices.min(axis=0)
            max_coords = part_vertices.max(axis=0)
            ranges = max_coords - min_coords

            # 选择最大的两个维度进行投影
            range_indices = np.argsort(ranges)[-2:]

            u_coord = part_vertices[:, range_indices[0]]
            v_coord = part_vertices[:, range_indices[1]]

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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于图像语义理解的纹理映射")
    parser.add_argument("--mesh", type=str, required=True, help="输入3D网格文件")
    parser.add_argument("--image", type=str, required=True, help="输入图像文件")
    parser.add_argument("--output", type=str, default="semantic_textured.glb", help="输出文件")
    parser.add_argument("--texture-size", type=int, default=1024, help="纹理分辨率")
    parser.add_argument("--debug", action="store_true", help="保存调试信息")

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

        # 分析图像语义
        semantic_regions = analyze_image_semantics(args.image)

        # 检测角色朝向
        orientation_info = detect_character_orientation(mesh.vertices)

        # 分割身体部位
        parts = segment_character_parts(mesh.vertices, orientation_info)

        # 创建语义感知的纹理
        texture_image = create_semantic_texture_mapping(
            mesh.vertices, mesh.faces, semantic_regions, args.image, args.texture_size
        )

        # 创建UV映射
        uv_coords = create_semantic_uv_mapping(mesh.vertices, parts, semantic_regions, args.texture_size)

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

        # 如果启用调试模式，保存语义分析结果
        if args.debug:
            save_debug_info(args.image, semantic_regions, output_path.parent)

        print(f"\n✅ 语义纹理映射完成!")
        print(f"📁 输出文件: {args.output}")
        print(f"🖼️ 纹理文件: {texture_path}")

        return True

    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_debug_info(image_path, semantic_regions, output_dir):
    """保存调试信息"""
    print("💾 保存调试信息")

    # 加载原始图像
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    img_array = np.array(img)
    debug_img = img_array.copy()

    # 在图像上标注检测到的区域
    colors = {
        'face': (255, 0, 0),      # 红色
        'eyes': (0, 255, 0),      # 绿色
        'mouth': (0, 0, 255),     # 蓝色
        'hair': (255, 255, 0),    # 黄色
        'body': (255, 0, 255),    # 紫色
        'hands': (0, 255, 255),   # 青色
    }

    for region_name, region_data in semantic_regions.items():
        if region_data is None:
            continue

        color = colors.get(region_name, (128, 128, 128))

        if region_name in ['face', 'mouth', 'hair', 'body'] and 'bbox' in region_data:
            x, y, w, h = region_data['bbox']
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(debug_img, region_name, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        elif region_name == 'eyes' and isinstance(region_data, dict):
            for eye_name, eye_data in region_data.items():
                if 'bbox' in eye_data:
                    x, y, w, h = eye_data['bbox']
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(debug_img, eye_name, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        elif region_name == 'hands' and isinstance(region_data, dict):
            for hand_name, hand_data in region_data.items():
                if 'bbox' in hand_data:
                    x, y, w, h = hand_data['bbox']
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(debug_img, hand_name, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 保存调试图像
    debug_path = output_dir / "semantic_analysis_debug.png"
    Image.fromarray(debug_img).save(debug_path)
    print(f"🔍 语义分析调试图像已保存: {debug_path}")

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
