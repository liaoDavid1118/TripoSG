#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级语义纹理映射系统
使用更精确的图像分析和语义理解
"""

import argparse
import numpy as np
import trimesh
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import cv2
from pathlib import Path
import sys

def advanced_face_detection(img_array):
    """使用更高级的面部检测方法"""
    print("🎯 执行高级面部检测")
    
    h, w = img_array.shape[:2]
    
    # 方法1: 基于肤色的改进检测
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # 多种肤色范围
    skin_ranges = [
        # 浅肤色
        ([0, 20, 70], [20, 255, 255]),
        # 中等肤色
        ([0, 25, 80], [25, 255, 255]),
        # 深肤色
        ([0, 30, 60], [30, 255, 200])
    ]
    
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    for lower, upper in skin_ranges:
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # 使用LAB色彩空间进一步优化
    # 肤色在LAB空间的特征
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    
    # 肤色通常有特定的a和b值范围
    lab_mask = ((a_channel > 120) & (a_channel < 150) & 
                (b_channel > 120) & (b_channel < 150) &
                (l_channel > 50))
    
    combined_mask = cv2.bitwise_or(combined_mask, lab_mask.astype(np.uint8) * 255)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # 找到最大的连通区域
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 选择最大且最接近面部比例的轮廓
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w_face, h_face = cv2.boundingRect(contour)
            
            # 面部比例检查
            aspect_ratio = w_face / h_face if h_face > 0 else 0
            
            # 理想的面部宽高比约为0.7-0.9
            ratio_score = 1.0 - abs(aspect_ratio - 0.8)
            
            # 位置得分：面部通常在图像上半部分
            position_score = 1.0 - (y / h)
            
            # 大小得分：面部应该占据合理的图像比例
            size_ratio = area / (w * h)
            size_score = min(size_ratio * 10, 1.0) if size_ratio < 0.1 else max(1.0 - (size_ratio - 0.1) * 5, 0)
            
            # 综合得分
            total_score = ratio_score * 0.4 + position_score * 0.3 + size_score * 0.3
            
            if total_score > best_score and area > 1000:  # 最小面积阈值
                best_score = total_score
                best_contour = contour
        
        if best_contour is not None:
            x, y, w_face, h_face = cv2.boundingRect(best_contour)
            
            # 扩展面部区域
            margin_x = int(w_face * 0.1)
            margin_y = int(h_face * 0.1)
            
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w_face = min(w - x, w_face + 2 * margin_x)
            h_face = min(h - y, h_face + 2 * margin_y)
            
            return {
                'bbox': (x, y, w_face, h_face),
                'center': (x + w_face // 2, y + h_face // 2),
                'confidence': best_score,
                'mask': combined_mask
            }
    
    # 如果检测失败，使用智能默认位置
    return get_smart_default_face_region(img_array)

def get_smart_default_face_region(img_array):
    """智能默认面部区域"""
    h, w = img_array.shape[:2]
    
    # 分析图像的亮度分布来猜测面部位置
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 计算图像上半部分的亮度分布
    upper_half = gray[:h//2, :]
    
    # 找到亮度较高的区域（通常是面部）
    bright_threshold = np.percentile(upper_half, 70)
    bright_mask = upper_half > bright_threshold
    
    # 找到亮区域的中心
    y_coords, x_coords = np.where(bright_mask)
    
    if len(x_coords) > 0:
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        # 基于中心点创建面部区域
        face_w = min(w // 3, h // 3)
        face_h = int(face_w * 1.2)  # 面部通常比较高
        
        x = max(0, center_x - face_w // 2)
        y = max(0, center_y - face_h // 3)  # 面部中心偏上
        
        face_w = min(w - x, face_w)
        face_h = min(h - y, face_h)
        
    else:
        # 最后的默认位置
        x, y = w // 4, h // 8
        face_w, face_h = w // 2, h // 2
    
    return {
        'bbox': (x, y, face_w, face_h),
        'center': (x + face_w // 2, y + face_h // 2),
        'confidence': 0.3,
        'mask': None
    }

def precise_facial_feature_detection(img_array, face_region):
    """精确的面部特征检测"""
    print("👁️ 执行精确面部特征检测")
    
    if face_region is None:
        return {}
    
    x, y, w, h = face_region['bbox']
    face_img = img_array[y:y+h, x:x+w]
    
    features = {}
    
    # 眼睛检测
    eyes = detect_eyes_precise(face_img, (x, y))
    if eyes:
        features['eyes'] = eyes
    
    # 嘴巴检测
    mouth = detect_mouth_precise(face_img, (x, y))
    if mouth:
        features['mouth'] = mouth
    
    # 鼻子检测
    nose = detect_nose_precise(face_img, (x, y))
    if nose:
        features['nose'] = nose
    
    # 眉毛检测
    eyebrows = detect_eyebrows_precise(face_img, (x, y))
    if eyebrows:
        features['eyebrows'] = eyebrows
    
    return features

def detect_eyes_precise(face_img, offset):
    """精确的眼睛检测"""
    h, w = face_img.shape[:2]
    offset_x, offset_y = offset
    
    # 眼睛通常在面部上1/3处
    eye_region_y = h // 6
    eye_region_h = h // 3
    eye_region = face_img[eye_region_y:eye_region_y + eye_region_h, :]
    
    # 转换为灰度图
    gray = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
    
    # 使用边缘检测找到眼睛轮廓
    edges = cv2.Canny(gray, 50, 150)
    
    # 形态学操作连接边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    eye_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50 or area > 2000:  # 过滤不合理的大小
            continue
        
        x_eye, y_eye, w_eye, h_eye = cv2.boundingRect(contour)
        
        # 眼睛的宽高比检查
        aspect_ratio = w_eye / h_eye if h_eye > 0 else 0
        if aspect_ratio < 1.2 or aspect_ratio > 3.0:  # 眼睛通常是椭圆形
            continue
        
        # 计算在原图中的坐标
        abs_x = offset_x + x_eye
        abs_y = offset_y + eye_region_y + y_eye
        
        eye_candidates.append({
            'bbox': (abs_x, abs_y, w_eye, h_eye),
            'center': (abs_x + w_eye // 2, abs_y + h_eye // 2),
            'area': area
        })
    
    # 选择最可能的两只眼睛
    if len(eye_candidates) >= 2:
        # 按面积排序，选择较大的两个
        eye_candidates.sort(key=lambda x: x['area'], reverse=True)
        
        # 确保两只眼睛在合理的水平位置
        eye1, eye2 = eye_candidates[0], eye_candidates[1]
        
        if eye1['center'][0] < eye2['center'][0]:
            left_eye, right_eye = eye1, eye2
        else:
            left_eye, right_eye = eye2, eye1
        
        return {
            'left_eye': left_eye,
            'right_eye': right_eye
        }
    
    # 如果检测失败，使用默认位置
    return get_default_eye_positions(face_img, offset)

def get_default_eye_positions(face_img, offset):
    """默认眼睛位置"""
    h, w = face_img.shape[:2]
    offset_x, offset_y = offset
    
    # 标准面部比例
    eye_y = offset_y + h // 3
    eye_h = h // 8
    eye_w = w // 6
    
    left_eye_x = offset_x + w // 4
    right_eye_x = offset_x + w * 3 // 4 - eye_w
    
    return {
        'left_eye': {
            'bbox': (left_eye_x, eye_y, eye_w, eye_h),
            'center': (left_eye_x + eye_w // 2, eye_y + eye_h // 2)
        },
        'right_eye': {
            'bbox': (right_eye_x, eye_y, eye_w, eye_h),
            'center': (right_eye_x + eye_w // 2, eye_y + eye_h // 2)
        }
    }

def detect_mouth_precise(face_img, offset):
    """精确的嘴巴检测"""
    h, w = face_img.shape[:2]
    offset_x, offset_y = offset
    
    # 嘴巴通常在面部下1/3处
    mouth_region_y = h * 2 // 3
    mouth_region_h = h // 4
    mouth_region = face_img[mouth_region_y:mouth_region_y + mouth_region_h, :]
    
    # 转换为HSV，检测红色区域（嘴唇）
    hsv = cv2.cvtColor(mouth_region, cv2.COLOR_RGB2HSV)
    
    # 红色范围（嘴唇颜色）
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # 找到轮廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        x_mouth, y_mouth, w_mouth, h_mouth = cv2.boundingRect(largest_contour)
        
        # 计算在原图中的坐标
        abs_x = offset_x + x_mouth
        abs_y = offset_y + mouth_region_y + y_mouth
        
        return {
            'bbox': (abs_x, abs_y, w_mouth, h_mouth),
            'center': (abs_x + w_mouth // 2, abs_y + h_mouth // 2)
        }
    
    # 默认嘴巴位置
    mouth_x = offset_x + w // 3
    mouth_y = offset_y + h * 3 // 4
    mouth_w = w // 3
    mouth_h = h // 12
    
    return {
        'bbox': (mouth_x, mouth_y, mouth_w, mouth_h),
        'center': (mouth_x + mouth_w // 2, mouth_y + mouth_h // 2)
    }

def detect_nose_precise(face_img, offset):
    """精确的鼻子检测"""
    h, w = face_img.shape[:2]
    offset_x, offset_y = offset
    
    # 鼻子通常在面部中央
    nose_x = offset_x + w // 3
    nose_y = offset_y + h // 2
    nose_w = w // 3
    nose_h = h // 4
    
    return {
        'bbox': (nose_x, nose_y, nose_w, nose_h),
        'center': (nose_x + nose_w // 2, nose_y + nose_h // 2)
    }

def detect_eyebrows_precise(face_img, offset):
    """精确的眉毛检测"""
    h, w = face_img.shape[:2]
    offset_x, offset_y = offset
    
    # 眉毛通常在眼睛上方
    eyebrow_y = offset_y + h // 6
    eyebrow_h = h // 12
    
    left_eyebrow_x = offset_x + w // 6
    left_eyebrow_w = w // 4
    
    right_eyebrow_x = offset_x + w * 7 // 12
    right_eyebrow_w = w // 4
    
    return {
        'left_eyebrow': {
            'bbox': (left_eyebrow_x, eyebrow_y, left_eyebrow_w, eyebrow_h),
            'center': (left_eyebrow_x + left_eyebrow_w // 2, eyebrow_y + eyebrow_h // 2)
        },
        'right_eyebrow': {
            'bbox': (right_eyebrow_x, eyebrow_y, right_eyebrow_w, eyebrow_h),
            'center': (right_eyebrow_x + right_eyebrow_w // 2, eyebrow_y + eyebrow_h // 2)
        }
    }

def create_precise_texture_layout(original_image, face_region, facial_features, texture_size=1024):
    """创建精确的纹理布局"""
    print("🎨 创建精确纹理布局")
    
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
    h, w = img_array.shape[:2]
    
    # 创建纹理画布
    texture = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 128
    
    # 精确的特征映射
    if face_region and 'bbox' in face_region:
        face_x, face_y, face_w, face_h = face_region['bbox']
        
        # 面部主区域 - 左上角大块
        face_texture_region = (0, texture_size // 2, texture_size // 2, texture_size)
        apply_region_to_texture(img_array, (face_x, face_y, face_w, face_h), 
                              texture, face_texture_region, enhance_face=True)
        
        # 精确放置面部特征
        if 'eyes' in facial_features:
            eyes = facial_features['eyes']
            if 'left_eye' in eyes and 'right_eye' in eyes:
                # 计算眼睛区域
                left_eye_bbox = eyes['left_eye']['bbox']
                right_eye_bbox = eyes['right_eye']['bbox']
                
                # 合并眼睛区域
                min_x = min(left_eye_bbox[0], right_eye_bbox[0])
                min_y = min(left_eye_bbox[1], right_eye_bbox[1])
                max_x = max(left_eye_bbox[0] + left_eye_bbox[2], 
                           right_eye_bbox[0] + right_eye_bbox[2])
                max_y = max(left_eye_bbox[1] + left_eye_bbox[3], 
                           right_eye_bbox[1] + right_eye_bbox[3])
                
                eyes_region = (min_x, min_y, max_x - min_x, max_y - min_y)
                
                # 眼睛放在面部区域的正确位置
                eye_texture_region = (texture_size // 8, texture_size * 3 // 8, 
                                    texture_size * 5 // 8, texture_size * 3 // 4)
                apply_region_to_texture(img_array, eyes_region, 
                                      texture, eye_texture_region, enhance_eyes=True)
        
        if 'mouth' in facial_features:
            mouth = facial_features['mouth']
            mouth_bbox = mouth['bbox']
            
            # 嘴巴放在面部区域的下方
            mouth_texture_region = (texture_size // 6, texture_size // 3, 
                                  texture_size // 2, texture_size * 5 // 8)
            apply_region_to_texture(img_array, mouth_bbox, 
                                  texture, mouth_texture_region, enhance_mouth=True)
    
    # 身体区域 - 右下角
    body_y_start = max(0, face_region['bbox'][1] + face_region['bbox'][3]) if face_region else h // 3
    body_region = (0, body_y_start, w, h - body_y_start)
    body_texture_region = (texture_size // 2, texture_size, 0, texture_size // 2)
    apply_region_to_texture(img_array, body_region, texture, body_texture_region)
    
    return Image.fromarray(texture)

def apply_region_to_texture(source_img, source_region, texture, texture_region, 
                          enhance_face=False, enhance_eyes=False, enhance_mouth=False):
    """将源图像区域应用到纹理的指定区域"""
    src_x, src_y, src_w, src_h = source_region
    tex_x1, tex_x2, tex_y1, tex_y2 = texture_region
    
    # 确保坐标在有效范围内
    src_x = max(0, min(src_x, source_img.shape[1] - 1))
    src_y = max(0, min(src_y, source_img.shape[0] - 1))
    src_x2 = max(src_x + 1, min(src_x + src_w, source_img.shape[1]))
    src_y2 = max(src_y + 1, min(src_y + src_h, source_img.shape[0]))
    
    tex_x1 = max(0, min(tex_x1, texture.shape[1] - 1))
    tex_y1 = max(0, min(tex_y1, texture.shape[0] - 1))
    tex_x2 = max(tex_x1 + 1, min(tex_x2, texture.shape[1]))
    tex_y2 = max(tex_y1 + 1, min(tex_y2, texture.shape[0]))
    
    # 提取源区域
    src_region = source_img[src_y:src_y2, src_x:src_x2]
    
    if src_region.size == 0:
        return
    
    # 调整大小
    target_w = tex_x2 - tex_x1
    target_h = tex_y2 - tex_y1
    
    if target_w > 0 and target_h > 0:
        resized = cv2.resize(src_region, (target_w, target_h), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        # 应用增强
        if enhance_face:
            resized = cv2.convertScaleAbs(resized, alpha=1.15, beta=10)
        elif enhance_eyes:
            resized = cv2.convertScaleAbs(resized, alpha=1.25, beta=15)
        elif enhance_mouth:
            resized = cv2.convertScaleAbs(resized, alpha=1.1, beta=8)
        
        # 应用到纹理
        texture[tex_y1:tex_y2, tex_x1:tex_x2] = resized

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高级语义纹理映射系统")
    parser.add_argument("--mesh", type=str, required=True, help="输入3D网格文件")
    parser.add_argument("--image", type=str, required=True, help="输入图像文件")
    parser.add_argument("--output", type=str, default="advanced_semantic_textured.glb", help="输出文件")
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

        # 加载和分析图像
        img = Image.open(args.image)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img_array = np.array(img)

        # 高级面部检测
        face_region = advanced_face_detection(img_array)
        print(f"面部检测置信度: {face_region.get('confidence', 0):.2f}")

        # 精确面部特征检测
        facial_features = precise_facial_feature_detection(img_array, face_region)

        # 创建精确纹理布局
        texture_image = create_precise_texture_layout(
            args.image, face_region, facial_features, args.texture_size
        )

        # 创建简单的UV映射（可以进一步优化）
        uv_coords = create_simple_uv_mapping(mesh.vertices)

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

        # 如果启用调试模式，保存分析结果
        if args.debug:
            # 分析3D模型结构
            mesh_analysis = analyze_3d_mesh_structure(mesh.vertices, mesh.faces)
            save_advanced_debug_info(img_array, face_region, facial_features, output_path.parent)
            save_3d_mesh_analysis(mesh, mesh_analysis, output_path.parent)

        print(f"\n✅ 高级语义纹理映射完成!")
        print(f"📁 输出文件: {args.output}")
        print(f"🖼️ 纹理文件: {texture_path}")

        return True

    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_uv_mapping(vertices):
    """创建简单的UV映射"""
    # 使用球面投影作为基础
    center = vertices.mean(axis=0)
    centered = vertices - center

    # 标准化
    max_dist = np.linalg.norm(centered, axis=1).max()
    normalized = centered / (max_dist + 1e-8)

    x, y, z = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    # 球面坐标
    theta = np.arctan2(y, x)
    phi = np.arccos(np.clip(z, -1, 1))

    # 转换为UV坐标
    u = (theta + np.pi) / (2 * np.pi)
    v = phi / np.pi

    return np.column_stack([u, v])

def save_advanced_debug_info(img_array, face_region, facial_features, output_dir):
    """保存高级调试信息"""
    print("💾 保存高级调试信息")

    debug_img = img_array.copy()
    h, w = img_array.shape[:2]

    # 创建详细的分析报告
    analysis_report = []
    analysis_report.append("=== 图像语义分析报告 ===")
    analysis_report.append(f"图像尺寸: {w} x {h}")

    # 标注面部区域
    if face_region and 'bbox' in face_region:
        x, y, face_w, face_h = face_region['bbox']
        cv2.rectangle(debug_img, (x, y), (x + face_w, y + face_h), (255, 0, 0), 3)
        confidence = face_region.get('confidence', 0)
        cv2.putText(debug_img, f"Face ({confidence:.2f})", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        analysis_report.append(f"\n面部检测:")
        analysis_report.append(f"  位置: ({x}, {y})")
        analysis_report.append(f"  尺寸: {face_w} x {face_h}")
        analysis_report.append(f"  置信度: {confidence:.3f}")
        analysis_report.append(f"  占图像比例: {(face_w * face_h) / (w * h) * 100:.1f}%")
        analysis_report.append(f"  宽高比: {face_w / face_h:.2f}")
        analysis_report.append(f"  中心位置: ({x + face_w//2}, {y + face_h//2})")
    else:
        analysis_report.append("\n面部检测: 失败")

    # 标注面部特征
    colors = {
        'eyes': (0, 255, 0),
        'mouth': (0, 0, 255),
        'nose': (255, 255, 0),
        'eyebrows': (255, 0, 255)
    }

    analysis_report.append(f"\n面部特征检测:")

    for feature_name, feature_data in facial_features.items():
        color = colors.get(feature_name, (128, 128, 128))
        analysis_report.append(f"\n  {feature_name}:")

        if feature_name == 'eyes' and isinstance(feature_data, dict):
            for eye_name, eye_data in feature_data.items():
                if 'bbox' in eye_data:
                    x, y, w_eye, h_eye = eye_data['bbox']
                    cv2.rectangle(debug_img, (x, y), (x + w_eye, y + h_eye), color, 2)
                    cv2.putText(debug_img, eye_name, (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    analysis_report.append(f"    {eye_name}: ({x}, {y}) {w_eye}x{h_eye}")
                    analysis_report.append(f"      中心: ({x + w_eye//2}, {y + h_eye//2})")

        elif isinstance(feature_data, dict) and 'bbox' in feature_data:
            x, y, w_feat, h_feat = feature_data['bbox']
            cv2.rectangle(debug_img, (x, y), (x + w_feat, y + h_feat), color, 2)
            cv2.putText(debug_img, feature_name, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            analysis_report.append(f"    位置: ({x}, {y})")
            analysis_report.append(f"    尺寸: {w_feat} x {h_feat}")
            analysis_report.append(f"    中心: ({x + w_feat//2}, {y + h_feat//2})")
        else:
            analysis_report.append(f"    检测失败")

    # 分析颜色分布
    analysis_report.append(f"\n颜色分析:")

    # 转换到不同色彩空间进行分析
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

    # 分析肤色分布
    skin_mask = detect_skin_regions(img_array)
    skin_percentage = np.sum(skin_mask > 0) / (w * h) * 100
    analysis_report.append(f"  肤色区域占比: {skin_percentage:.1f}%")

    # 分析亮度分布
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    brightness_mean = np.mean(gray)
    brightness_std = np.std(gray)
    analysis_report.append(f"  平均亮度: {brightness_mean:.1f}")
    analysis_report.append(f"  亮度标准差: {brightness_std:.1f}")

    # 分析对比度
    contrast = brightness_std / brightness_mean if brightness_mean > 0 else 0
    analysis_report.append(f"  对比度: {contrast:.3f}")

    # 保存调试图像
    debug_path = output_dir / "advanced_semantic_debug.png"
    Image.fromarray(debug_img).save(debug_path)
    print(f"🔍 高级语义分析调试图像已保存: {debug_path}")

    # 保存肤色检测结果
    skin_debug_img = img_array.copy()
    skin_debug_img[skin_mask == 0] = [128, 128, 128]  # 非肤色区域变灰
    skin_debug_path = output_dir / "skin_detection_debug.png"
    Image.fromarray(skin_debug_img).save(skin_debug_path)
    print(f"🎨 肤色检测调试图像已保存: {skin_debug_path}")

    # 保存分析报告
    report_path = output_dir / "semantic_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(analysis_report))
    print(f"📊 语义分析报告已保存: {report_path}")

def detect_skin_regions(img_array):
    """检测肤色区域"""
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # 多种肤色范围
    skin_ranges = [
        ([0, 20, 70], [20, 255, 255]),
        ([0, 25, 80], [25, 255, 255]),
        ([0, 30, 60], [30, 255, 200])
    ]

    combined_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)

    for lower, upper in skin_ranges:
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    return combined_mask

def analyze_3d_mesh_structure(vertices, faces):
    """分析3D网格结构"""
    print("🔍 分析3D网格结构")

    center = vertices.mean(axis=0)
    centered = vertices - center

    # PCA分析
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 计算各种几何特征
    distances = np.linalg.norm(centered, axis=1)

    # 投影到主轴
    proj_x = np.dot(centered, eigenvectors[:, 0])
    proj_y = np.dot(centered, eigenvectors[:, 1])
    proj_z = np.dot(centered, eigenvectors[:, 2])

    # 分析形状特征
    aspect_ratios = eigenvalues / eigenvalues[0]

    # 检测对称性
    symmetry_score = analyze_symmetry(centered, eigenvectors)

    # 分析顶点密度分布
    density_analysis = analyze_vertex_density(vertices, faces)

    # 检测突出部分（可能是四肢）
    protrusions = detect_protrusions(centered, distances)

    analysis = {
        'center': center,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'aspect_ratios': aspect_ratios,
        'symmetry_score': symmetry_score,
        'density_analysis': density_analysis,
        'protrusions': protrusions,
        'vertex_count': len(vertices),
        'face_count': len(faces),
        'bounding_box': {
            'min': vertices.min(axis=0),
            'max': vertices.max(axis=0),
            'size': vertices.max(axis=0) - vertices.min(axis=0)
        }
    }

    return analysis

def analyze_symmetry(centered, eigenvectors):
    """分析模型的对称性"""
    # 使用主轴作为对称轴进行分析
    main_axis = eigenvectors[:, 0]

    # 将顶点投影到垂直于主轴的平面
    proj_to_plane = centered - np.outer(np.dot(centered, main_axis), main_axis)

    # 计算对称性得分
    # 简单方法：比较左右两侧的点分布
    side_axis = eigenvectors[:, 1]
    side_proj = np.dot(proj_to_plane, side_axis)

    left_points = proj_to_plane[side_proj < 0]
    right_points = proj_to_plane[side_proj > 0]

    if len(left_points) > 0 and len(right_points) > 0:
        # 计算左右两侧的分布相似性
        left_std = np.std(left_points, axis=0)
        right_std = np.std(right_points, axis=0)

        symmetry = 1.0 - np.mean(np.abs(left_std - right_std) / (left_std + right_std + 1e-8))
        return max(0, min(1, symmetry))

    return 0.5

def analyze_vertex_density(vertices, faces):
    """分析顶点密度分布"""
    # 计算每个顶点的邻居数量
    vertex_neighbors = [[] for _ in range(len(vertices))]

    for face in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    vertex_neighbors[face[i]].append(face[j])

    neighbor_counts = [len(set(neighbors)) for neighbors in vertex_neighbors]

    return {
        'mean_neighbors': np.mean(neighbor_counts),
        'std_neighbors': np.std(neighbor_counts),
        'min_neighbors': np.min(neighbor_counts),
        'max_neighbors': np.max(neighbor_counts)
    }

def detect_protrusions(centered, distances):
    """检测突出部分（四肢等）"""
    # 使用距离阈值检测突出部分
    distance_threshold = np.percentile(distances, 85)  # 前15%的远点

    protrusion_mask = distances > distance_threshold
    protrusion_points = centered[protrusion_mask]

    if len(protrusion_points) == 0:
        return []

    # 使用聚类分析将突出点分组
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=0.1, min_samples=10).fit(protrusion_points)
    labels = clustering.labels_

    protrusions = []
    for label in set(labels):
        if label == -1:  # 噪声点
            continue

        cluster_points = protrusion_points[labels == label]
        cluster_center = cluster_points.mean(axis=0)
        cluster_size = len(cluster_points)

        protrusions.append({
            'center': cluster_center,
            'size': cluster_size,
            'points': cluster_points
        })

    return protrusions

def save_3d_mesh_analysis(mesh, analysis, output_dir):
    """保存3D网格分析结果"""
    print("💾 保存3D网格分析结果")

    report = []
    report.append("=== 3D网格结构分析报告 ===")
    report.append(f"顶点数量: {analysis['vertex_count']}")
    report.append(f"面数量: {analysis['face_count']}")

    # 边界框信息
    bbox = analysis['bounding_box']
    report.append(f"\n边界框:")
    report.append(f"  最小坐标: ({bbox['min'][0]:.3f}, {bbox['min'][1]:.3f}, {bbox['min'][2]:.3f})")
    report.append(f"  最大坐标: ({bbox['max'][0]:.3f}, {bbox['max'][1]:.3f}, {bbox['max'][2]:.3f})")
    report.append(f"  尺寸: ({bbox['size'][0]:.3f}, {bbox['size'][1]:.3f}, {bbox['size'][2]:.3f})")

    # 主轴分析
    report.append(f"\n主轴分析:")
    eigenvalues = analysis['eigenvalues']
    eigenvectors = analysis['eigenvectors']
    aspect_ratios = analysis['aspect_ratios']

    for i, (val, vec, ratio) in enumerate(zip(eigenvalues, eigenvectors.T, aspect_ratios)):
        report.append(f"  主轴{i+1}: 特征值={val:.3f}, 比例={ratio:.3f}")
        report.append(f"    方向: ({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})")

    # 形状分类
    if aspect_ratios[1] > 0.7 and aspect_ratios[2] > 0.7:
        shape_type = "球形"
    elif aspect_ratios[1] > 0.3 and aspect_ratios[2] < 0.3:
        shape_type = "柱形"
    elif aspect_ratios[2] < 0.2:
        shape_type = "扁平形"
    else:
        shape_type = "不规则形"

    report.append(f"\n形状分类: {shape_type}")
    report.append(f"对称性得分: {analysis['symmetry_score']:.3f}")

    # 顶点密度分析
    density = analysis['density_analysis']
    report.append(f"\n顶点密度分析:")
    report.append(f"  平均邻居数: {density['mean_neighbors']:.1f}")
    report.append(f"  邻居数标准差: {density['std_neighbors']:.1f}")
    report.append(f"  最少邻居数: {density['min_neighbors']}")
    report.append(f"  最多邻居数: {density['max_neighbors']}")

    # 突出部分分析
    protrusions = analysis['protrusions']
    report.append(f"\n突出部分检测:")
    report.append(f"  检测到 {len(protrusions)} 个突出部分")

    for i, prot in enumerate(protrusions):
        center = prot['center']
        size = prot['size']
        report.append(f"  突出部分{i+1}: 中心({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), 大小={size}")

    # 语义部位推测
    report.append(f"\n语义部位推测:")

    # 基于形状和突出部分推测身体部位
    if len(protrusions) >= 4:
        report.append("  可能是人形或动物模型（检测到多个突出部分）")
        report.append("  突出部分可能对应：四肢、头部等")
    elif len(protrusions) >= 2:
        report.append("  可能是简化的人形模型")
    else:
        report.append("  可能是简单几何体或头部模型")

    if aspect_ratios[1] > 0.6:
        report.append("  模型相对对称，适合使用对称UV映射")
    else:
        report.append("  模型不对称，需要特殊的UV映射策略")

    # 保存报告
    report_path = output_dir / "3d_mesh_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"📊 3D网格分析报告已保存: {report_path}")

    # 创建可视化图像
    create_mesh_visualization(mesh, analysis, output_dir)

def create_mesh_visualization(mesh, analysis, output_dir):
    """创建网格可视化"""
    print("🎨 创建网格可视化")

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        vertices = mesh.vertices
        center = analysis['center']
        centered = vertices - center

        # 创建3D散点图
        fig = plt.figure(figsize=(15, 5))

        # 子图1：原始网格
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   c='blue', alpha=0.6, s=0.1)
        ax1.set_title('原始网格')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # 子图2：主轴分析
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(centered[:, 0], centered[:, 1], centered[:, 2],
                   c='gray', alpha=0.6, s=0.1)

        # 绘制主轴
        eigenvectors = analysis['eigenvectors']
        eigenvalues = analysis['eigenvalues']
        colors = ['red', 'green', 'blue']

        for i, (vec, val, color) in enumerate(zip(eigenvectors.T, eigenvalues, colors)):
            scale = np.sqrt(val) * 2
            ax2.quiver(0, 0, 0, vec[0]*scale, vec[1]*scale, vec[2]*scale,
                      color=color, arrow_length_ratio=0.1, linewidth=3,
                      label=f'主轴{i+1}')

        ax2.set_title('主轴分析')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()

        # 子图3：突出部分检测
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(centered[:, 0], centered[:, 1], centered[:, 2],
                   c='lightgray', alpha=0.3, s=0.1)

        # 标注突出部分
        protrusions = analysis['protrusions']
        colors_prot = plt.cm.Set1(np.linspace(0, 1, len(protrusions)))

        for i, (prot, color) in enumerate(zip(protrusions, colors_prot)):
            points = prot['points']
            center_prot = prot['center']

            ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                       c=[color], alpha=0.8, s=2, label=f'突出部分{i+1}')
            ax3.scatter([center_prot[0]], [center_prot[1]], [center_prot[2]],
                       c='black', s=50, marker='x')

        ax3.set_title('突出部分检测')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        if len(protrusions) > 0:
            ax3.legend()

        plt.tight_layout()

        # 保存可视化
        viz_path = output_dir / "mesh_analysis_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 网格分析可视化已保存: {viz_path}")

    except ImportError:
        print("⚠️ matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"⚠️ 创建可视化时出错: {e}")

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
