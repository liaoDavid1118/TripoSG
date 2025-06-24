# TripoSG 纹理映射完整指南

## 🎯 概述

本指南介绍如何为TripoSG生成的3D模型添加高质量纹理贴图。我们提供了多种纹理映射方法和风格选择。

## 📁 生成的文件

### 基础3D模型
- `output.glb` - 原始无纹理3D模型

### 带纹理的3D模型
- `output_textured.glb` - 基础纹理版本
- `output_artistic.glb` - 艺术风格纹理
- `output_wood.glb` - 木纹程序化纹理

### 纹理图像
- `output_textured_texture.png` - 基础纹理图像
- `output_artistic_texture.png` - 艺术风格纹理图像
- `output_wood_texture.png` - 木纹纹理图像

## 🚀 使用方法

### 1. 基础纹理映射

```bash
# 激活conda环境
conda activate TripoSG

# 生成带纹理的3D模型
python run_inference_with_texture.py \
  --image-input assets/example_data/hjswed.png \
  --output-path ./output_textured.glb \
  --texture-size 1024
```

### 2. 高级纹理映射

```bash
# 使用原始图像创建艺术风格纹理
python advanced_texture_mapping.py \
  --mesh output.glb \
  --image assets/example_data/hjswed.png \
  --output output_artistic.glb \
  --projection spherical \
  --style artistic \
  --material pbr

# 创建程序化木纹纹理
python advanced_texture_mapping.py \
  --mesh output.glb \
  --output output_wood.glb \
  --projection cylindrical \
  --procedural wood \
  --material pbr
```

## 🎨 纹理风格选项

### 图像风格
- **realistic** - 现实风格（增强细节和对比度）
- **artistic** - 艺术风格（边缘增强，颜色饱和）
- **vintage** - 复古风格（降低饱和度，暖色调）
- **cartoon** - 卡通风格（减少细节，增强颜色）

### 程序化纹理
- **wood** - 木纹纹理
- **marble** - 大理石纹理
- **metal** - 金属纹理

## 🗺️ UV投影方法

### 球面投影 (spherical)
- **适用于**: 球形或接近球形的物体
- **特点**: 均匀分布，适合大多数情况
- **推荐**: 人物、动物、圆形物体

### 柱面投影 (cylindrical)
- **适用于**: 柱形或高长比较大的物体
- **特点**: 垂直方向拉伸较少
- **推荐**: 建筑物、树木、瓶子

### 平面投影 (planar)
- **planar_z**: Z轴平面投影
- **planar_y**: Y轴平面投影  
- **planar_x**: X轴平面投影
- **适用于**: 相对平坦的物体
- **推荐**: 地形、平面物体

## 🎭 材质类型

### PBR材质 (pbr)
- **特点**: 物理基础渲染，最真实
- **参数**: 金属度、粗糙度、发光
- **推荐**: 现代渲染引擎

### 金属材质 (metallic)
- **特点**: 高金属度，低粗糙度
- **适用**: 金属物体

### 光泽材质 (glossy)
- **特点**: 低金属度，低粗糙度
- **适用**: 塑料、陶瓷

### 简单材质 (simple)
- **特点**: 基础材质，兼容性好
- **适用**: 简单场景

## 📐 参数详解

### 纹理分辨率 (--texture-size)
- **512**: 低质量，文件小
- **1024**: 标准质量（推荐）
- **2048**: 高质量，文件大
- **4096**: 超高质量，文件很大

### 推理参数
- **--num-inference-steps**: 推理步数（30-100）
  - 更多步数 = 更高质量，更慢速度
- **--guidance-scale**: 引导尺度（5.0-10.0）
  - 更高值 = 更忠实原图，更少创意
- **--seed**: 随机种子
  - 固定种子可重现相同结果

## 🛠️ 实用示例

### 示例1：为人物模型添加现实纹理
```bash
python run_inference_with_texture.py \
  --image-input portrait.jpg \
  --output-path person_textured.glb \
  --texture-size 2048 \
  --num-inference-steps 50 \
  --guidance-scale 7.0
```

### 示例2：创建艺术风格雕塑
```bash
# 先生成基础模型
python run_inference.py \
  --image-input sculpture.jpg \
  --output-path sculpture_base.glb

# 添加大理石纹理
python advanced_texture_mapping.py \
  --mesh sculpture_base.glb \
  --output sculpture_marble.glb \
  --projection spherical \
  --procedural marble \
  --material pbr
```

### 示例3：创建复古风格物体
```bash
python advanced_texture_mapping.py \
  --mesh object.glb \
  --image vintage_photo.jpg \
  --output object_vintage.glb \
  --projection cylindrical \
  --style vintage \
  --material glossy
```

## 📱 查看3D模型

### 在线查看器
- [glTF Viewer](https://gltf-viewer.donmccurdy.com/)
- [Three.js Editor](https://threejs.org/editor/)

### 桌面软件
- **Blender** (免费，功能强大)
- **3D Viewer** (Windows内置)
- **MeshLab** (免费，科学可视化)

### 游戏引擎
- **Unity** (支持GLB导入)
- **Unreal Engine** (支持GLB导入)

## 🔧 故障排除

### 常见问题

1. **纹理看起来拉伸**
   - 尝试不同的投影方法
   - 调整纹理分辨率

2. **颜色不正确**
   - 检查原始图像质量
   - 尝试不同的风格设置

3. **文件太大**
   - 降低纹理分辨率
   - 使用网格简化

4. **纹理不显示**
   - 确保查看器支持PBR材质
   - 尝试简单材质类型

### 性能优化

1. **减少文件大小**
   ```bash
   # 简化网格面数
   python run_inference_with_texture.py \
     --faces 50000 \
     --texture-size 512
   ```

2. **批量处理**
   ```bash
   # 处理多个图像
   for img in *.jpg; do
     python run_inference_with_texture.py \
       --image-input "$img" \
       --output-path "${img%.*}_textured.glb"
   done
   ```

## 📊 质量对比

| 设置 | 文件大小 | 质量 | 处理时间 |
|------|----------|------|----------|
| 512px, 简单材质 | ~5MB | 低 | 快 |
| 1024px, PBR材质 | ~15MB | 中 | 中等 |
| 2048px, PBR材质 | ~50MB | 高 | 慢 |
| 4096px, PBR材质 | ~200MB | 极高 | 很慢 |

## 🎉 总结

通过本指南，您可以：
1. ✅ 为TripoSG生成的3D模型添加高质量纹理
2. ✅ 选择适合的投影方法和材质类型
3. ✅ 创建多种风格的纹理效果
4. ✅ 优化文件大小和质量平衡
5. ✅ 在各种软件中查看和使用3D模型

现在您可以创建具有专业级纹理的3D模型了！
