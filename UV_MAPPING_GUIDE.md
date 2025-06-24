# TripoSG UV坐标映射功能使用指南

## 🎯 功能概述

现在 TripoSG 的 `inference_triposg.py` 脚本支持在生成 GLB 文件时自动添加 UV 坐标映射信息，这将大大改善后续纹理映射的效果，解决原图贴图错位问题。

## 🚀 快速开始

### 基础用法（不生成UV坐标）
```bash
python scripts/inference_triposg.py \
  --image-input your_image.jpg \
  --output-path output.glb
```

### 生成UV坐标（推荐）
```bash
python scripts/inference_triposg.py \
  --image-input your_image.jpg \
  --output-path output_with_uv.glb \
  --generate-uv
```

### 指定UV映射方法
```bash
python scripts/inference_triposg.py \
  --image-input your_image.jpg \
  --output-path output_spherical.glb \
  --generate-uv \
  --uv-method spherical
```

## 📋 命令行参数

### 基础参数
- `--image-input`: 输入图像路径（必需）
- `--output-path`: 输出GLB文件路径（默认: `./output.glb`）
- `--seed`: 随机种子（默认: 42）
- `--num-inference-steps`: 推理步数（默认: 50）
- `--guidance-scale`: 引导尺度（默认: 7.0）
- `--faces`: 目标面数，-1表示不简化（默认: -1）

### UV坐标参数
- `--generate-uv`: 启用UV坐标生成（标志参数）
- `--uv-method`: UV映射方法，可选值：
  - `smart`: 智能投影（默认，自动选择最佳方法）
  - `spherical`: 球面投影
  - `cylindrical`: 柱面投影
  - `planar`: 平面投影

## 🗺️ UV映射方法详解

### 1. 智能投影 (smart) - 推荐
```bash
--uv-method smart
```
**特点**：
- 自动分析网格形状特征
- 根据形状比率选择最佳投影方法
- 适用于大多数场景

**选择逻辑**：
- 球形物体 → 球面投影
- 柱形物体 → 柱面投影  
- 扁平物体 → 平面投影

### 2. 球面投影 (spherical)
```bash
--uv-method spherical
```
**适用场景**：
- 人物、动物模型
- 球形或接近球形的物体
- 需要均匀纹理分布的模型

**优点**：
- 纹理分布均匀
- 主轴自动对齐
- 处理极点奇异性

### 3. 柱面投影 (cylindrical)
```bash
--uv-method cylindrical
```
**适用场景**：
- 建筑物、柱子
- 瓶子、罐子等柱状物体
- 具有明显主轴的物体

**优点**：
- 主轴自动检测和对齐
- 高度方向纹理连续
- 适合有明显方向性的物体

### 4. 平面投影 (planar)
```bash
--uv-method planar
```
**适用场景**：
- 扁平物体
- 徽章、标牌
- 相对平坦的表面

**优点**：
- 最小的纹理失真
- 自动选择最佳投影方向
- 适合平面或接近平面的物体

## 📊 输出信息解读

生成完成后，脚本会显示详细信息：

```
✅ 3D模型生成完成!
📁 输出文件: output_with_uv.glb
📊 网格信息:
  - 顶点数: 12543
  - 面数: 25086
  - UV坐标: ✅ 已生成 (12543 个)
  - UV映射方法: smart
```

## 🔧 高级用法

### 高质量生成
```bash
python scripts/inference_triposg.py \
  --image-input portrait.jpg \
  --output-path high_quality.glb \
  --generate-uv \
  --uv-method spherical \
  --num-inference-steps 100 \
  --guidance-scale 8.0
```

### 简化网格并保持UV坐标
```bash
python scripts/inference_triposg.py \
  --image-input model.jpg \
  --output-path simplified.glb \
  --generate-uv \
  --faces 5000
```

## 🎨 与纹理映射工具配合使用

生成带UV坐标的GLB文件后，可以直接用于现有的纹理映射工具：

### 1. 角色纹理映射
```bash
python character_texture_mapping.py \
  --mesh output_with_uv.glb \
  --image original_image.jpg \
  --output textured_character.glb
```

### 2. 语义纹理映射
```bash
python semantic_image_texture_mapping.py \
  --mesh output_with_uv.glb \
  --image original_image.jpg \
  --output semantic_textured.glb
```

### 3. 高级纹理映射
```bash
python advanced_texture_mapping.py \
  --mesh output_with_uv.glb \
  --image original_image.jpg \
  --output advanced_textured.glb
```

## 🔍 测试功能

运行测试脚本验证UV生成功能：

```bash
python test_uv_generation.py
```

## 📝 注意事项

1. **向后兼容性**：默认不生成UV坐标，需要明确指定 `--generate-uv`
2. **网格简化**：如果指定了 `--faces` 参数，UV坐标会在简化后重新生成
3. **性能影响**：生成UV坐标会增加少量计算时间
4. **文件大小**：带UV坐标的GLB文件会稍大一些

## 🐛 故障排除

### 问题：UV坐标超出[0,1]范围
**解决方案**：脚本会自动裁剪到有效范围，如果仍有问题，尝试不同的UV映射方法

### 问题：纹理映射仍然错位
**解决方案**：
1. 尝试不同的UV映射方法
2. 检查原始图像质量
3. 使用语义感知的纹理映射工具

### 问题：生成的UV坐标质量不佳
**解决方案**：
1. 使用 `smart` 方法让系统自动选择
2. 根据物体类型手动选择合适的投影方法
3. 检查网格质量，必要时调整推理参数

## 🎉 总结

通过添加UV坐标映射功能，TripoSG现在可以生成更适合纹理处理的3D模型，为后续的原图贴图处理提供了坚实的基础。建议在所有需要纹理映射的场景中都启用此功能。
