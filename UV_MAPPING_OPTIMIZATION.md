# UV映射优化总结

## 🎯 优化目标

解决原始纹理映射中的贴图错位问题，提供更准确、更稳定的UV映射算法。

## 🔧 主要优化内容

### 1. 改进的球面投影
**原始问题**：
- 简单的球面坐标转换导致极点奇异性
- 没有考虑网格的主轴方向
- 接缝处理不当

**优化方案**：
- 使用PCA分析找到网格主轴方向
- 将顶点转换到主轴坐标系
- 改进的极角计算，避免数值不稳定
- 处理极点附近的奇异性
- 更好的接缝处理

```python
# 关键改进
cov_matrix = np.cov(centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
aligned_vertices = centered @ eigenvectors
```

### 2. 改进的柱面投影
**原始问题**：
- 固定使用Z轴作为柱轴
- 没有考虑网格的实际方向

**优化方案**：
- 自动检测网格的主轴方向
- 使用罗德里格斯旋转公式对齐坐标系
- 更稳定的高度标准化

```python
# 关键改进
main_axis = eigenvectors[:, idx[0]]  # 主轴作为柱轴
R = rodrigues_rotation_matrix(main_axis, z_axis)
aligned_vertices = centered @ R.T
```

### 3. 改进的平面投影
**原始问题**：
- 固定的投影轴选择
- 没有考虑网格的最佳投影方向

**优化方案**：
- 支持自动选择最佳投影平面
- 使用PCA找到最小变化方向作为投影法线
- 更稳定的坐标标准化

### 4. 新增智能UV投影
**功能**：
- 自动分析网格形状特征
- 根据形状比率选择最佳投影方法
- 球形物体 → 球面投影
- 柱形物体 → 柱面投影
- 扁平物体 → 平面投影

```python
# 形状分析
ratio1 = eigenvalues[1] / eigenvalues[0]
ratio2 = eigenvalues[2] / eigenvalues[0]

if ratio1 > 0.7 and ratio2 > 0.7:
    # 球形 - 球面投影
elif ratio1 > 0.3 and ratio2 < 0.3:
    # 柱形 - 柱面投影
else:
    # 扁平 - 平面投影
```

### 5. 新增保角投影
**功能**：
- 使用立体投影减少角度失真
- 更好地保持纹理的形状
- 适合需要精确角度保持的场景

### 6. UV质量评估与修复
**功能**：
- 快速评估UV映射质量
- 检测UV坐标超出范围
- 自动修复无效坐标
- 采样评估提高大网格处理速度

## 📊 优化效果对比

### 投影方法对比

| 投影方法 | 适用场景 | 优点 | 缺点 |
|----------|----------|------|------|
| **智能投影** | 通用 | 自动选择最佳方法 | 可能不适合特殊需求 |
| **改进球面投影** | 球形物体 | 均匀分布，主轴对齐 | 极点仍有轻微失真 |
| **改进柱面投影** | 柱形物体 | 主轴自动对齐 | 顶部底部可能拉伸 |
| **保角投影** | 精确角度要求 | 角度失真最小 | 面积失真较大 |
| **自动平面投影** | 扁平物体 | 最佳投影方向 | 只适合相对平坦的物体 |

### 质量指标

**原始版本**：
- UV覆盖范围：不稳定
- 坐标超出范围：经常发生
- 处理速度：慢（大网格）

**优化版本**：
- UV覆盖范围：稳定在[0,1]
- 坐标超出范围：自动修复
- 处理速度：快（采样评估）

## 🚀 使用建议

### 1. 默认推荐
```bash
# 使用智能投影（推荐）
python advanced_texture_mapping.py \
  --mesh model.glb \
  --image texture.jpg \
  --projection smart
```

### 2. 特定场景优化

**人物/动物模型**：
```bash
# 使用改进的球面投影
python advanced_texture_mapping.py \
  --projection spherical \
  --style realistic
```

**建筑/柱状物体**：
```bash
# 使用改进的柱面投影
python advanced_texture_mapping.py \
  --projection cylindrical \
  --style architectural
```

**需要精确角度**：
```bash
# 使用保角投影
python advanced_texture_mapping.py \
  --projection conformal \
  --material pbr
```

### 3. 质量优化
```bash
# 高质量设置
python advanced_texture_mapping.py \
  --texture-size 2048 \
  --style realistic \
  --material pbr
```

## 🔍 技术细节

### PCA主轴分析
```python
# 计算协方差矩阵
cov_matrix = np.cov(centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 特征值排序，获得主轴方向
idx = np.argsort(eigenvalues)[::-1]
main_axis = eigenvectors[:, idx[0]]
```

### 罗德里格斯旋转
```python
# 计算旋转矩阵
rotation_axis = np.cross(main_axis, target_axis)
angle = np.arccos(np.dot(main_axis, target_axis))
R = rodrigues_rotation_matrix(rotation_axis, angle)
```

### 立体投影
```python
# 立体投影到复平面
denom = 1 - z + 1e-8
w_real = x / denom
w_imag = y / denom
```

## 📈 性能优化

### 1. 采样评估
- 大网格（>100万面）自动采样1000个面进行质量评估
- 显著提高处理速度
- 保持评估准确性

### 2. 向量化计算
- 使用NumPy向量化操作
- 避免Python循环
- 提高计算效率

### 3. 内存优化
- 及时释放中间变量
- 使用就地操作减少内存拷贝

## 🎉 优化成果

### 解决的问题
1. ✅ **贴图错位**：通过主轴对齐和改进的坐标计算
2. ✅ **接缝问题**：更好的边界处理和坐标修复
3. ✅ **性能问题**：采样评估和向量化计算
4. ✅ **稳定性问题**：数值稳定性改进和异常处理

### 新增功能
1. ✅ **智能投影**：自动选择最佳投影方法
2. ✅ **保角投影**：减少角度失真
3. ✅ **质量评估**：实时UV映射质量监控
4. ✅ **自动修复**：无效坐标自动修复

### 生成的优化版本
- `output_improved.glb` - 智能投影版本
- `output_conformal.glb` - 保角投影版本
- 对应的纹理文件和质量报告

现在的UV映射系统更加稳定、准确，能够处理各种复杂的3D模型，显著减少了贴图错位问题！
