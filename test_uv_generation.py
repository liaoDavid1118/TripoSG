#!/usr/bin/env python3
"""
测试UV坐标生成功能
"""

import sys
import os
import numpy as np
import trimesh

# 添加脚本目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# 导入UV生成函数
from inference_triposg import generate_uv_coordinates, smart_uv_projection, spherical_projection, cylindrical_projection, planar_projection

def create_test_mesh():
    """创建一个简单的测试网格（立方体）"""
    # 创建立方体顶点
    vertices = np.array([
        [-1, -1, -1],  # 0
        [ 1, -1, -1],  # 1
        [ 1,  1, -1],  # 2
        [-1,  1, -1],  # 3
        [-1, -1,  1],  # 4
        [ 1, -1,  1],  # 5
        [ 1,  1,  1],  # 6
        [-1,  1,  1],  # 7
    ], dtype=np.float32)
    
    # 创建立方体面
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # 底面
        [4, 7, 6], [4, 6, 5],  # 顶面
        [0, 4, 5], [0, 5, 1],  # 前面
        [2, 6, 7], [2, 7, 3],  # 后面
        [0, 3, 7], [0, 7, 4],  # 左面
        [1, 5, 6], [1, 6, 2],  # 右面
    ])
    
    return vertices, faces

def test_uv_methods():
    """测试所有UV映射方法"""
    print("🧪 开始测试UV坐标生成功能...")
    
    # 创建测试网格
    vertices, faces = create_test_mesh()
    print(f"📦 创建测试立方体 - 顶点数: {len(vertices)}, 面数: {len(faces)}")
    
    # 测试所有UV映射方法
    methods = ['smart', 'spherical', 'cylindrical', 'planar']
    
    for method in methods:
        print(f"\n🔍 测试 {method} 投影...")
        try:
            uv_coords = generate_uv_coordinates(vertices, method=method)
            
            # 验证UV坐标
            if uv_coords is not None and len(uv_coords) == len(vertices):
                # 检查UV坐标范围
                u_min, u_max = uv_coords[:, 0].min(), uv_coords[:, 0].max()
                v_min, v_max = uv_coords[:, 1].min(), uv_coords[:, 1].max()
                
                print(f"  ✅ UV坐标生成成功")
                print(f"  📊 UV范围: U[{u_min:.3f}, {u_max:.3f}], V[{v_min:.3f}, {v_max:.3f}]")
                
                # 检查是否在有效范围内
                if 0 <= u_min <= u_max <= 1 and 0 <= v_min <= v_max <= 1:
                    print(f"  ✅ UV坐标范围有效")
                else:
                    print(f"  ⚠️ UV坐标超出[0,1]范围")
                    
            else:
                print(f"  ❌ UV坐标生成失败")
                
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
    
    return True

def test_mesh_with_uv():
    """测试创建带UV坐标的网格"""
    print(f"\n🔧 测试创建带UV坐标的网格...")
    
    vertices, faces = create_test_mesh()
    uv_coords = generate_uv_coordinates(vertices, method='smart')
    
    try:
        # 创建网格
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # 添加UV坐标
        material = trimesh.visual.material.SimpleMaterial(
            diffuse=[255, 255, 255, 255]
        )
        
        texture_visual = trimesh.visual.TextureVisuals(
            uv=uv_coords,
            material=material
        )
        
        mesh.visual = texture_visual
        
        # 验证
        if hasattr(mesh.visual, 'uv') and len(mesh.visual.uv) == len(vertices):
            print(f"  ✅ 带UV坐标的网格创建成功")
            print(f"  📊 顶点数: {len(mesh.vertices)}")
            print(f"  📊 面数: {len(mesh.faces)}")
            print(f"  📊 UV坐标数: {len(mesh.visual.uv)}")
            
            # 测试导出
            test_output = "test_mesh_with_uv.glb"
            mesh.export(test_output)
            print(f"  💾 测试文件已保存: {test_output}")
            
            # 清理测试文件
            if os.path.exists(test_output):
                os.remove(test_output)
                print(f"  🗑️ 测试文件已清理")
                
            return True
        else:
            print(f"  ❌ UV坐标添加失败")
            return False
            
    except Exception as e:
        print(f"  ❌ 网格创建失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 TripoSG UV坐标生成功能测试")
    print("=" * 60)
    
    try:
        # 测试UV映射方法
        test_uv_methods()
        
        # 测试网格创建
        test_mesh_with_uv()
        
        print(f"\n" + "=" * 60)
        print("✅ 所有测试完成!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
