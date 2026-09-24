#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格化 / 模型优化模块（基于 open3d）

把（稀疏或稠密）点云变成三角网格，并做简化、平滑、导出。
这是「三维重建计算」链路的最后一环：点云 → 网格 → 模型优化。

注意：本模块只做「点云 → 网格」这后半段；
「图像 → 稠密点云」（密集重建 MVS）是另一个模块的事，之后会补。
"""

import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def points_to_mesh(points, colors=None, out_dir="output_mesh",
                   voxel_size=None, poisson_depth=8,
                   target_faces=20000, smooth_iterations=2,
                   density_crop=True, density_radius_ratio=2.0,
                   density_min_points=1, gamma=1.5, write_files=True):
    """点云 → 网格 → 简化 → 平滑 → 导出（PLY / OBJ / GLB）。

    Args:
        points: (N, 3) 点坐标
        colors: (N, 3) 颜色，0-255 或 0-1 均可（可选）
        out_dir: 输出目录
        voxel_size: 体素下采样尺寸，None 则按包围盒自动估算
        poisson_depth: Poisson 重建八叉树深度（越大越精细，越慢）
        target_faces: 简化后的目标面数
        smooth_iterations: Laplacian 平滑迭代次数（0 表示不平滑）
        density_crop: 密度裁剪——去掉 Poisson 在空区域（天空/鼓包）凭空生成的曲面
        density_radius_ratio: 密度半径 = voxel_size * 该比例
        density_min_points: 半径内至少这么多稠密点才算「有支撑」
        gamma: 提亮系数（>1 提亮，c^(1/gamma)）；1.0 表示不动
        write_files: 是否导出 PLY/OBJ/GLB 到 out_dir；False 则只计算并返回网格

    Returns:
        mesh: open3d.geometry.TriangleMesh
    """
    if write_files:
        os.makedirs(out_dir, exist_ok=True)

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(f"points 应为 (N>=3, 3)，实际 {points.shape}")

    # 1. 建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors_01 = None
    if colors is not None and len(colors) == len(points):
        c = np.asarray(colors, dtype=np.float64)
        if c.max() > 1.0:
            c = c / 255.0
        colors_01 = np.clip(c, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors_01)

    # 提亮（gamma 校正，c^(1/gamma)）：网格在带光照的查看器（MeshLab 等）里会被
    # 压暗，这里预先把颜色提亮，导出后观感更好（gamma=1.0 表示不动）。
    if gamma != 1.0 and colors_01 is not None:
        colors_01 = np.clip(np.power(colors_01, 1.0 / gamma), 0.0, 1.0)

    print(f"[1] 输入点数: {len(pcd.points)}")

    # 2. 按包围盒自动估算尺度参数（点云尺度不定，必须自适应）
    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if voxel_size is None:
        voxel_size = bbox_diag / 100.0
    radius = bbox_diag / 25.0

    # 3. 体素下采样（稀疏点云时几乎不减）
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"[2] 体素下采样后: {len(pcd.points)} 点 (voxel={voxel_size:.4f})")

    # 4. 法向估计 + 朝向一致化（Poisson 依赖法向）
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    print("[3] 法向估计 + 朝向一致化完成")

    # 5. Poisson 表面重建
    print(f"[4] Poisson 重建中 (depth={poisson_depth})...")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth)

    if len(mesh.triangles) == 0:
        print("[警告] Poisson 网格为空——点云太稀疏。这是预期现象，"
              "等密集重建输出稠密点云后网格会好很多。")
        return mesh

    # Poisson 会在点云外生成一个闭合"鼓包"，裁掉点云包围盒之外的部分
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.05, bbox.get_center())
    mesh = mesh.crop(bbox)
    print(f"[4] Poisson 网格: {len(mesh.vertices)} 顶点 / {len(mesh.triangles)} 面")

    # 5. 密度裁剪：Poisson 会在没有点云支撑的空区域（天空/鼓包）凭空生成曲面。
    # 用「网格顶点附近有无稠密点支撑」把低密度区域砍掉。
    if density_crop and len(points) > 0:
        tree = cKDTree(points)  # 原始稠密点（下采样前）
        r = voxel_size * density_radius_ratio
        verts = np.asarray(mesh.vertices)
        d, _ = tree.query(verts, k=density_min_points)
        d_kth = d[:, -1] if d.ndim > 1 else d
        keep = d_kth < r
        mesh.remove_vertices_by_mask(np.logical_not(keep).tolist())
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        print(f"[5] 密度裁剪: {keep.sum()}/{len(keep)} 顶点保留 "
              f"(r={r:.3f}, min={density_min_points})")

    # 6. 网格简化（二次误差度量）
    if len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)
        print(f"[5] 简化后: {len(mesh.triangles)} 面")

    # 7. Laplacian 平滑
    if smooth_iterations > 0 and len(mesh.vertices) > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smooth_iterations)
        print(f"[6] Laplacian 平滑 {smooth_iterations} 次完成")
    mesh.compute_vertex_normals()

    # 8. 把点云颜色就近映射到网格顶点（粗糙的顶点着色，正式贴纹理后续再做）
    if colors_01 is not None:
        tree = cKDTree(points)  # 用原始点（下采样前的）找最近邻
        _, idx = tree.query(np.asarray(mesh.vertices))
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors_01[idx])
        print("[7] 顶点着色完成")

    # 9. 导出（write_files=False 时只返回网格，不落盘）
    if write_files:
        ply_path = os.path.join(out_dir, "mesh.ply")
        obj_path = os.path.join(out_dir, "mesh.obj")
        glb_path = os.path.join(out_dir, "mesh.glb")
        o3d.io.write_triangle_mesh(ply_path, mesh)
        o3d.io.write_triangle_mesh(obj_path, mesh)
        print(f"[8] 导出 PLY: {ply_path}")
        print(f"[8] 导出 OBJ: {obj_path}")
        try:
            o3d.io.write_triangle_mesh(glb_path, mesh)
            print(f"[8] 导出 GLB(前端 Three.js 直接加载): {glb_path}")
        except Exception as e:
            print(f"[8] GLB 导出失败（前端可先用 OBJ）: {e}")

    return mesh


if __name__ == "__main__":
    # 自包含演示：直接读 output/ 里的稀疏点云跑网格化，不依赖其它模块
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sfm_project
    structure = np.load(os.path.join(_ROOT, "output", "structure.npy"))
    colors = np.load(os.path.join(_ROOT, "output", "colors.npy"))
    out_dir = os.path.join(_ROOT, "output", "mesh")

    mesh = points_to_mesh(structure, colors, out_dir=out_dir)

    print("\n" + "=" * 50)
    print(f"完成！最终网格: {len(mesh.vertices)} 顶点 / {len(mesh.triangles)} 面")
    print("查看方式：")
    print(f"  - Windows 3D 查看器打开 GLB: {os.path.join(out_dir, 'mesh.glb')}")
    print(f"  - MeshLab/CloudCompare 打开 PLY: {os.path.join(out_dir, 'mesh.ply')}")
    print("=" * 50)
