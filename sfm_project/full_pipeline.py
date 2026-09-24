#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整重建统一入口：SfM 稀疏 → 稠密 → 网格化，一条龙封装成可调用函数。

供后端（B，FastAPI/Flask）与前端（C，取 GLB）直接 import，无需接触
SFMPipeline / dense_reconstruct / points_to_mesh 的内部细节。

用法：
    from full_pipeline import reconstruct_full
    result = reconstruct_full("images2", save=True)          # 全链路
    print(result.mesh_glb)    # 前端直接加载的 GLB 路径
    print(result.to_dict())   # JSON 可序列化的统计 + 产物路径
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass

from config import Config
from main import reconstruct, ReconstructionResult
from core.dense_mvs import dense_reconstruct, DenseResult
from core.meshing import points_to_mesh


_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# 稠密重建的默认参数（与 dense_mvs.__main__ 演示一致，是实测效果较好的那组）。
# 想要更快 / 更省显存，通过 dense_kwargs 覆盖（如 num_depths=96, max_dim=480）。
_DEFAULT_DENSE = dict(
    num_ref_views=6, num_depths=256, max_dim=720,
    viewing_angle=70.0, peak_quantile=0.2, median_ksize=7,
)


@dataclass
class FullResult:
    """完整重建（稀疏 + 稠密 + 网格）的结果。

    字段：
        sparse:       稀疏 SfM 结果（ReconstructionResult）
        dense:        稠密点云结果（DenseResult）；do_dense=False 时为 None
        mesh:         open3d 网格（TriangleMesh）；do_mesh=False 时为 None
        dense_ply:    稠密点云 PLY 路径（未生成则为 None）
        mesh_ply / mesh_obj / mesh_glb: 网格产物路径（未生成则为 None）
        images_used:  参与重建（已跳过坏帧后）的图像数量
    """
    sparse: ReconstructionResult
    dense: DenseResult = None
    mesh: object = None
    dense_ply: str = None
    mesh_ply: str = None
    mesh_obj: str = None
    mesh_glb: str = None
    images_used: int = 0

    @property
    def num_points(self):
        """最终点云数量（有稠密用稠密，否则退回稀疏）。"""
        if self.dense is not None:
            return self.dense.num_points
        return self.sparse.num_points

    def to_dict(self):
        """转 JSON 可序列化字典（统计 + 产物路径，不含大数组 / open3d 对象）。"""
        mesh_v = mesh_f = 0
        if self.mesh is not None:
            mesh_v = int(len(np.asarray(self.mesh.vertices)))
            mesh_f = int(len(np.asarray(self.mesh.triangles)))
        return {
            'sparse': self.sparse.to_dict(),
            'images_used': int(self.images_used),
            'num_points': int(self.num_points),
            'dense': self.dense.to_dict() if self.dense is not None else None,
            'mesh_vertices': mesh_v,
            'mesh_triangles': mesh_f,
            'dense_ply': self.dense_ply,
            'mesh_ply': self.mesh_ply,
            'mesh_obj': self.mesh_obj,
            'mesh_glb': self.mesh_glb,
        }


def _list_images(image_path):
    """把目录或文件路径列表统一成「有序图像路径列表」（与 SFMPipeline._load_images 同规则）。"""
    if isinstance(image_path, (list, tuple)):
        return [p for p in image_path if p.lower().endswith(_IMAGE_EXTS)]
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")
    names = sorted([f for f in os.listdir(image_path)
                    if f.lower().endswith(_IMAGE_EXTS)])
    return [os.path.join(image_path, n) for n in names]


def reconstruct_full(image_path, config=None, save=True, do_dense=True, do_mesh=True,
                     visualize=False, dense_kwargs=None, mesh_kwargs=None):
    """完整重建统一入口：SfM 稀疏 → 稠密 → 网格，一条龙。

    Args:
        image_path: 图像目录路径，或图像文件路径列表（顺序敏感，环拍需按序传入）。
        config: 可选 Config；不传则新建（自动标定 K）。
        save: 是否把点云/网格产物写到 output/（False 则只算不写，便于测试/试跑）。
        do_dense: 是否做稠密重建（plane-sweep MVS）。
        do_mesh: 是否做网格化（在稠密点云上；do_dense=False 时退化到稀疏点云）。
        visualize: 是否弹 matplotlib 窗口（后端/服务器/无头环境必须 False）。
        dense_kwargs: 传给 dense_reconstruct 的可选参数字典（覆盖默认值）。
        mesh_kwargs: 传给 points_to_mesh 的可选参数字典。

    Returns:
        FullResult
    """
    if config is None:
        config = Config()
    dense_kwargs = dict(_DEFAULT_DENSE, **(dense_kwargs or {}))
    mesh_kwargs = dict(mesh_kwargs or {})

    # 1. 稀疏 SfM（内部会按 AUTO_CALIBRATE 自动确定 K，并写回 config.K）
    sparse = reconstruct(image_path, config=config, save=save, visualize=visualize)

    if not (do_dense or do_mesh):
        return FullResult(sparse=sparse, images_used=sparse.num_images)

    # 2. 按同样顺序加载图像，并按 registered_indices 对齐（跳过坏帧后位姿与图像不再等长）
    image_names = _list_images(image_path)
    images = [cv2.imread(p) for p in image_names]
    idx = sparse.registered_indices
    if idx is None or len(idx) != len(sparse.rotations):
        idx = list(range(len(sparse.rotations)))
    images_used = [images[i] for i in idx]
    n_used = len(images_used)

    if n_used < 2:
        print(f"[reconstruct_full] 注册帧数不足（{n_used}<2），跳过稠密/网格化")
        return FullResult(sparse=sparse, images_used=n_used)

    # 3. 稠密重建
    dense = None
    dense_ply = None
    if do_dense:
        dense = dense_reconstruct(
            images_used, sparse.rotations, sparse.motions, config.K,
            sparse_points=sparse.points, **dense_kwargs)
        if save:
            import open3d as o3d
            dense_dir = os.path.join(config.OUTPUT_PATH, "dense")
            os.makedirs(dense_dir, exist_ok=True)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dense.points)
            pcd.colors = o3d.utility.Vector3dVector(dense.colors / 255.0)
            dense_ply = os.path.join(dense_dir, "dense_cloud.ply")
            o3d.io.write_point_cloud(dense_ply, pcd)

    # 4. 网格化（优先稠密点云，否则稀疏点云）
    mesh = None
    mesh_ply = mesh_obj = mesh_glb = None
    if do_mesh:
        if dense is not None:
            mesh_points, mesh_colors = dense.points, dense.colors
            mesh_out = os.path.join(config.OUTPUT_PATH, "dense", "mesh")
        else:
            mesh_points, mesh_colors = sparse.points, sparse.colors
            mesh_out = os.path.join(config.OUTPUT_PATH, "mesh")
        mesh = points_to_mesh(mesh_points, mesh_colors, out_dir=mesh_out,
                              write_files=save, **mesh_kwargs)
        if save:
            mesh_ply = os.path.join(mesh_out, "mesh.ply")
            mesh_obj = os.path.join(mesh_out, "mesh.obj")
            mesh_glb = os.path.join(mesh_out, "mesh.glb")

    return FullResult(sparse=sparse, dense=dense, mesh=mesh,
                      dense_ply=dense_ply, mesh_ply=mesh_ply, mesh_obj=mesh_obj,
                      mesh_glb=mesh_glb, images_used=n_used)


if __name__ == "__main__":
    # 自包含演示：等价于「后端」的一次完整调用
    import sys
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    result = reconstruct_full(Config().DATA_PATH, save=True)

    print("\n" + "=" * 50)
    print("完整重建完成")
    print(f"  稀疏点: {result.sparse.num_points}  稠密点: "
          f"{result.dense.num_points if result.dense else 0}  图像: {result.images_used}")
    if result.mesh is not None:
        print(f"  网格: {len(np.asarray(result.mesh.vertices))} 顶点 / "
              f"{len(np.asarray(result.mesh.triangles))} 面")
    print(f"  前端 GLB: {result.mesh_glb}")
    print("=" * 50)
