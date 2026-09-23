#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化工具模块
提供3D点云可视化、匹配结果可视化等功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


def visualize_3d_matplotlib(structure, colors, title='3D Reconstruction Result'):
    """
    使用matplotlib显示3D点云

    Args:
        structure: Nx3的点云坐标
        colors: Nx3的颜色值（RGB，0-255）
        title: 窗口标题
    """
    if structure is None or len(structure) == 0:
        print("No points to visualize")
        return

    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    # 创建3D坐标轴
    ax = fig.add_subplot(111, projection='3d')

    # 准备颜色
    if colors is not None:
        # 确保颜色值在0-1之间
        if colors.max() > 1:
            colors_normalized = colors / 255.0
        else:
            colors_normalized = colors
    else:
        # 如果没有颜色，使用深度信息作为颜色
        z_normalized = (structure[:, 2] - structure[:, 2].min()) / \
                       (structure[:, 2].max() - structure[:, 2].min())
        colors_normalized = plt.cm.jet(z_normalized)[:, :3]

    # 绘制点云
    scatter = ax.scatter(structure[:, 0], structure[:, 1], structure[:, 2],
                         c=colors_normalized, s=2, alpha=0.7)

    # 设置坐标轴标签
    ax.set_xlabel('X axis', fontsize=12)
    ax.set_ylabel('Y axis', fontsize=12)
    ax.set_zlabel('Z axis', fontsize=12)

    # 设置坐标轴范围
    x_range = structure[:, 0].max() - structure[:, 0].min()
    y_range = structure[:, 1].max() - structure[:, 1].min()
    z_range = structure[:, 2].max() - structure[:, 2].min()
    max_range = max(x_range, y_range, z_range)

    center_x = (structure[:, 0].max() + structure[:, 0].min()) / 2
    center_y = (structure[:, 1].max() + structure[:, 1].min()) / 2
    center_z = (structure[:, 2].max() + structure[:, 2].min()) / 2

    ax.set_xlim(center_x - max_range / 2, center_x + max_range / 2)
    ax.set_ylim(center_y - max_range / 2, center_y + max_range / 2)
    ax.set_zlim(center_z - max_range / 2, center_z + max_range / 2)

    # 设置视角（可选）
    ax.view_init(elev=135, azim=90)

    # 添加标题和网格
    ax.set_title(f'Total Points: {len(structure)}', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_3d_mayavi(structure, colors=None):
    """
    使用mayavi显示3D点云（更流畅，支持交互）
    需要安装: pip install mayavi

    Args:
        structure: Nx3的点云坐标
        colors: Nx3的颜色值（可选）
    """
    try:
        from mayavi import mlab
    except ImportError:
        print("Mayavi not installed. Skipping mayavi visualization.")
        return

    if structure is None or len(structure) == 0:
        print("No points to visualize")
        return

    # 准备颜色
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
    else:
        # 使用深度信息作为颜色
        z_normalized = (structure[:, 2] - structure[:, 2].min()) / \
                       (structure[:, 2].max() - structure[:, 2].min())
        colors = plt.cm.jet(z_normalized)[:, :3]

    # 创建点云可视化
    fig = mlab.figure(size=(800, 600), bgcolor=(0.1, 0.1, 0.1))

    # 绘制点云
    pts = mlab.points3d(structure[:, 0], structure[:, 1], structure[:, 2],
                        scale_factor=0.02, color=(1, 1, 1))

    # 如果有颜色信息，设置颜色
    if colors is not None:
        pts.mlab_source.dataset.point_data.scalars = colors
        pts.module_manager.scalar_lut_manager.lut_mode = 'RdYlBu'

    # 添加坐标轴
    mlab.axes(pts, color=(0.8, 0.8, 0.8))

    # 显示
    mlab.show()


def visualize_matches(image1, image2, keypoints1, keypoints2, matches,
                      title='Feature Matches'):
    """
    可视化特征匹配结果

    Args:
        image1: 第一张图像
        image2: 第二张图像
        keypoints1: 第一张图像的关键点
        keypoints2: 第二张图像的关键点
        matches: 匹配结果列表
        title: 窗口标题
    """
    # 创建匹配可视化图像
    img_matches = np.empty((max(image1.shape[0], image2.shape[0]),
                            image1.shape[1] + image2.shape[1], 3),
                           dtype=np.uint8)

    # 绘制匹配
    cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                    matches, img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 显示图像
    cv2.imshow(title, img_matches)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def visualize_camera_poses(rotations, motions, points_3d=None):
    """
    可视化相机位姿

    Args:
        rotations: 旋转矩阵列表
        motions: 平移向量列表
        points_3d: 3D点云（可选）
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制相机位姿
    for i, (R, t) in enumerate(zip(rotations, motions)):
        # 相机位置
        camera_pos = t.flatten()

        # 绘制相机位置点
        ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2],
                   c='red', s=50, marker='o')

        # 绘制相机朝向（相机光轴方向）
        # 相机坐标系下的Z轴方向（在相机坐标系中是(0,0,1)）
        camera_z = R @ np.array([0, 0, 1])
        ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                  camera_z[0], camera_z[1], camera_z[2],
                  length=0.5, color='blue', alpha=0.5)

        # 添加标签
        ax.text(camera_pos[0], camera_pos[1], camera_pos[2],
                f'Cam {i}', fontsize=10)

    # 绘制点云
    if points_3d is not None and len(points_3d) > 0:
        # 下采样以加快显示速度
        if len(points_3d) > 5000:
            indices = np.random.choice(len(points_3d), 5000, replace=False)
            points_subset = points_3d[indices]
        else:
            points_subset = points_3d

        ax.scatter(points_subset[:, 0], points_subset[:, 1], points_subset[:, 2],
                   c='gray', s=1, alpha=0.3)

    # 设置坐标轴
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Camera Poses and Point Cloud')

    # 设置等比例坐标轴
    max_range = max(ax.get_xlim()[1] - ax.get_xlim()[0],
                    ax.get_ylim()[1] - ax.get_ylim()[0],
                    ax.get_zlim()[1] - ax.get_zlim()[0])
    mid_x = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    mid_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    mid_z = (ax.get_zlim()[0] + ax.get_zlim()[1]) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    plt.tight_layout()
    plt.show()


def visualize_depth_map(depth_map, title='Depth Map'):
    """
    可视化深度图

    Args:
        depth_map: 深度图（2D数组）
        title: 窗口标题
    """
    plt.figure(figsize=(10, 8))
    plt.title(title)

    # 使用jet colormap显示深度图
    im = plt.imshow(depth_map, cmap='jet', interpolation='bilinear')
    plt.colorbar(im, label='Depth Value')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_reprojection(image, points_2d_obs, points_2d_proj, title='Reprojection'):
    """
    可视化重投影误差

    Args:
        image: 原始图像
        points_2d_obs: 观测到的2D点
        points_2d_proj: 重投影的2D点
        title: 窗口标题
    """
    img_copy = image.copy()

    # 绘制观测点（绿色）
    for pt in points_2d_obs:
        cv2.circle(img_copy, tuple(pt.astype(int)), 3, (0, 255, 0), -1)

    # 绘制重投影点（红色）
    for pt in points_2d_proj:
        cv2.circle(img_copy, tuple(pt.astype(int)), 3, (0, 0, 255), -1)

    # 绘制连线（蓝色）
    for obs, proj in zip(points_2d_obs, points_2d_proj):
        cv2.line(img_copy, tuple(obs.astype(int)),
                 tuple(proj.astype(int)), (255, 0, 0), 1)

    # 显示图像
    cv2.imshow(title, img_copy)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def visualize_error_distribution(reprojection_errors):
    """
    可视化重投影误差分布

    Args:
        reprojection_errors: 误差列表
    """
    plt.figure(figsize=(12, 5))

    # 子图1：直方图
    plt.subplot(1, 2, 1)
    plt.hist(reprojection_errors, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Reprojection Error (pixels)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)

    # 子图2：累积分布
    plt.subplot(1, 2, 2)
    sorted_errors = np.sort(reprojection_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative, 'b-', linewidth=2)
    plt.xlabel('Reprojection Error (pixels)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution')
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    plt.figtext(0.02, 0.02,
                f'Mean: {np.mean(reprojection_errors):.4f} px\n'
                f'Std: {np.std(reprojection_errors):.4f} px\n'
                f'Median: {np.median(reprojection_errors):.4f} px\n'
                f'Max: {np.max(reprojection_errors):.4f} px',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


# 主函数，用于测试
if __name__ == "__main__":
    # 测试代码
    print("Testing visualization module...")

    # 生成测试点云
    test_points = np.random.randn(1000, 3) * 10
    test_colors = np.random.randint(0, 255, (1000, 3))

    # 测试matplotlib可视化
    visualize_3d_matplotlib(test_points, test_colors, title='Test Point Cloud')

    print("Visualization test completed!")