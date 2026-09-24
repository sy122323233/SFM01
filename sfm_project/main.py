#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFM主程序 - 模块化版本
"""

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import cv2
from config import Config
from core.feature_extractor import FeatureExtractor
from core.feature_matcher import FeatureMatcher
from core.reconstruction import IncrementalReconstructor
from core.bundle_adjustment import BundleAdjustment
from core.calibration import auto_intrinsics
from utils.io_utils import save_results, save_performance_report, save_point_cloud_ply
from utils.visualization import visualize_3d_matplotlib, visualize_error_distribution


@dataclass
class ReconstructionResult:
    """三维重建的统一结果结构（对外 API 返回该对象）。

    字段:
        points: (N, 3) 三维点坐标
        colors: (N, 3) 颜色，RGB，0-255
        reprojection_error: 平均重投影误差（像素）
        num_points: 点数量
        num_images: 参与重建的图像数量
        timings: 各阶段耗时（秒）
        rotations / motions: 每帧相机位姿，供后续密集重建/融合使用
    """
    points: np.ndarray
    colors: np.ndarray
    reprojection_error: float
    num_points: int
    num_images: int
    timings: dict
    rotations: list
    motions: list
    registered_indices: list = None  # 每帧位姿对应的原始图像索引（跳过坏帧后与图像全集不再一致）

    def to_dict(self):
        """转为 JSON 可序列化的字典（只含标量统计，不含大数组）。"""
        return {
            'num_points': int(self.num_points),
            'num_images': int(self.num_images),
            'reprojection_error': float(self.reprojection_error),
            'timings': {k: float(v) for k, v in self.timings.items()},
            'total_time': float(sum(self.timings.values())),
        }


class SFMPipeline:
    """SFM完整流程"""
    def __init__(self, config):
        self.config = config
        self.feature_extractor = FeatureExtractor(**config.FEATURE)   # 特征提取器
        self.feature_matcher = FeatureMatcher(config.MATCHING['norm_type'])    # 特征匹配器
        self.reconstructor = IncrementalReconstructor(config.K)     # 增量重建器
        self.ba_optimizer = BundleAdjustment(config.K)                # Bundle Adjustment 优化器

        # 计时
        self.timings = {}

    def run(self, image_path, save=False, visualize=False):
        """运行完整 SFM 流程，返回 ReconstructionResult。

        Args:
            image_path: 图像目录路径，或图像文件路径列表。
            save: 是否把点云/统计写入 config.OUTPUT_PATH。
            visualize: 是否弹出 matplotlib 可视化窗口（服务端/无头环境应设为 False）。
        """

        # 1. 加载图像
        print("=" * 50)
        print("Step 1: Loading images...")
        image_names = self._load_images(image_path)

        # 2. 特征提取
        print("\nStep 2: Extracting features...")
        t1 = time.time()
        key_points, descriptors, colors_list, images, kept_idx = \
            self.feature_extractor.extract_from_images(image_names)
        self.timings['feature_extraction'] = time.time() - t1
        print(f"  Extracted features from {len(key_points)} images")
        print(f"  Time: {self.timings['feature_extraction']:.2f}秒")

        if len(key_points) < 2:
            raise ValueError("Not enough images with valid features (need >= 2)")

        # 3. 特征匹配
        print("\nStep 3: Matching features...")
        t2 = time.time()
        matches = self.feature_matcher.match_sequential(
            descriptors, self.config.MATCHING['ratio']
        )
        self.timings['feature_matching'] = time.time() - t2
        print(f"  Generated {len(matches)} match pairs")
        print(f"  Time: {self.timings['feature_matching']:.2f}秒")

        # 3.5 自动确定内参 K（EXIF 焦距 → 默认值）
        if getattr(self.config, 'AUTO_CALIBRATE', True):
            print("\nStep 3.5: Auto-calibrating intrinsics K...")
            H_img, W_img = images[0].shape[:2]
            new_K, source = auto_intrinsics(image_names[kept_idx[0]], W_img, H_img)
            old_K = self.config.K
            self.config.K = new_K
            # 用新内参重建增量重建器与 BA 优化器
            self.reconstructor = IncrementalReconstructor(self.config.K)
            self.ba_optimizer = BundleAdjustment(self.config.K)
            print(f"  K 来源: {source}")
            print(f"  旧 K: fx={old_K[0, 0]:.1f} fy={old_K[1, 1]:.1f}")
            print(f"  新 K: fx={new_K[0, 0]:.1f} fy={new_K[1, 1]:.1f}")

        # 4. 初始化重建
        print("\nStep 4: Initial reconstruction...")
        if len(matches[0]) < 8:
            raise RuntimeError(
                f"前两帧匹配点不足 ({len(matches[0])} < 8)，无法初始化重建。"
                f"请检查数据集前两张图是否重叠。")
        t3 = time.time()
        structure = self.reconstructor.initialize(
            key_points[0], key_points[1], matches[0]
        )
        self.timings['initialization'] = time.time() - t3
        print(f"  Initial points: {len(structure)}")
        print(f"  Time: {self.timings['initialization']:.2f}秒")

        # 5. 增量重建（惰性匹配：跳过坏帧时始终与「最近成功帧」匹配，避免错位）
        print("\nStep 5: Incremental reconstruction...")
        t4 = time.time()
        points_before = len(structure)

        registered = [0, 1]   # 已成功注册的图像在 key_points 里的位置索引
        last_good = 1         # 最近一次成功注册的帧

        for i in range(2, len(key_points)):
            if last_good == i - 1:
                # 没有跳过坏帧：直接复用步骤3预计算的相邻帧匹配，避免重复匹配
                m = matches[i - 1]
            else:
                # 前面跳过过坏帧：当前帧与「最近成功帧」重新匹配
                m = self.feature_matcher._match_pair(
                    descriptors[last_good], descriptors[i],
                    self.config.MATCHING['ratio']
                )
                print(f"  Re-matching frame {last_good + 1} <-> {i + 1}: {len(m)} matches")
            try:
                success = self.reconstructor.add_frame(
                    key_points[last_good], key_points[i],
                    descriptors[last_good], descriptors[i], m
                )
            except Exception as e:
                print(f"  Error processing frame {i + 1}: {e}")
                success = False

            if success:
                last_good = i
                registered.append(i)
            else:
                print(f"  Skipped frame {i + 1}（坏帧，已跳过，不影响后续帧）")

        self.timings['incremental'] = time.time() - t4
        points_after = len(self.reconstructor.structure)
        print(f"  Points before: {points_before}, after: {points_after}")
        print(f"  成功注册 {len(registered)}/{len(key_points)} 帧")
        print(f"  Time: {self.timings['incremental']:.2f}秒")

        # 把「成功注册」的帧抽成与 rotations/motions/correspondence 同序的子集，
        # 跳过坏帧后「图像全集」与「已注册位姿」不再一一对应，后续步骤必须用子集。
        registered_key_points = [key_points[i] for i in registered]
        registered_colors = [colors_list[i] for i in registered]
        # 换算成「原始图像文件」索引，供稠密重建按帧对齐
        registered_image_idx = [kept_idx[i] for i in registered]

        # 6. 为每个3D点分配颜色（取所有观测的平均颜色）
        print("\nStep 6: Assigning colors to 3D points...")
        colors_3d = self._assign_colors_to_points(
            self.reconstructor.structure,
            self.reconstructor.correspondence,
            registered_colors,
            registered_key_points
        )

        # 7. BA 优化
        print("\nStep 7: Bundle Adjustment...")
        t5 = time.time()

        if len(self.reconstructor.structure) > 0:

            if self.config.BA.get('use_cuda', False):
                print(" 使用 CUDA 版本 BA...")
                structure_optimized = self.ba_optimizer.optimize_cuda(
                    self.reconstructor.structure,
                    self.reconstructor.rotations,
                    self.reconstructor.motions,
                    registered_key_points,
                    self.reconstructor.correspondence
                )
            else:
                print(" 使用 CPU 版本 BA...")
                structure_optimized = self.ba_optimizer.optimize(
                    self.reconstructor.structure,
                    self.reconstructor.rotations,
                    self.reconstructor.motions,
                    registered_key_points,
                    self.reconstructor.correspondence
                )

            # 记录 BA 耗时
            self.timings['bundle_adjustment'] = time.time() - t5

            # 重要：BA 后需要重新分配颜色，因为点的数量变了
            self.reconstructor.structure = structure_optimized

            print("\nStep 7.5: Re-assigning colors after filtering...")
            colors_3d = self._assign_colors_to_points(
                self.reconstructor.structure,
                self.reconstructor.correspondence,
                registered_colors,
                registered_key_points
            )
        else:
            self.timings['bundle_adjustment'] = 0
        # 8. 清理无效点
        print("\nStep 8: Cleaning points...")
        points_before_clean = len(self.reconstructor.structure)
        self.reconstructor.structure, colors_3d = self._clean_points(
            self.reconstructor.structure, colors_3d
        )
        points_after_clean = len(self.reconstructor.structure)
        print(f"  Removed {points_before_clean - points_after_clean} invalid points")

        # 9. 计算重投影误差
        print("\nStep 9: Computing reprojection error...")
        reproj_error, all_errors = self._compute_reprojection_error(
            self.reconstructor.structure,
            self.reconstructor.rotations,
            self.reconstructor.motions,
            registered_key_points,
            self.reconstructor.correspondence
        )
        print(f"  Average reprojection error: {reproj_error:.4f} pixels")

        # 10. 输出统计信息
        self._print_statistics(reproj_error)

        # 11. 保存结果（可选）
        if save:
            print("\nStep 10: Saving results...")
            os.makedirs(self.config.OUTPUT_PATH, exist_ok=True)

            # 保存点云
            save_results(self.reconstructor.structure, colors_3d, self.config.OUTPUT_PATH)

            # 额外保存一个 MeshLab 可直接打开的彩色 PLY 点云
            save_point_cloud_ply(
                os.path.join(self.config.OUTPUT_PATH, "sparse_cloud_meshlab.ply"),
                self.reconstructor.structure,
                colors_3d
            )

            # 保存性能报告
            save_performance_report(
                os.path.join(self.config.OUTPUT_PATH, 'performance.json'),
                self.timings, reproj_error,
                points_before_clean, points_after_clean
            )

        # 12. 可视化（可选）
        if visualize:
            print("\nStep 11: Visualizing...")
            if len(self.reconstructor.structure) > 0:
                visualize_3d_matplotlib(self.reconstructor.structure, colors_3d,
                                        f'3D Reconstruction ({len(self.reconstructor.structure)} points)')

                # 如果有误差数据，也可视化误差分布
                if len(all_errors) > 0:
                    visualize_error_distribution(all_errors)
            else:
                print("  No points to visualize!")

        # 13. 返回结构化结果
        return ReconstructionResult(
            points=self.reconstructor.structure,
            colors=colors_3d,
            reprojection_error=reproj_error,
            num_points=len(self.reconstructor.structure),
            num_images=len(registered),
            timings=dict(self.timings),
            rotations=self.reconstructor.rotations,
            motions=self.reconstructor.motions,
            registered_indices=registered_image_idx,
        )

    def _assign_colors_to_points(self, structure, correspondences, colors_list, key_points_list):
        """为每个3D点分配颜色（取所有观测的平均颜色）"""
        # 初始化颜色累加器
        color_sum = np.zeros((len(structure), 3))
        color_count = np.zeros(len(structure))

        # 遍历所有帧
        for i in range(len(correspondences)):
            if i >= len(colors_list) or i >= len(key_points_list):
                continue

            corr = correspondences[i]
            colors = colors_list[i]

            for j in range(min(len(corr), len(colors))):
                point3d_id = int(corr[j])
                if point3d_id >= 0 and point3d_id < len(structure):
                    color_sum[point3d_id] += colors[j]
                    color_count[point3d_id] += 1

        # 计算平均颜色
        colors_3d = np.zeros((len(structure), 3))
        for i in range(len(structure)):
            if color_count[i] > 0:
                colors_3d[i] = color_sum[i] / color_count[i]
            else:
                colors_3d[i] = [128, 128, 128]  # 默认灰色

        return colors_3d

    def _load_images(self, image_path):
        """加载图像文件列表。支持目录路径，或图像文件路径列表。"""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

        # 传入的是文件路径列表：保持给定顺序（顺序匹配依赖它）
        if isinstance(image_path, (list, tuple)):
            image_names = [p for p in image_path
                           if p.lower().endswith(valid_extensions)]
            if len(image_names) == 0:
                raise ValueError("No valid image files in the given list")
            return image_names

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path not found: {image_path}")

        names = sorted([f for f in os.listdir(image_path)
                        if f.lower().endswith(valid_extensions)])

        if len(names) == 0:
            raise ValueError(f"No image files found in {image_path}")

        print(f"  找到 {len(names)} 张图像")
        return [os.path.join(image_path, name) for name in names]

    def _clean_points(self, structure, colors_3d):
        """清理无效点"""
        valid_indices = []
        for i in range(len(structure)):
            if not np.isnan(structure[i][0]) and not np.isinf(structure[i][0]):
                valid_indices.append(i)

        structure = structure[valid_indices]
        if colors_3d is not None and len(colors_3d) > 0:
            colors_3d = colors_3d[valid_indices]

        return structure, colors_3d

    def _compute_reprojection_error(self, structure, rotations, motions,
                                    key_points_list, correspondences):
        """计算重投影误差"""
        total_error = 0
        total_points = 0
        all_errors = []

        for i in range(len(rotations)):
            if i >= len(key_points_list):
                continue

            r_vec, _ = cv2.Rodrigues(rotations[i])
            t_vec = motions[i]

            if i < len(correspondences):
                point3d_ids = correspondences[i]
            else:
                continue

            key_points = key_points_list[i]

            for j in range(min(len(point3d_ids), len(key_points))):
                point3d_id = int(point3d_ids[j])
                if point3d_id < 0 or point3d_id >= len(structure):
                    continue

                point3d = structure[point3d_id]
                point2d_obs = key_points[j].pt

                point2d_proj, _ = cv2.projectPoints(
                    point3d.reshape(1, 1, 3),
                    r_vec, t_vec, self.config.K, np.array([])
                )
                point2d_proj = point2d_proj.reshape(2)

                error = np.linalg.norm(point2d_obs - point2d_proj)
                all_errors.append(error)
                total_error += error
                total_points += 1

        if total_points > 0:
            return total_error / total_points, all_errors
        else:
            return float('inf'), []

    def _print_statistics(self, reproj_error):
        """打印统计信息"""
        print("\n" + "=" * 50)
        print("PERFORMANCE STATISTICS")
        print("=" * 50)
        print(f"Feature extraction:  {self.timings.get('feature_extraction', 0):.2f}s")
        print(f"Feature matching:    {self.timings.get('feature_matching', 0):.2f}s")
        print(f"Initialization:      {self.timings.get('initialization', 0):.2f}s")
        print(f"Incremental:         {self.timings.get('incremental', 0):.2f}s")
        print(f"Bundle adjustment:   {self.timings.get('bundle_adjustment', 0):.2f}s")
        print(f"Total time:          {sum(self.timings.values()):.2f}s")
        print(
            f"Final points:        {len(self.reconstructor.structure) if self.reconstructor.structure is not None else 0}")
        print(f"Reprojection error:  {reproj_error:.4f} pixels")
        print("=" * 50)


def reconstruct(image_path, config=None, save=False, visualize=False):
    """对外统一入口：给定图像（目录或文件路径列表），运行完整 SfM 并返回结果。

    后端 / 前端 / 测试都通过这个函数调用，不需要接触内部 SFMPipeline 细节。

    Args:
        image_path: 图像目录路径，或图像文件路径列表。
        config: 可选，自定义 Config；默认新建一个 Config()。
        save: 是否把点云/统计写入 config.OUTPUT_PATH。
        visualize: 是否弹出 matplotlib 可视化窗口（无头环境设为 False）。

    Returns:
        ReconstructionResult
    """
    if config is None:
        config = Config()
    pipeline = SFMPipeline(config)
    return pipeline.run(image_path, save=save, visualize=visualize)


def main():
    """主函数：命令行演示入口（保存结果 + 可视化）。"""
    try:
        config = Config()

        if not os.path.exists(config.DATA_PATH):
            print(f"Error: Dataset path not found: {config.DATA_PATH}")
            print("Please check the DATA_PATH in config.py")
            return

        result = reconstruct(config.DATA_PATH, config=config,
                             save=True, visualize=True)

        print("\n" + "=" * 50)
        print("SFM completed successfully!")
        print(f"  Points: {result.num_points}")
        print(f"  Reprojection error: {result.reprojection_error:.4f} px")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()