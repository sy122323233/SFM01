#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFM主程序 - 模块化版本
"""

import os
import sys
import time
import numpy as np
import cv2
from config import Config
from core.feature_extractor import FeatureExtractor
from core.feature_matcher import FeatureMatcher
from core.reconstruction import IncrementalReconstructor
from core.bundle_adjustment import BundleAdjustment
from utils.io_utils import save_results, save_performance_report, save_point_cloud_ply
from utils.visualization import visualize_3d_matplotlib, visualize_error_distribution


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

    def run(self, image_path):
        """运行完整SFM流程"""

        # 1. 加载图像
        print("=" * 50)
        print("Step 1: Loading images...")
        image_names = self._load_images(image_path)

        # 2. 特征提取
        print("\nStep 2: Extracting features...")
        t1 = time.time()
        key_points, descriptors, colors_list, images = \
            self.feature_extractor.extract_from_images(image_names)
        self.timings['feature_extraction'] = time.time() - t1
        print(f"  Extracted features from {len(key_points)} images")
        print(f"  Time: {self.timings['feature_extraction']:.2f}秒")

        if len(key_points) < 2:
            print("Error: Not enough images with valid features!")
            return None, None

        # 3. 特征匹配
        print("\nStep 3: Matching features...")
        t2 = time.time()
        matches = self.feature_matcher.match_sequential(
            descriptors, self.config.MATCHING['ratio']
        )
        self.timings['feature_matching'] = time.time() - t2
        print(f"  Generated {len(matches)} match pairs")
        print(f"  Time: {self.timings['feature_matching']:.2f}秒")

        # 4. 初始化重建
        print("\nStep 4: Initial reconstruction...")
        t3 = time.time()
        structure = self.reconstructor.initialize(
            key_points[0], key_points[1], matches[0]
        )
        self.timings['initialization'] = time.time() - t3
        print(f"  Initial points: {len(structure)}")
        print(f"  Time: {self.timings['initialization']:.2f}秒")

        # 5. 增量重建
        print("\nStep 5: Incremental reconstruction...")
        t4 = time.time()
        points_before = len(structure)

        for i in range(1, len(matches)):
            print(f"  Processing frame {i + 1}/{len(matches)}...")
            try:
                success = self.reconstructor.add_frame(
                    key_points[i], key_points[i + 1],
                    descriptors[i], descriptors[i + 1],
                    matches[i]
                )
                if not success:
                    print(f"  Warning: Failed to add frame {i + 1}")
                    continue
            except Exception as e:
                print(f"  Error processing frame {i + 1}: {e}")
                continue

        self.timings['incremental'] = time.time() - t4
        points_after = len(self.reconstructor.structure)
        print(f"  Points before: {points_before}, after: {points_after}")
        print(f"  Time: {self.timings['incremental']:.2f}秒")

        # 6. 为每个3D点分配颜色（取所有观测的平均颜色）
        print("\nStep 6: Assigning colors to 3D points...")
        colors_3d = self._assign_colors_to_points(
            self.reconstructor.structure,
            self.reconstructor.correspondence,
            colors_list,
            key_points
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
                    key_points,
                    self.reconstructor.correspondence
                )
            else:
                print(" 使用 CPU 版本 BA...")
                structure_optimized = self.ba_optimizer.optimize(
                    self.reconstructor.structure,
                    self.reconstructor.rotations,
                    self.reconstructor.motions,
                    key_points,
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
                colors_list,
                key_points
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
            key_points,
            self.reconstructor.correspondence
        )
        print(f"  Average reprojection error: {reproj_error:.4f} pixels")

        # 10. 输出统计信息
        self._print_statistics(reproj_error)

        # 11. 保存结果
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
        # 12. 可视化
        print("\nStep 11: Visualizing...")
        if len(self.reconstructor.structure) > 0:
            visualize_3d_matplotlib(self.reconstructor.structure, colors_3d,
                                    f'3D Reconstruction ({len(self.reconstructor.structure)} points)')

            # 如果有误差数据，也可视化误差分布
            if len(all_errors) > 0:
                visualize_error_distribution(all_errors)
        else:
            print("  No points to visualize!")

        return self.reconstructor.structure, colors_3d

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
        """加载图像文件列表"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path not found: {image_path}")

        image_names = os.listdir(image_path)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_names = sorted([f for f in image_names
                              if f.lower().endswith(valid_extensions)])

        if len(image_names) == 0:
            raise ValueError(f"No image files found in {image_path}")

        print(f"  找到 {len(image_names)} 张图像")
        return [os.path.join(image_path, name) for name in image_names]

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


def main():
    """主函数"""
    try:
        # 创建配置
        config = Config()

        # 检查数据集路径
        if not os.path.exists(config.DATA_PATH):
            print(f"Error: Dataset path not found: {config.DATA_PATH}")
            print("Please check the DATA_PATH in config.py")
            return

        # 创建并运行SFM流程（正常运行，不加载已有结果）
        pipeline = SFMPipeline(config)
        structure, colors = pipeline.run(config.DATA_PATH)

        print("\n" + "=" * 50)
        print("SFM completed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()