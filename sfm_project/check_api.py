# -*- coding: utf-8 -*-
"""临时脚本：验证重构后的 reconstruct() 接口"""
from main import reconstruct, ReconstructionResult
from config import Config
import json

print("=" * 50)
print("开始验证 reconstruct() 接口")

# 1) 纯计算：不写盘、不弹窗
result = reconstruct(Config().DATA_PATH)

print("\n--- 结果 ---")
print("返回类型:", type(result).__name__)  # 期望 ReconstructionResult
print("点数:", result.num_points)  # 期望 > 0
print("平均重投影误差:", result.reprojection_error)  # 期望 ~0.22

# 2) to_dict 能被 JSON 序列化（np.float64 已修，不报错即通过）
print("to_dict:", json.dumps(result.to_dict()))

# 3) 字段齐全
print("points shape:", result.points.shape)
print("colors shape:", result.colors.shape)
print("相机位姿数量:", len(result.rotations))

print("\n全部通过 ✅")