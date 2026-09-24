"""SFM 三维重建包：对外暴露 reconstruct()（稀疏）与 reconstruct_full()（全链路）统一入口。

用法（在 SFM01 目录下，或把 SFM01 加入 sys.path 后）：
    from sfm_project import reconstruct, reconstruct_full
    result = reconstruct("images2", save=True)           # 只做稀疏 SfM
    full = reconstruct_full("images2", save=True)        # 稀疏 → 稠密 → 网格 一条龙
    print(full.mesh_glb)   # 前端 Three.js 直接加载的 GLB 路径

注：包内模块沿用「顶层导入」（如 `from config import Config`），
为保证 `import sfm_project` 也能解析，这里把包目录临时加入 sys.path。
等后端真正接入时，可再改成标准相对导入（`from .config import ...`）。
"""

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from main import reconstruct, SFMPipeline, ReconstructionResult  # noqa: E402
from full_pipeline import reconstruct_full, FullResult  # noqa: E402

__all__ = ['reconstruct', 'SFMPipeline', 'ReconstructionResult',
           'reconstruct_full', 'FullResult']
