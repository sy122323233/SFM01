#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时预览脚本：把 mesh.ply 渲染成 PNG（无 GUI，Agg 后端）。"""
import os
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ply = os.path.join(_ROOT, "output", "mesh", "mesh.ply")
out = os.path.join(_ROOT, "output", "mesh", "preview.png")

mesh = o3d.io.read_triangle_mesh(ply)
verts = np.asarray(mesh.vertices)
tris = np.asarray(mesh.triangles)
colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")

polys = verts[tris]
fc = colors[tris].mean(axis=1) if colors is not None else None
coll = Poly3DCollection(polys, facecolors=fc, edgecolors="none")
ax.add_collection3d(coll)

mn, mx = verts.min(axis=0), verts.max(axis=0)
c = (mn + mx) / 2
r = float((mx - mn).max()) / 2 * 1.05
ax.set_xlim(c[0] - r, c[0] + r)
ax.set_ylim(c[1] - r, c[1] + r)
ax.set_zlim(c[2] - r, c[2] + r)
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=20, azim=-60)
ax.axis("off")

plt.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
print(f"saved {out}  ({len(verts)} verts / {len(tris)} faces)")
