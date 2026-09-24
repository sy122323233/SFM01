# B / C / D 三位同学分工任务说明书

> 本文是给 **B（后端）、C（前端）、D（测试）** 三位同学的操作手册，按「分步照做」的程度写，能复制粘贴就复制粘贴。
> A（计算/算法层）已经把重建能力封装成统一接口，你们不用碰算法内部，只调 `reconstruct_full()` 即可。

---

## 0. 先花 5 分钟看：A 已经交出了什么

**一个函数**，跑完整条「稀疏→稠密→网格」链路：

```python
from sfm_project import reconstruct_full

result = reconstruct_full("images2", save=True)   # 图片目录 / 文件路径列表

result.mesh_glb       # ★ 前端 Three.js 直接加载的 GLB 路径
result.mesh_ply       # 网格 PLY
result.dense_ply      # 稠密点云 PLY
result.num_points     # 最终点云数量
result.to_dict()      # JSON 可序列化统计 + 产物路径
```

**关键约定**（详细字段见《接口使用说明与算法说明.md》）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `image_path` | — | 图片目录，或图片文件路径列表（环拍需按序） |
| `save` | `True` | 是否写盘到 `output/` |
| `do_dense` / `do_mesh` | `True` | 是否做稠密 / 网格 |
| `visualize` | `False` | **后端/服务器必须 False**，否则弹窗口卡死 |
| `dense_kwargs` | — | 调精度/速度，如 `dict(num_depths=96, max_dim=480)` |

**怎么自己先跑通验证**（在项目根目录 `SFM01/` 下）：

```bash
cd SFM01
python -c "from sfm_project import reconstruct_full; r=reconstruct_full('images2', save=True); print(r.mesh_glb)"
```

跑通后 `output/dense/mesh/mesh.glb` 就是前端要加载的模型。当前 10 张 Temple 实测：稀疏 4006 点 / 稠密 36.9 万点 / 网格 19999 面 / GLB 约 656KB。

---

## 1. 总分工与依赖关系

| 角色 | 负责 | 产出 |
|---|---|---|
| **A**（已完成） | 计算/算法层：SfM + 稠密 + 网格 | `reconstruct_full()` + 产物文件 |
| **B** | 后端/数据层：Web 服务、存储、任务队列、日志、鉴权 | FastAPI 服务 |
| **C** | 前端/展示层：上传、进度、Three.js 3D 展示 | 网页 |
| **D** | 测试/集成/部署：单测/集成/性能/部署、最终报告 | pytest 用例 + 文档 |

**依赖链**：A 已交付接口 → **B 先出 API** → C 联调 B → D 全程并行（边写边测）。

> ⏱ 建议顺序：B 和 C 可同时开工（C 先写死一个 GLB 本地预览，B 先把 API 跑起来），D 从第一天就开始写 A 模块的单元测试。

---

## 2. B 同学（后端 / 数据层）—— 分步照做

### 2.1 目标

把 A 的 `reconstruct_full()` 包成 Web 服务，提供「上传图片 → 触发重建 → 查进度 → 下载模型」的 API，并做好存储、后台任务、日志、鉴权。

### 2.2 技术选型 + 安装

```bash
pip install fastapi uvicorn
# 数据库用 Python 自带的 sqlite3，零额外安装；鉴权用 itsdangerous（轻量）
pip install itsdangerous
```

### 2.3 目录结构（在 SFM01/ 下新建）

```
backend/
├─ app.py          # FastAPI 入口 + 路由
├─ database.py     # SQLite 建表/CRUD
├─ tasks.py        # 后台重建任务（线程池）
├─ auth.py         # 简单 token 鉴权
└─ uploads/        # 上传的图片（加到 .gitignore）
```

### 2.4 数据库表（最小可用，三张表）

```
project: id, name, created_at          # 一次重建 = 一个项目
task:    id, project_id, status, error, result, created_at   # 重建任务状态
model:   id, project_id, file_path, kind                     # 产出的 GLB/PLY 路径
```

### 2.5 API 设计

```
GET  /api/health              健康检查（D 测试会打）
POST /api/upload              上传多图 → project_id
POST /api/reconstruct         触发重建 → task_id
GET  /api/task/{task_id}      查进度（C 轮询用）
GET  /api/model/{project_id}  下载模型（返回 GLB 文件流）
POST /api/register            注册（可选，先做简单版）
POST /api/login               登录 → token（可选）
```

### 2.6 后台任务（**关键**：绝不能在请求线程里同步重建）

重建很慢（数秒到数分钟），同步跑会超时。用线程池 + 任务表记录状态：

```python
executor = ThreadPoolExecutor(max_workers=1)   # GPU 只有 8GB，一次只跑一个
executor.submit(run_reconstruct, task_id, project_id, img_dir)  # 立即返回 task_id
```

### 2.7 完整代码骨架（可直接抄）

**`backend/database.py`**

```python
import os, sqlite3
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db")

def _conn():
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row; return c

def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS project (id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, created_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS task (id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER, status TEXT DEFAULT 'pending', error TEXT,
            result TEXT, created_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS model (id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER, file_path TEXT, kind TEXT);
        """)

def create_project(name):
    with _conn() as c:
        return c.execute("INSERT INTO project(name) VALUES (?)", (name,)).lastrowid

def create_task(project_id):
    with _conn() as c:
        return c.execute("INSERT INTO task(project_id) VALUES (?)", (project_id,)).lastrowid

def update_task(task_id, status, error=None, result=None):
    with _conn() as c:
        c.execute("UPDATE task SET status=?, error=?, result=? WHERE id=?",
                  (status, error, result, task_id))

def get_task(task_id):
    with _conn() as c:
        r = c.execute("SELECT * FROM task WHERE id=?", (task_id,)).fetchone()
        return dict(r) if r else None

def add_model(project_id, file_path, kind):
    with _conn() as c:
        c.execute("INSERT INTO model(project_id, file_path, kind) VALUES (?,?,?)",
                  (project_id, file_path, kind))

def list_models(project_id):
    with _conn() as c:
        return [r["file_path"] for r in
                c.execute("SELECT file_path FROM model WHERE project_id=? ORDER BY id DESC",
                          (project_id,)).fetchall()]
```

**`backend/tasks.py`**

```python
import os, sys
from concurrent.futures import ThreadPoolExecutor

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)          # 让 backend 能 import sfm_project

from database import update_task, add_model
from sfm_project import reconstruct_full

executor = ThreadPoolExecutor(max_workers=1)  # 显存有限，串行

def run_reconstruct(task_id, project_id, img_dir):
    update_task(task_id, "running")
    try:
        result = reconstruct_full(img_dir, save=True, visualize=False)
        if result.mesh_glb: add_model(project_id, result.mesh_glb, "glb")
        if result.mesh_ply: add_model(project_id, result.mesh_ply, "ply")
        if result.dense_ply: add_model(project_id, result.dense_ply, "ply")
        update_task(task_id, "success",
                    result=str(result.to_dict()))
    except Exception as e:
        update_task(task_id, "failed", error=str(e))
```

**`backend/app.py`**

```python
import os, sys
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from database import init_db, create_project, create_task, get_task, list_models
from tasks import executor, run_reconstruct

app = FastAPI(title="三维重建后端")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])  # C 前端跨端口要用

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
init_db()

@app.get("/api/health")
def health(): return {"status": "ok"}

@app.post("/api/upload")
async def upload(project_name: str = Form("default"), files: list[UploadFile] = File(...)):
    project_id = create_project(project_name)
    img_dir = os.path.join(UPLOAD_DIR, str(project_id))
    os.makedirs(img_dir, exist_ok=True)
    for f in files:
        with open(os.path.join(img_dir, f.filename), "wb") as out:
            out.write(await f.read())
    return {"project_id": project_id, "images": len(files)}

@app.post("/api/reconstruct")
def reconstruct_api(project_id: int = Form(...)):
    img_dir = os.path.join(UPLOAD_DIR, str(project_id))
    if not os.path.isdir(img_dir):
        raise HTTPException(404, "project not found")
    task_id = create_task(project_id)
    executor.submit(run_reconstruct, task_id, project_id, img_dir)
    return {"task_id": task_id}

@app.get("/api/task/{task_id}")
def task_status(task_id: int):
    t = get_task(task_id)
    if t is None: raise HTTPException(404, "task not found")
    return t

@app.get("/api/model/{project_id}")
def download_model(project_id: int):
    models = list_models(project_id)
    if not models: raise HTTPException(404, "no model yet")
    return FileResponse(models[0])   # 优先返回最新 GLB
```

启动：`uvicorn backend.app:app --reload --port 8000`（在 SFM01 目录下）。

### 2.8 日志、鉴权、CORS

- **日志**：`import logging; logging.basicConfig(filename="backend.log", level=logging.INFO)`，每个接口、每个任务的开始/结束/耗时/报错都记下来（答辩讲「可观测性」用）。
- **鉴权**：登录发 token（`itsdangerous` 简单签名即可），`/api/upload`、`/api/reconstruct` 校验。先做「注册→登录→token」的最小闭环，够演示。
- **CORS**：上面的 `CORSMiddleware` 已开，前端跨端口调用才不报错。

### 2.9 验收标准

- [ ] `GET /api/health` 返回 ok
- [ ] 上传多图 → 拿到 project_id → 触发重建 → 拿到 task_id
- [ ] 轮询 task 直到 success，`/api/model/{id}` 能下载 GLB
- [ ] 重建期间服务不卡死（后台线程生效）

---

## 3. C 同学（前端 / 展示层）—— 分步照做

### 3.1 目标

网页：上传图片 → 看重建进度 → Three.js 在浏览器里 3D 查看模型。

### 3.2 技术选型（不 npm、不打包，单文件搞定）

HTML + 原生 JS + Three.js（CDN importmap 引入）。

### 3.3 页面组成（一个页面三个区）

1. **上传区**：多图选择 → `POST /api/upload` 拿 project_id → `POST /api/reconstruct` 拿 task_id
2. **进度区**：`setInterval` 轮询 `GET /api/task/{id}`，显示 待处理/重建中/成功/失败
3. **展示区**：成功后用 GLTFLoader 加载 `GET /api/model/{project_id}` 返回的 GLB

### 3.4 完整代码（`frontend/index.html`）

```html
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>三维重建</title>
<style>body{font-family:sans-serif;margin:0}#status{margin:8px}canvas{display:block}</style></head>
<body>
  <div id="status">选择图片开始</div>
  <input type="file" id="files" multiple accept="image/*">
  <button id="btn">上传并重建</button>

  <script type="importmap">
  { "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" } }
  </script>

  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

    const API = 'http://127.0.0.1:8000';   // 后端地址（可改）
    const status = document.getElementById('status');

    // Three.js 场景
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 3);
    const renderer = new THREE.WebGLRenderer({antialias:true});
    renderer.setSize(innerWidth, innerHeight);
    document.body.appendChild(renderer.domElement);
    scene.add(new THREE.AmbientLight(0xffffff, 1.0));
    scene.add(new THREE.DirectionalLight(0xffffff, 1.2));
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    (function animate(){requestAnimationFrame(animate); controls.update(); renderer.render(scene,camera);})();

    function loadModel(projectId){
      const loader = new GLTFLoader();
      loader.load(`${API}/api/model/${projectId}`, (gltf)=>{
        const box = new THREE.Box3().setFromObject(gltf.scene);
        const c = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3()).length();
        gltf.scene.position.sub(c);
        camera.position.set(c.x, c.y, c.z + size * 1.5);
        controls.target.copy(c);
        scene.add(gltf.scene);
        status.textContent = '重建完成 ✅';
      });
    }

    function poll(taskId, projectId){
      status.textContent = '重建中...';
      const t = setInterval(async ()=>{
        const r = await (await fetch(`${API}/api/task/${taskId}`)).json();
        if (r.status === 'success'){ clearInterval(t); loadModel(projectId); }
        else if (r.status === 'failed'){ clearInterval(t); status.textContent = '失败：' + r.error; }
      }, 2000);
    }

    document.getElementById('btn').onclick = async ()=>{
      const files = document.getElementById('files').files;
      if (!files.length) return;
      const fd = new FormData();
      for (const f of files) fd.append('files', f);
      const up = await (await fetch(`${API}/api/upload`, {method:'POST', body:fd})).json();
      const rec = await (await fetch(`${API}/api/reconstruct`,
          {method:'POST', body:new FormData(Object.entries({project_id: up.project_id}))})).json();
      poll(rec.task_id, up.project_id);
    };
  </script>
</body>
</html>
```

### 3.5 注意的坑

- **模型格式**：直接加载 A 导出的 **GLB**，别自己从 PLY 转。
- **跨域**：前端页面和后端不同端口，后端已开 CORS（B 那边）。
- **大文件**：模型几十 MB，前端加个「加载中」提示；后端用流式下载。
- **本地预览**：B 还没好时，可先硬编码一个 `loader.load('output/dense/mesh/mesh.glb')` 做静态预览，先把 Three.js 跑通。

### 3.6 验收标准

- [ ] 上传多图能触发后端重建
- [ ] 进度轮询实时更新状态
- [ ] 重建完成后浏览器能 360° 旋转查看 GLB 模型

---

## 4. D 同学（测试 / 集成 / 部署）—— 分步照做

### 4.1 目标

写测试（单元/集成/性能/模型质量），输出测试文档 + 缺陷分析，最后汇总报告。任务书明确要求 **TDD**。

### 4.2 单元测试（pytest）

```bash
pip install pytest
```

`tests/conftest.py`（让测试能 import 到 sfm_project）：

```python
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # SFM01
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sfm_project   # 触发其内部 sys.path 处理
```

`tests/test_calibration.py`：

```python
from core.calibration import build_intrinsics, read_exif_focal

def test_build_intrinsics_center():
    K = build_intrinsics(2800, 2832, 2128)
    assert K[0,0] == 2800 and K[0,2] == 1416 and K[1,2] == 1064

def test_read_exif_none_when_no_exif():
    assert read_exif_focal('images2/00001.jpg', 2832) is None
```

`tests/test_meshing.py`：

```python
import numpy as np
from core.meshing import points_to_mesh

def test_points_to_mesh_no_write(tmp_path):
    pts = np.random.rand(500, 3)
    mesh = points_to_mesh(pts, out_dir=str(tmp_path), write_files=False)
    assert mesh is not None
    assert len(mesh.vertices) > 0
```

### 4.3 集成测试（全链路）

`tests/test_integration.py`：

```python
from sfm_project import reconstruct, reconstruct_full

def test_reconstruct_sparse():
    r = reconstruct("images2", save=False, visualize=False)
    assert r.num_images == 10            # 10 张 Temple 全部注册
    assert r.reprojection_error < 0.5    # 重投影误差合格
    assert r.points.shape[1] == 3        # 点云是 (N,3)

def test_reconstruct_full(tmp_path):      # 慢，可打 @pytest.mark.slow
    full = reconstruct_full("images2", save=False,
                            dense_kwargs=dict(num_depths=64, max_dim=320, num_ref_views=4))
    assert full.dense is not None
    assert full.dense.num_points > full.sparse.num_points
    assert full.mesh is not None
```

> 项目里已有 `sfm_project/check_api.py` 可参考。跑的时候若控制台报 `UnicodeEncodeError`，加 `PYTHONIOENCODING=utf-8`。

### 4.4 性能测试与模型质量测试

- **性能**：`result.timings` 里已有各阶段耗时（特征提取/匹配/BA/总计）。整理成表，对比 `BA.use_cuda` 开/关、`num_depths` 大小。
- **模型质量**：重投影误差（目标 <0.5px，实测 0.2241px）、点云数量、网格面数/是否闭合/有无洞。

### 4.5 缺陷分析（TDD 闭环）

把测试发现的 bug 记录成表：**编号 / 描述 / 复现步骤 / 预期 / 实际 / 状态**，体现「测出问题→改→复测」的迭代。

### 4.6 部署

写一个 Dockerfile 或一键启动脚本，把后端 + 前端 + 重建环境打包。**版本务必写对**（本机 py3.14：open3d==0.20.0、torch 2.10+cu130）：

```dockerfile
# 简化示意，D 同学按实际调
FROM python:3.14-slim
RUN pip install opencv-python open3d==0.20.0 scipy fastapi uvicorn
RUN pip install torch --index-url https://download.pytorch.org/whl/cu130
COPY . /app
WORKDIR /app
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.7 验收标准

- [ ] 单元测试 + 集成测试全部通过
- [ ] 有一份测试报告（通过率、缺陷表、性能对比）
- [ ] 能一键启动（脚本或 Docker）

---

## 5. 硬件与网络（香橙派 + 多摄像头 + TCP）—— 最大缺口

这是任务书相对上学期**最大的新增**，也最容易漏。不属于常规 Web 部分，建议这样安排：

1. **采集端（香橙派 + 多摄像头）**：多路摄像头同步采集，TCP 把图片传给上位机 PC。
2. **上位机（PC）**：接收图片 → 调 A 的 `reconstruct_full()` → 展示。
3. **TCP 通讯**：香橙派做客户端、PC 做服务端，传图片字节流 + 控制指令（开始/停止）。

> 建议分工：**B 负责 TCP 上下位机协议 + 数据层**（正好是后端/通讯），**A 配合做多摄像头图像对齐/融合**。

> **降级方案（务必写进文档）**：硬件买不到/来不及，就写「用本地多目录/多组图片模拟多摄像头采集」，明确写清「硬件用模拟器替代」，答辩有交代。现在跑通的就是单摄像头环拍。

---

## 6. 关键环境约束（三人都别踩）

- 本机 **Python 3.14、无 conda**；`open3d` 只能 0.20.0、`torch` 2.10+cu130，**GPU RTX 5060 8GB**。
- `nerfstudio` / `instant-ngp` / `Fast3R` 都装不上或显存不够，**主线别依赖**；NeRF 只作为「可选进阶模块」在文档里提一句。
- 重建是 GPU 密集，8GB 显存别开太大批量；后端后台任务 `max_workers=1` 就是这个原因。
- 项目根目录 `SFM01/`，`import sfm_project` 前要保证它在 `sys.path`（backend/tasks.py 里已经处理）。

---

## 7. 建议时间安排

| 阶段 | B | C | D |
|---|---|---|---|
| 第 1-2 天 | 搭 FastAPI + 数据库 + 健康检查 | 静态 GLB 预览跑通 Three.js | 写 calibration/meshing 单测 |
| 第 3-4 天 | 上传/重建/进度/下载 API + 后台任务 | 对接上传 + 进度轮询 | 集成测试（全链路） |
| 第 5-6 天 | 日志/鉴权/CORS + TCP 协议 | 模型加载 + 交互打磨 | 性能/模型质量测试 |
| 第 7 天 | 联调 | 联调 | 部署 + 汇总报告 |
