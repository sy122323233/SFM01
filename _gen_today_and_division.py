# -*- coding: utf-8 -*-
"""生成《今日工作记录 与 BCD 分工任务说明》Word 文档（2026-09-23）"""
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SONG = u'宋体'
BLACK = RGBColor(0, 0, 0)
CODE = 'Consolas'

doc = Document()

# ---------- 全局样式 ----------
normal = doc.styles['Normal']
normal.font.name = 'Times New Roman'
normal.font.size = Pt(10.5)
normal.font.color.rgb = BLACK
normal._element.rPr.rFonts.set(qn('w:eastAsia'), SONG)

for level, size in [(1, 15), (2, 12.5), (3, 11), (4, 10.5)]:
    st = doc.styles[f'Heading {level}']
    st.font.name = SONG
    st.font.color.rgb = BLACK
    st._element.rPr.rFonts.set(qn('w:eastAsia'), SONG)
    st.font.size = Pt(size)
    st.font.bold = True


def _set(run, size=10.5, bold=False, name=SONG, color=BLACK):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)


def para(text='', size=10.5, bold=False, align=None, indent=True, space_after=4):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Pt(21)  # 首行缩进 2 字符
    p.paragraph_format.space_after = Pt(space_after)
    if align:
        p.alignment = align
    run = p.add_run(text)
    _set(run, size=size, bold=bold)
    return p


def bullet(text, size=10.5, bold=False, level=0):
    p = doc.add_paragraph(style='List Bullet' if level == 0 else 'List Bullet 2')
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    _set(run, size=size, bold=bold)
    return p


def numbered(text, size=10.5, bold=False):
    p = doc.add_paragraph(style='List Number')
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    _set(run, size=size, bold=bold)
    return p


def shade(p, fill='F2F2F2'):
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:fill'), fill)
    pPr.append(shd)


def code_block(text, size=9):
    """代码块：等宽字体 + 浅灰底 + 左缩进，保留换行。"""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.4)
    p.paragraph_format.right_indent = Cm(0.4)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.0
    shade(p)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        run = p.add_run(line if line else ' ')
        run.font.name = CODE
        run._element.rPr.rFonts.set(qn('w:eastAsia'), SONG)
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor(0x20, 0x20, 0x20)
        if i < len(lines) - 1:
            run.add_break()
    return p


# ======================================================================
# 封面标题
# ======================================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('今日工作记录 与 BCD 分工任务说明')
_set(run, size=20, bold=True)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run('（2026-09-23 · 供下一次对话与组内交接使用）')
_set(run, size=11, bold=False, color=RGBColor(0x60, 0x60, 0x60))

para('', indent=False)

# ======================================================================
# 阅读指引
# ======================================================================
doc.add_heading('阅读指引（先说结论，方便快速定位）', level=1)
para('本文档分两大部分，各服务一个目的：')
bullet('第一部分「今天做了什么」：记录 2026-09-23 这一天对代码做的全部改动、为什么这么做、验证结果、以及踩过的坑，保证换一个对话/换一个人接手时能立刻看懂现在的代码状态。')
bullet('第二部分「BCD 分工任务如何完成」：把 B（后端）、C（前端）、D（测试）三位同学各自的活拆成可照做的具体步骤，附上关键代码骨架，越细越好。')

# ======================================================================
# 第一部分
# ======================================================================
doc.add_heading('第一部分：今天做了什么（2026-09-23）', level=1)

doc.add_heading('1. 背景：为什么今天要做这两件事', level=2)
para('项目是「软件工程2」课程设计——把上学期的稀疏 SfM 算法包成一个完整的三维重建系统。上学期已经能跑通稀疏重建（SIFT→匹配→位姿→三角化→BA→稀疏点云），本学期补上了「密集重建 + 网格化」（plane-sweep 稠密点云 + open3d Poisson 网格），Temple 数据集（images2/ 下 10 张图）的结果已经让 A 满意。')
para('今天碰到的实际问题：A 换用「另外一组 25 张图」测试时，程序报错“图像数 25 与位姿数 18 不一致”，并且匹配点太少导致重建失败。A 一开始以为是内参 K 的问题，于是提出两个需求：')
bullet('跳过坏帧：数据集里混着坏帧（模糊/过曝/无匹配的图），增量重建要能自动跳过它们，不让“坏帧”导致位姿和图像数量对不上、也不影响后续帧。')
bullet('自动 K：原来的内参 K 是硬编码的 Temple 官方标定值，换新图组就用不了，需要程序自己根据图片自动确定 K。')

doc.add_heading('2. 功能一：跳过坏帧（skip bad frames）—— 已完成并验证', level=2)

doc.add_heading('2.1 问题根因', level=3)
para('原来的增量重建是「顺序匹配」：第 i 帧总是和第 i-1 帧匹配，然后用第 i-1 帧的位姿去注册第 i 帧。只要中间有一帧是坏帧（匹配点太少、无法注册），后面所有帧都会“错位”——因为第 i 帧被跳过之后，第 i+1 帧仍然去匹配那个被跳过的坏帧 i，于是位姿对不上，最终“图像数 25 与位姿数 18 不一致”。')

doc.add_heading('2.2 解决办法：惰性匹配（lazy matching）', level=3)
para('核心思路：当前帧不匹配“上一个编号的帧”，而是匹配“最近一次成功注册的帧”。这样即使中间跳过了一堆坏帧，当前帧也能和它前面最近的好帧正确匹配，位姿不会错位。')
para('同时，因为跳帧之后“图像全集”和“已注册的位姿”不再一一对应，必须记录每一帧位姿对应的原始图像索引，后续的颜色分配、BA、重投影、稠密重建都要用这个对齐后的子集。')

doc.add_heading('2.3 具体改动的文件与代码', level=3)
para('改动了 4 个文件，关键改动如下（都是已写好的，可直接看源码）：')

bullet('core/feature_extractor.py', bold=True)
para('extract_from_images() 现在多返回一个第 5 个值 valid_indices——记录成功提取特征的帧在原始图像列表里的索引（有些图读不出来会被丢掉，需要对齐）。')

bullet('main.py', bold=True)
para('主要改动：')
numbered('ReconstructionResult 数据类新增字段 registered_indices（每帧位姿对应的原始图像索引）。')
numbered('Step 5 增量重建循环改成惰性匹配，核心逻辑如下：')
code_block("""registered = [0, 1]   # 已成功注册的帧索引
last_good = 1         # 最近一次成功注册的帧

for i in range(2, len(key_points)):
    if last_good == i - 1:
        m = matches[i - 1]              # 没跳过坏帧：直接复用预计算的相邻匹配
    else:
        m = self.feature_matcher._match_pair(
            descriptors[last_good], descriptors[i],
            self.config.MATCHING['ratio'])   # 跳过坏帧：与「最近成功帧」重新匹配
    try:
        success = self.reconstructor.add_frame(
            key_points[last_good], key_points[i],
            descriptors[last_good], descriptors[i], m)
    except Exception:
        success = False
    if success:
        last_good = i
        registered.append(i)
    else:
        print(f"Skipped frame {i + 1}（坏帧，已跳过，不影响后续帧）")""")
numbered('后续所有步骤（颜色分配、BA、重投影）都改用 registered 子集：registered_key_points / registered_colors / registered_image_idx。')
numbered('返回结果里 num_images = len(registered)，registered_indices = registered_image_idx。')

bullet('core/dense_mvs.py（只改 __main__）', bold=True)
para('稠密重建入口原来要求「图像数 == 位姿数」，现在改成按 registered_indices 对齐：')
code_block("""idx = result.registered_indices
if idx is None or len(idx) != len(result.rotations):
    idx = list(range(len(result.rotations)))
images = [images[i] for i in idx]   # 只保留已注册帧对应的图像""")

doc.add_heading('2.4 验证结果', level=3)
para('人为制造一个坏帧做测试：在 10 张 Temple 图的第 3 个位置插入一张纯黑图（共 11 张），跑完整流水线，结果：')
code_block("""成功注册 10/10 帧
registered_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
# 注意：索引 3（那张纯黑坏帧）被跳过了，其余帧正确映射回原位置""")
para('说明坏帧被正确跳过，且后续帧通过与“最近成功帧”重新匹配成功接上了，位姿没有错位。这正是 25 张图组报错所需的修复。')

doc.add_heading('3. 功能二：自动 K（auto-K）—— 已完成并验证', level=2)

doc.add_heading('3.1 最终方案（先说结论）', level=3)
para('自动 K 最终走「EXIF 焦距 → 默认 f = 1.0×图像宽」这条路线，方形像素、主点（principal point）放在图像中心。逻辑在 core/calibration.py 的 auto_intrinsics() 函数里，config.py 新增开关 Config.AUTO_CALIBRATE（默认 True）。')

doc.add_heading('3.2 过程：为什么最后没有用“自标定”', level=3)
para('A 一开始想做“自标定”（self-calibration）：只凭两张图的基础矩阵 F，反推出焦距 f。这是计算机视觉里经典的难题，尝试了两种方法，实测都不可靠：')
bullet('方法一：本质矩阵的 σ1=σ2 约束（E = KᵀF K，正确 f 时 E 的前两个奇异值应相等）。实测在环拍数据上，|σ1-σ2| 的极小值总落在搜索下边界 f=0.5W，是伪极小。')
bullet('方法二：cheirality + 重投影误差搜索（对每个候选 f 分解 E、三角化、算重投影误差，取误差最小的 f）。实测重投影误差随 f 单调增大（f 越小误差越小），极小值同样总在 0.5W 下边界，没有真实极小值。')
para('两个方法还会随抽样点数变化在 0.78W~1.2W 之间乱跳，极不稳定。结论：这套环拍数据上自标定不可靠，继续用会给出错误的 f，反而毁掉重建。所以果断弃用，改走“EXIF → 默认值”这条稳健、可预测的路线。')
para('（注：这背后的物理原因是：SfM 的焦距、位姿、深度尺度是耦合的，两视图下只凭几何无法唯一确定 f；即使加上正深度约束，重投影误差对 f 也不敏感，反而被“f 越小点越近、误差越小”的假象主导。）')

doc.add_heading('3.3 具体改动的文件与代码', level=3)
bullet('core/calibration.py（新建）', bold=True)
para('核心函数：')
code_block("""def build_intrinsics(f, W, H):
    # 方形像素 + 主点居中
    return np.array([[f, 0, W/2.0],
                     [0, f, H/2.0],
                     [0, 0, 1.0]])

def read_exif_focal(image_path, W):
    # 从 EXIF 读焦距(mm)→像素，失败返回 None
    # 优先 FocalLength(mm) × FocalPlaneXResolution(px/mm)
    # 否则 FocalLengthIn35mmFilm / 36 × W

def auto_intrinsics(image_path, W, H, default_f_ratio=1.0):
    # 优先级：EXIF → 默认 f = default_f_ratio × W
    f = read_exif_focal(image_path, W)
    if f 不合理: f = default_f_ratio * W
    return build_intrinsics(f, W, H), 来源字符串""")
bullet('config.py', bold=True)
para('新增开关 Config.AUTO_CALIBRATE = True；官方 K 保留为 _OFFICIAL_K 只作参考；更新了文件头注释。')
bullet('main.py', bold=True)
para('在特征匹配之后、初始化重建之前插入 Step 3.5：')
code_block("""if getattr(self.config, 'AUTO_CALIBRATE', True):
    H_img, W_img = images[0].shape[:2]
    new_K, source = auto_intrinsics(image_names[kept_idx[0]], W_img, H_img)
    self.config.K = new_K
    self.reconstructor = IncrementalReconstructor(self.config.K)
    self.ba_optimizer = BundleAdjustment(self.config.K)""")

doc.add_heading('3.4 验证结果', level=3)
para('用自动 K 重跑 Temple（10 张图，2832×2128，无 EXIF，所以走默认 f=1.0W=2832px）：')
code_block("""新 K: fx=2832.0  fy=2832.0  cx=1416.0  cy=1064.0
成功注册 10/10 帧
平均重投影误差: 0.2241 像素
最终点数: 4006（CUDA BA 过滤后）""")
para('对比：之前硬编码官方 K 时重投影误差 0.2338px。自动 K 不仅没破坏已满意的结果，反而略好一点（因为主点居中 + 焦距 1.0W 更贴合实际 2832×2128 的图）。')

doc.add_heading('4. 今天所有改动的文件清单（供下一次对话快速对照）', level=2)
para('本次会话涉及的 5 个文件：')
numbered('core/calibration.py —— 新建。自动 K 模块（EXIF→默认），已删掉不可靠的自标定函数。')
numbered('config.py —— 新增 AUTO_CALIBRATE 开关，更新注释（官方 K 只作参考）。')
numbered('main.py —— 加 Step 3.5 自动 K；Step 5 增量重建改惰性匹配+registered 子集；ReconstructionResult 加 registered_indices；返回 num_images=注册帧数。')
numbered('core/feature_extractor.py —— extract_from_images 多返回 valid_indices。')
numbered('core/dense_mvs.py —— __main__ 按 registered_indices 对齐图像，删掉“图像数必须等于位姿数”的硬检查。')

doc.add_heading('5. 重要约定与坑（务必记牢，下次别重踩）', level=2)
numbered('位姿约定：rotations[i] / motions[i] 是「世界→相机」的变换；投影 x = K(RX + t)；相机中心 C = -Rᵀt；第 0 帧是世界原点。dense_mvs 和网格化都按这个约定，别改反。')
numbered('图像尺寸：images2/ 实际是 2832×2128（W×H），不是老注释里写的 640×480，也不是官方 3072×2048。')
numbered('CUDA BA 名不副实：core/bundle_adjustment_cuda.py 不是真正的 BA，只是并行算重投影误差后按阈值过滤点；真正的优化在 CPU 的 bundle_adjustment.py。')
numbered('自标定别再试了：两种方法都落到 f=0.5W 的伪极小，已弃用。若以后要更准的 K，优先做「BA 中联合优化焦距 f」，而不是两视图自标定。')
numbered('当前未完成的后续项（已在别处记录）：密集重建全分辨率、NCC 一致性、左右一致性校验、纹理映射（网格贴图还偏暗、天空被贴进去的问题）。')

# ======================================================================
# 第二部分：BCD 分工
# ======================================================================
doc.add_heading('第二部分：B、C、D 同学的分工任务如何完成', level=1)

doc.add_heading('1. 总分工与依赖关系（先记住这张图）', level=2)
para('四人按“系统分层”分工：')
bullet('A —— 计算/算法层（就是本记录的作者）：封装重建算法、密集重建、GPU 优化、可行性分析。今天已完成“跳过坏帧 + 自动 K”。')
bullet('B —— 后端/数据层：用户与项目管理、图片存储、数据库、任务队列、日志、API 鉴权。')
bullet('C —— 前端/展示交互层：Web 上传页、重建状态监控、Three.js 3D 模型展示。')
bullet('D —— 测试/集成/部署：单元/集成/性能测试、测试文档、部署方案、最终报告汇总。')
para('依赖关系：A 先交出“可调用的重建 API/服务”→ B 和 C 联调；D 全程并行（别人写代码，D 写测试）。')

doc.add_heading('2. A 已经交出的东西（B、C、D 都要依赖，直接拿来用）', level=2)

doc.add_heading('2.1 统一重建入口', level=3)
code_block("""# main.py 里的对外统一入口（B 同学直接 import 它）
from main import reconstruct, ReconstructionResult

result = reconstruct(
    image_path,      # 图片目录路径，或图片文件路径列表
    config=None,     # 可选 Config；不传则新建
    save=False,      # 是否把点云/统计写到 output/
    visualize=False, # 是否弹 matplotlib 窗口（后端/服务器必须 False）
)
# 返回 ReconstructionResult 对象""")
para('ReconstructionResult 的字段（B/C/D 关心的都在这）：')
code_block("""points: (N,3) 三维点坐标
colors: (N,3) 颜色 RGB 0-255
reprojection_error: 平均重投影误差（像素）
num_points: 点数量
num_images: 参与重建的图像数量（已跳过坏帧后的帧数）
timings: 各阶段耗时字典
rotations / motions: 每帧相机位姿（供密集重建用）
registered_indices: 每帧位姿对应的原始图像索引（跳帧后与图像全集不再一致）
to_dict(): 转成 JSON 可序列化字典（只含标量统计）""")

doc.add_heading('2.2 输出产物（都在 output/ 目录）', level=3)
code_block("""output/point_cloud.ply / .obj / .txt     稀疏点云
output/sparse_cloud_meshlab.ply           MeshLab 可直接打开的彩色点云
output/statistics.json / performance.json 统计与耗时
output/mesh/mesh.glb / .obj / .ply       网格模型（含 GLB，C 同学前端直接加载）
output/dense/dense_cloud.ply             稠密点云（~76 万点）
output/dense/mesh/mesh.glb / .obj / .ply 稠密网格（含纹理，可给前端）""")

doc.add_heading('3. B 同学（后端/数据层）怎么完成 —— 分步照做', level=2)
para('目标：把 A 的 reconstruct() 包成一个 Web 服务，提供“上传图片 → 触发重建 → 查进度 → 下载模型”的 API，并做好数据存储、任务队列、日志、鉴权。')

doc.add_heading('3.1 选框架和搭骨架', level=3)
para('推荐 FastAPI（自动生成 API 文档、写起来快）或 Flask（更简单）。安装：')
code_block("pip install fastapi uvicorn sqlalchemy\n# 或 flask\npip install flask")
para('在项目里新建 backend/ 目录，结构建议：')
code_block("""backend/
  app.py            # FastAPI 入口 + 路由
  database.py       # 数据库连接（SQLite 即可，单文件、零配置）
  models.py         # 表结构：用户、项目、任务、模型
  tasks.py          # 后台重建任务（线程/进程池 或 Celery）
  auth.py           # 登录/鉴权（token 或 JWT）
  static/           # 存放上传的图片、产出的模型文件""")

doc.add_heading('3.2 数据库表设计（最小可用）', level=3)
code_block("""User:    id, username, password_hash
Project: id, user_id, name, created_at   # 一次重建 = 一个项目
Image:   id, project_id, file_path       # 该项目上传的图片
Task:    id, project_id, status, error, created_at, finished_at
         # status: pending / running / success / failed
Model:   id, project_id, file_path       # 产出的 GLB/PLY 路径""")

doc.add_heading('3.3 核心 API 设计', level=3)
code_block("""POST /api/register           注册（用户名+密码）
POST /api/login              登录，返回 token
POST /api/upload             上传多张图片 → 存到 static/，返回 project_id
POST /api/reconstruct        触发重建（project_id）→ 返回 task_id
GET  /api/task/{task_id}     查询重建进度/状态（轮询用）
GET  /api/model/{project_id} 下载模型文件（GLB/PLY）
GET  /api/health             健康检查（D 同学测试会用到）""")

doc.add_heading('3.4 怎么调用 A 的重建（关键：必须放后台）', level=3)
para('因为重建很慢（Temple 一次约 24 秒），绝不能在 HTTP 请求线程里同步跑，否则请求超时。要放到后台线程/进程，用任务表记录状态：')
code_block("""# tasks.py 里，简化版（线程池方案）
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=2)   # 同时最多 2 个重建任务

def run_reconstruct(task_id, project_id, image_paths):
    update_task(task_id, 'running')
    try:
        from main import reconstruct
        result = reconstruct(image_paths, save=True, visualize=False)
        update_task(task_id, 'success')
        save_model_path(task_id, project_id)   # 记录产出模型路径
    except Exception as e:
        update_task(task_id, 'failed', error=str(e))

# 路由里：executor.submit(run_reconstruct, ...) 后立刻返回 task_id""")
para('进阶（可选加分）：用 Celery + Redis 做真正的分布式任务队列，配合消息队列讲清楚“生产者/消费者”，正好对上课程 PPT 里的 Pipeline/消息队列知识点。')

doc.add_heading('3.5 日志与鉴权', level=3)
bullet('日志：用 Python logging，把每次请求、每个重建任务的开始/结束/耗时/报错写到日志文件，方便 D 同学和答辩时讲“可观测性”。')
bullet('鉴权：登录发 token，/api/reconstruct、/api/upload 等敏感接口校验 token（可用 itsdangerous 或 PyJWT 简单实现）。')
bullet('跨域：前端 C 同学调接口会遇到 CORS，后端要开 CORS（FastAPI 加 CORSMiddleware）。')

doc.add_heading('4. C 同学（前端/展示层）怎么完成 —— 分步照做', level=2)
para('目标：做 Web 页面，用户上传图片 → 看重建进度 → 用 Three.js 在浏览器里 3D 查看重建出的模型。')

doc.add_heading('4.1 技术选型（尽量简单）', level=3)
para('HTML + 原生 JS + Three.js（CDN 引入即可，不用 npm 打包）：')
code_block("""<!-- three.js 与 GLTFLoader、OrbitControls 用 CDN 引入 -->
<script type="importmap">
{ "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" } }
</script>""")

doc.add_heading('4.2 页面组成（三个页面/三个区域）', level=3)
numbered('上传区：多图选择/拖拽 → 调 POST /api/upload，拿到 project_id，再调 POST /api/reconstruct 拿到 task_id。')
numbered('进度区：用 setInterval 定时轮询 GET /api/task/{task_id}，显示“待处理/重建中/成功/失败”，成功后可下载。')
numbered('展示区：Three.js 加载 A 产出的 GLB 模型。')

doc.add_heading('4.3 Three.js 加载 GLB 的关键代码', level=3)
code_block("""import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.8));
scene.add(new THREE.DirectionalLight(0xffffff, 1.2));

const loader = new GLTFLoader();
loader.load('/api/model/1', (gltf) => {   // 后端返回的模型 URL
    scene.add(gltf.scene);
    // 自动框住模型并调相机
});

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
animate();""")

doc.add_heading('4.4 前端要注意的坑', level=3)
bullet('模型格式：优先用 A 已导出的 GLB（output/mesh/mesh.glb，open3d 已能直接导出），别自己从 PLY 转。')
bullet('模型文件可能很大（几十 MB），后端要用流式下载，前端加载时给个进度提示。')
bullet('跨域 CORS：前端页面和后端 API 不同端口时要开后端 CORS（已在上文 B 部分提醒）。')

doc.add_heading('5. D 同学（测试/部署）怎么完成 —— 分步照做', level=2)
para('目标：给整个系统写测试（单元/集成/性能/模型质量），输出测试文档与缺陷分析，最后汇总最终报告。任务书明确要求 TDD。')

doc.add_heading('5.1 单元测试（pytest，对着 A 的模块逐个测）', level=3)
code_block("""pip install pytest
# 项目里新建 tests/ 目录，例如 tests/test_calibration.py
import numpy as np
from core.calibration import build_intrinsics, read_exif_focal

def test_build_intrinsics_center():
    K = build_intrinsics(2800, 2832, 2128)
    assert K[0,0] == 2800 and K[0,2] == 1416 and K[1,2] == 1064

def test_read_exif_none_when_no_exif():
    # Temple 的 jpg 无 EXIF，应返回 None
    assert read_exif_focal('images2/00001.jpg', 2832) is None""")
para('其它要测的：feature_matcher 的匹配、reconstruction 的位姿、calibration 的 auto_intrinsics、dense_mvs 的多视图融合、meshing 的导出。')

doc.add_heading('5.2 集成测试（全链路）', level=3)
para('跑通“上传 → 重建 → 下载模型”整条链路，重点断言：')
code_block("""result = reconstruct('images2', save=False, visualize=False)
assert result.num_images == 10            # 10 张 Temple 全部注册
assert result.reprojection_error < 0.5    # 重投影误差合格
assert result.points.shape[1] == 3        # 点云是 (N,3)""")
para('项目里已有 check_api.py，D 可以在此基础上扩展（注意它最后有一行 emoji 在 GBK 控制台会报 UnicodeEncodeError，跑的时候加 PYTHONIOENCODING=utf-8）。')

doc.add_heading('5.3 性能测试与模型质量测试', level=3)
bullet('性能：A 的 result.timings 里已经有各阶段耗时（特征提取 ~8s、匹配 ~14s、BA ~1.4s、总计 ~24s），D 整理成表格，对比 CPU/GPU（BA 有 use_cuda 开关）。')
bullet('模型质量：重投影误差（目标 <0.5px，实测 0.22px）、点云数量、网格是否有洞/是否闭合。')
bullet('缺陷分析：把测试发现的 bug 记录成表（编号、描述、复现步骤、状态），体现 TDD 的“闭环迭代”。')

doc.add_heading('5.4 部署', level=3)
para('给系统写一个 Dockerfile 或一键启动脚本，把后端 + 前端 + 重建环境打包。注意本机是 py3.14，装 open3d 要 0.20.0 版、torch 要 2.10+cu130，Docker 镜像里版本要写对，否则装不上。')

doc.add_heading('6. 硬件与网络（香橙派 + 多摄像头 + TCP）—— 最大缺口，单独说明', level=2)
para('这是本学期任务书相对上学期的“最大新增”，也最容易漏。它不属于 B/C/D 常规 Web 部分，建议这样安排：')
numbered('采集端（香橙派 + 多摄像头）：负责多路摄像头同步采集图片，通过 TCP 把图片传给上位机（PC）。')
numbered('上位机（PC）：接收图片 → 调 A 的 reconstruct() 做重建 → 展示。')
numbered('TCP 通讯：香橙派作客户端，PC 作服务端，传图片字节流 + 控制指令（开始/停止采集）。')
para('建议分工：B 负责 TCP 上下位机通讯协议 + 数据层（正好是“后端/通讯”），A 负责多摄像头图像对齐/融合的算法，或两人合做。')
para('如果硬件买不到或来不及，必须写“降级方案”：用本地多个目录/多组图片模拟多摄像头采集（现在跑通的就是单摄像头环拍），在文档里明确写清“硬件用模拟器替代”，这样答辩有交代。')

doc.add_heading('7. 关键环境约束（B/C/D 都别踩）', level=2)
bullet('本机 Python 3.14，无 conda；open3d 只能装 0.20.0（已装好），torch 2.10+cu130，GPU RTX 5060 8GB。')
bullet('nerfstudio / instant-ngp / Fast3R 都装不上或显存不够，别在主线里依赖它们；NeRF 只作为“可选进阶模块”在文档里提一句。')
bullet('项目路径 C:\\Users\\speechless\\Desktop\\SFM01\\sfm_project，用 PyCharm 打开；运行重建是 GPU 密集，注意 8GB 显存别开太大批量。')

doc.save('今日工作记录与BCD分工说明.docx')
print('已生成: 今日工作记录与BCD分工说明.docx')
