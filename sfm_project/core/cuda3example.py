import torch
import time


def main():
    # 检查CUDA
    print("CUDA 可用:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("请检查CUDA与PyTorch安装！")
        return

    # gpu预热
    WARMUP = 5
    REPEAT = 20
    torch.cuda.synchronize()

    # 1. 向量加法
    print("\n========== 1. 向量加法 ==========")
    n = 16 * 1024 * 1024

    # CPU
    a_cpu = torch.rand(n)
    b_cpu = torch.rand(n)
    t0 = time.time()
    for _ in range(REPEAT):
        c_cpu = a_cpu + b_cpu
    t_cpu = (time.time() - t0) / REPEAT

    # GPU
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    for _ in range(WARMUP):
        _ = a_gpu + b_gpu
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(REPEAT):
        c_gpu = a_gpu + b_gpu
    torch.cuda.synchronize()
    t_gpu = (time.time() - t0) / REPEAT

    # 计算加速比
    speed1 = t_cpu / max(t_gpu, 1e-6)
    print(f"CPU 平均: {t_cpu:.4f}s")
    print(f"GPU 平均: {t_gpu:.4f}s")
    print(f"加速比: {speed1:.2f} 倍")

    # 2. 矩阵乘法
    print("\n========== 2. 矩阵乘法 ==========")
    M, N, K = 2048, 2048, 2048

    # CPU
    A_cpu = torch.randn(M, K)
    B_cpu = torch.randn(K, N)
    t0 = time.time()
    C_cpu = A_cpu @ B_cpu
    t_cpu = time.time() - t0

    # GPU
    A_gpu = A_cpu.cuda()
    B_gpu = B_cpu.cuda()
    for _ in range(WARMUP):
        _ = A_gpu @ B_gpu
    torch.cuda.synchronize()
    t0 = time.time()
    C_gpu = A_gpu @ B_gpu
    torch.cuda.synchronize()
    t_gpu = time.time() - t0

    speed2 = t_cpu / max(t_gpu, 1e-6)
    print(f"CPU: {t_cpu:.4f}s")
    print(f"GPU: {t_gpu:.4f}s")
    print(f"加速比: {speed2:.2f} 倍")

    # 3. 图像滤波（修复除零错误）
    print("\n========== 3. 图像滤波 ==========")
    B, C, H, W = 1, 1, 1080, 1920

    # CPU
    img_cpu = torch.randn(B, C, H, W)   # 生成随机图像
    kernel_cpu = torch.ones(1, 1, 3, 3) / 9.0  # 3x3 均值卷积核
    t0 = time.time()
    out_cpu = torch.nn.functional.conv2d(img_cpu, kernel_cpu, padding=1)
    t_cpu = time.time() - t0

    # GPU
    img_gpu = img_cpu.cuda()
    kernel_gpu = kernel_cpu.cuda()
    for _ in range(WARMUP):
        _ = torch.nn.functional.conv2d(img_gpu, kernel_gpu, padding=1)
    torch.cuda.synchronize()

    # 【修复】多跑几次，防止时间太小变成0
    loop = 10
    t0 = time.time()
    for _ in range(loop):
        out_gpu = torch.nn.functional.conv2d(img_gpu, kernel_gpu, padding=1)
    torch.cuda.synchronize()
    t_gpu = (time.time() - t0) / loop

    # 安全除法
    speed3 = t_cpu / max(t_gpu, 1e-6)
    print(f"CPU: {t_cpu:.4f}s")
    print(f"GPU: {t_gpu:.4f}s")
    print(f"加速比: {speed3:.2f} 倍")

    print("\nCUDA 基础样例全部完成")


if __name__ == '__main__':
    main()