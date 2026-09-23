
import pycuda.autoinit
import pycuda.driver as drv
print(f"CUDA设备: {drv.Device(0).name()}")
print(f"CUDA版本: {drv.get_version()}")