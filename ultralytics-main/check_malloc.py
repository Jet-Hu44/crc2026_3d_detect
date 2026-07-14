"""
CANN 7.0 API 深度检查
"""
import sys
sys.path.append('/usr/local/Ascend/ascend-toolkit/latest/python/site-packages')
import acl
import numpy as np

acl.init("")
acl.rt.set_device(0)

print("=== malloc(100, X) — 已 set_device ===")
for v in [0, 1, 2]:
    result = acl.rt.malloc(100, v)
    if isinstance(result, tuple):
        ptr, ret = result[0], result[-1]
    else:
        ptr, ret = result, 0
    print(f"  malloc(100, {v}) -> ptr={ptr:#x}, ret={ret} {'✅' if ret==0 else '❌ err='+str(ret)}")
    if ptr and ret == 0:
        acl.rt.free(ptr)

print("\n=== memcpy signature hint ===")
# 尝试用 malloc_host + 指针 方式
host_ptr, ret = acl.rt.malloc_host(100)
print(f"  malloc_host(100) -> {host_ptr:#x}, ret={ret}")
if ret == 0:
    dev_ptr, ret = acl.rt.malloc(100, 0)
    dev_ptr = dev_ptr[0] if isinstance(dev_ptr, tuple) else dev_ptr
    print(f"  malloc(100,0) -> {dev_ptr:#x}, ret={ret}")
    if dev_ptr and ret == 0:
        # 拷贝 numpy 到 host
        data = np.ones(25, dtype=np.float32)
        ctypes_buf = data.ctypes.data
        acl.rt.memcpy(host_ptr, 100, ctypes_buf, 100, 3)  # 3=HOST_TO_HOST
        # 尝试 H2D
        for d in range(4):
            try:
                ret = acl.rt.memcpy(dev_ptr, 100, host_ptr, 100, d)
                print(f"  memcpy(dev,100,host,100,{d}) -> {ret} {'✅' if ret==0 else ''}")
            except Exception as e:
                print(f"  memcpy(dev,100,host,100,{d}) -> {e}")
        acl.rt.free(dev_ptr)
    acl.rt.free_host(host_ptr)

print("\n=== 尝试 np array 直接传入 memcpy ===")
dev_ptr, ret = acl.rt.malloc(100, 0)
dev_ptr = dev_ptr[0] if isinstance(dev_ptr, tuple) else dev_ptr
if dev_ptr and ret == 0:
    data = np.ones(25, dtype=np.float32)
    try:
        ret = acl.rt.memcpy(dev_ptr, 100, data, 100, 1)
        print(f"  memcpy(dev,100, np_array, 100, 1) -> {ret} {'✅' if ret==0 else ''}")
    except Exception as e:
        print(f"  memcpy(dev,100, np_array, 100, 1) -> {e}")
    acl.rt.free(dev_ptr)

print("\n=== load_from_file 返回值 ===")
result = acl.mdl.load_from_file("best4.om")
print(f"  load_from_file -> type={type(result).__name__}, len={len(result) if hasattr(result,'__len__') else '?'}, value={result if not hasattr(result,'__len__') else result[:2]}")

acl.rt.reset_device(0)
acl.finalize()
