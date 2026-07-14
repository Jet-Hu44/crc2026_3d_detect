"""
检查 CANN 7.0 malloc 第2参数
"""
import sys
sys.path.append('/usr/local/Ascend/ascend-toolkit/latest/python/site-packages')
import acl

print("=== acl 整型常量 ===")
for name in sorted(dir(acl)):
    val = getattr(acl, name)
    if isinstance(val, int):
        print(f"  acl.{name} = {val}")

print("\n=== acl.rt 整型常量 ===")
for name in sorted(dir(acl.rt)):
    val = getattr(acl.rt, name)
    if isinstance(val, int):
        print(f"  acl.rt.{name} = {val}")

print("\n=== 尝试 malloc(100, X) ===")
acl.init("")
for v in [0, 1, 2, 3]:
    try:
        result = acl.rt.malloc(100, v)
        if isinstance(result, tuple):
            ptr, ret = result
        else:
            ptr, ret = result, -1
        print(f"  malloc(100, {v}) -> ptr={ptr}, ret={ret}  ✅")
        acl.rt.free(ptr)
    except Exception as e:
        print(f"  malloc(100, {v}) -> {e}  ❌")

print("\n=== 尝试 memcpy dir: 0-3 ===")
import numpy as np
src, ret = acl.rt.malloc(100, 0)
dst, ret = acl.rt.malloc(100, 0)
data = np.ones(25, dtype=np.float32).tobytes()
for d in range(4):
    try:
        ret = acl.rt.memcpy(src, 100, data, 100, d)
        print(f"  memcpy(src,100,data,100,{d}) -> {ret}")
    except Exception as e:
        print(f"  memcpy(src,100,data,100,{d}) -> {e}")
acl.rt.free(src)
acl.rt.free(dst)
acl.finalize()
