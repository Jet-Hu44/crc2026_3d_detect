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

print("\n=== 尝试 malloc(100, *) ===")
for name in sorted(dir(acl)):
    val = getattr(acl, name)
    if isinstance(val, int):
        try:
            ptr, ret = acl.rt.malloc(100, val)
            print(f"  malloc(100, {name}={val}) -> ptr={ptr}, ret={ret}")
            acl.rt.free(ptr)
        except Exception as e:
            pass
