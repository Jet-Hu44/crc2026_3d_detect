"""
ACL API 诊断脚本 — 检查 CANN 7.0 的实际 API
"""
import sys
sys.path.append('/usr/local/Ascend/ascend-toolkit/latest/python/site-packages')
import acl
import inspect

print("=" * 60)
print("CANN 7.0 ACL API 诊断")
print("=" * 60)

# 1. 所有非私有属性
attrs = sorted([x for x in dir(acl) if not x.startswith('_')])
print(f"\n[1] acl 模块属性 ({len(attrs)} 个):")
for a in attrs:
    print(f"    acl.{a}")

# 2. acl.rt 属性
print(f"\n[2] acl.rt 属性:")
rt_attrs = sorted([x for x in dir(acl.rt) if not x.startswith('_')])
for a in rt_attrs:
    print(f"    acl.rt.{a}")

# 3. acl.mdl 属性
print(f"\n[3] acl.mdl 属性:")
mdl_attrs = sorted([x for x in dir(acl.mdl) if not x.startswith('_')])
for a in mdl_attrs:
    print(f"    acl.mdl.{a}")

# 4. 关键函数签名
print(f"\n[4] 关键函数签名:")
for func_name in ['init', 'malloc', 'memcpy', 'set_device',
                   'load_from_file', 'create_stream', 'execute',
                   'synchronize_stream', 'free', 'unload',
                   'reset_device', 'finalize', 'create_data_buffer',
                   'create_dataset', 'add_dataset_buffer', 'destroy_dataset',
                   'get_input_size_by_index', 'get_output_size_by_index',
                   'create_desc', 'get_desc', 'destroy_desc']:
    try:
        if hasattr(acl, func_name):
            f = getattr(acl, func_name)
        elif hasattr(acl.rt, func_name):
            f = getattr(acl.rt, func_name)
        elif hasattr(acl.mdl, func_name):
            f = getattr(acl.mdl, func_name)
        else:
            print(f"    ??? {func_name}: 未找到")
            continue
        sig = inspect.signature(f)
        print(f"    {func_name}{sig}")
    except Exception as e:
        print(f"    {func_name}: 检查失败 ({e})")

# 5. 所有整型常量
print(f"\n[5] 整型常量:")
for name in sorted(dir(acl)):
    val = getattr(acl, name)
    if isinstance(val, int) and not name.startswith('_'):
        print(f"    acl.{name} = {val}")

print("\n" + "=" * 60)
print("诊断完成")
