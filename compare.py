import analyze


def time_all(cmake_args=[], **kwargs):
    result = {}
    for name, model in analyze.models.items():
        exe = analyze.compile(name, extra_args=model.cmake_args + cmake_args, **kwargs)
        time = analyze.timeit_exe(exe, extra_args=model.simulate_args, root_dir=name)
        result[name] = time
    return result


# fabm2_result = time_all(build_type="Release", fabm_dir="fabm2")
# fabm3_result = time_all(build_type="Release", fabm_dir="fabm3")
# print(fabm2_result)
# print(fabm3_result)
# for name in fabm2_result.keys():
#     print(
#         f"{name}: FABM2 {fabm2_result[name]:.3f}s, FABM3 {fabm3_result[name]:.3f}s, ratio {fabm3_result[name]/fabm2_result[name]:.2f}"
#     )

# ref_result = time_all(build_type="Release", extra_release_flags=[])
# qxhost_result = time_all(build_type="Release", extra_release_flags=["/QxHost"])
# print(qxhost_result)
# print(ref_result)
# for name in qxhost_result.keys():
#     print(
#         f"{name}: FABM2 {ref_result[name]:.3f}s, FABM3 {qxhost_result[name]:.3f}s, ratio {qxhost_result[name]/ref_result[name]:.2f}"
#     )

ref_result = time_all(build_type="Release")
alt_result = time_all(build_type="Release", cmake_args=["-DCMAKE_Fortran_FLAGS_RELEASE=/O3 /DNDEBUG /QxHost"])
print(alt_result)
print(ref_result)
for name in alt_result.keys():
    print(
        f"{name}: reference {ref_result[name]:.3f}s, alternative {alt_result[name]:.3f}s, alt:ref ratio {alt_result[name]/ref_result[name]:.2f}"
    )
