from pathlib import Path
from collections.abc import Iterable, Mapping

import yaml

import analyze


class Timer:
    def __init__(
        self,
        cmake_args: list[str] = [],
        models: Iterable[str] = analyze.models,
        **kwargs,
    ):
        self.cmake_args = cmake_args
        self.models = models
        self.kwargs = kwargs
        self.model2exe: dict[str, Path] = {}

    def compile(self, build_dir: Path):
        for name in self.models:
            model = analyze.models[name]
            exe = analyze.compile(
                Path(name),
                extra_args=model.cmake_args + self.cmake_args,
                build_dir=build_dir / name,
                **self.kwargs,
            )
            self.model2exe[name] = exe

    def __call__(self) -> Iterable[float]:
        for name, exe in self.model2exe.items():
            model = analyze.models[name]
            time = analyze.timeit_exe(
                exe, extra_args=model.simulate_args, root_dir=Path(name)
            )
            yield time


def single_precision():
    ref = Timer(build_type="Release", fabm_dir="fabm2")
    exp = Timer(
        build_type="Release",
        models=[n for n in analyze.models if n not in ["bfm", "ergom"]],
        cmake_args=["-DFABM_REAL_KIND_INTERNAL='SELECTED_REAL_KIND(6)'"],
        fabm_dir="fabm2",
    )
    return ref, exp


def fabm3():
    ref = Timer(build_type="Release", fabm_dir="fabm2")
    exp = Timer(build_type="Release", fabm_dir="fabm3")
    return ref, exp


def fabm3_o3_qxhost():
    ref = Timer(build_type="Release", fabm_dir="fabm2")
    exp = Timer(
        build_type="Release",
        fabm_dir="fabm3",
        cmake_args=["-DCMAKE_Fortran_FLAGS_RELEASE=/O3 /DNDEBUG /QxHost"],
    )
    return ref, exp


def qxhost():
    ref = Timer(build_type="Release", extra_release_flags=[])
    exp = Timer(build_type="Release", extra_release_flags=["/QxHost"])
    return ref, exp


def o3():
    ref = Timer(build_type="Release")
    exp = Timer(
        build_type="Release",
        cmake_args=["-DCMAKE_Fortran_FLAGS_RELEASE=/O3 /DNDEBUG /QxHost"],
    )
    return ref, exp


def report(
    ref_timer: Timer,
    exp_timer: Timer,
    exp_name: str = "experiment",
    ref_name: str = "reference",
):
    base = Path(__file__).parent.absolute()
    ref_timer.compile(build_dir=base / f"{exp_name}/build-ref")
    exp_timer.compile(build_dir=base / f"{exp_name}/build-exp")
    ref = {}
    exp = {}
    models = ref_timer.models = exp_timer.models
    for name, exptime, reftime in zip(models, exp_timer(), ref_timer()):
        exp[name] = exptime
        ref[name] = reftime
    print(ref)
    print(exp)
    for name in exp.keys():
        print(
            f"{name}: {ref_name} {ref[name]:.3f}s, {exp_name} {exp[name]:.3f}s, ratio {exp[name]/ref[name]:.2f}"
        )
    with open(f"compare_{ref_name}_vs_{exp_name}.txt", "w") as f:
        yaml.safe_dump((ref, exp), f)


if __name__ == "__main__":
    # FABM2 single vs double precision
    # fabm2, fabm2_sp = single_precision()
    # report(fabm2, fabm2_sp, exp_name="fabm2_sp", ref_name="fabm2")

    # FABM2 vs FABM3
    # fabm2, fabm3 = fabm3()
    # report(fabm2, fabm3, exp_name="fabm3", ref_name="fabm2")

    # FABM3 O3 QxHost vs FABM2
    fabm2, fabm3_o3_qxhost = fabm3_o3_qxhost()
    report(
        fabm2,
        fabm3_o3_qxhost,
        exp_name="fabm3_o3_qxhost",
        ref_name="fabm2",
    )

    # QxHost vs reference
    # ref, qxhost = qxhost()
    # report(ref, qxhost, exp_name="qxhost", ref_name="reference")

    # O3 vs reference
    # ref, o3 = o3()
    # report(ref, o3, exp_name="o3", ref_name="reference")
