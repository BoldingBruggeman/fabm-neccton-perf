import os
import subprocess
import io
import shutil
import timeit
import argparse
from typing import Optional
from pathlib import Path
from collections.abc import Container

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.text
import adjustText


class Model:
    def __init__(
        self,
        *,
        cmake_args: list[str],
        host: str = "nemo",
        dt: float,
        nx: int,
        ny: int,
        nz: int,
        ndays: float = 30,
        fabm_yaml: str = "fabm.yaml",
        diagnostics: Optional[str] = None,
    ):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dt = dt
        self.fabm_yaml = fabm_yaml
        self.ndays = ndays
        self.diagnostics = diagnostics
        self.cmake_args = [f"-DFABM_HOST={host}"] + cmake_args

    @property
    def simulate_args(self) -> list[str]:
        args = [
            "--nx",
            str(self.nx),
            "--ny",
            str(self.ny),
            "--nz",
            str(self.nz),
            "--dt",
            str(self.dt),
            "-n",
            str(int(round(self.ndays * 3600 * 24 / self.dt))),
            "--nomask",
        ]
        if self.diagnostics is not None:
            args += ["--diag", self.diagnostics]
        else:
            args += ["--nodiag"]
        return args + ["-s", self.fabm_yaml]


models = dict(
    ecosmo=Model(
        cmake_args=[
            "-DFABM_NERSC_BASE=nersc",  # "-DFABM_NERSC_BASE=nersc-modular"
            "-DFABM_ERSEM_BASE=ersem",
        ],
        host="hycom",
        nx=15,
        ny=14,
        nz=50,
        dt=600,
        ndays=150,
        fabm_yaml="nersc/dvm/fabm.yaml",
        # fabm_yaml="nersc-modular/modular/fabm.yaml.modular",
    ),
    bfm=Model(
        cmake_args=[
            "-DFABM_EXTRA_INSTITUTES=spectral",
            "-DFABM_OGS_BASE=ogs",
            "-DFABM_SPECTRAL_BASE=spectral",
        ],
        nx=16,
        ny=15,
        nz=124,
        dt=360,
        ndays=5,
        fabm_yaml="ogs/fabm_multispectral_2xDetritus.yaml",
        # fabm_yaml="ogs/fabm_diatoms_60PFTs_no-repr_OASIM.yaml",
    ),
    ersem=Model(
        cmake_args=["-DFABM_ERSEM_BASE=ersem"],
        nx=16,
        ny=16,
        nz=51,
        dt=300,
        fabm_yaml="fabm.yaml.UKMO.operational",
    ),
    ergom=Model(
        cmake_args=[
            "-DFABM_EXTRA_INSTITUTES=bsh;spectral",
            "-DFABM_BSH_BASE=ergom/src",
            "-DFABM_SPECTRAL_BASE=spectral",
            "-DFABM_OGS_BASE=ogs",
        ],
        nx=16,
        ny=16,
        nz=56,
        dt=90,
        ndays=5,
        fabm_yaml="fabm_spectral.yaml",
        diagnostics="diag.yaml",
    ),
    pisces=Model(
        cmake_args=["-DFABM_PISCES_BASE=fabm-pisces-4.2"],
        nx=20,
        ny=24,
        nz=75,
        dt=3600,
        ndays=200,
        fabm_yaml="fabm-pisces-4.2/testcases/fabm.yaml",
    ),
)


FABM_ROOT = Path(__file__).parent.resolve() / "fabm"
CMAKE_EXE = "cmake"
VTUNE_EXE = Path(os.environ["VTUNE_PROFILER_2024_DIR"]) / "bin64/vtune.exe"


def compile(
    root_dir: Path,
    *,
    fabm_dir: str = "fabm3",
    build_type: str = "RelWithDebInfo",
    clear: bool = True,
    extra_args: list[str] = [],
    extra_compiler_flags: list[str] = [],
    extra_release_flags: list[str] = ["/QxHost"],
    build_dir: Optional[Path] = None,
) -> Path:

    if build_dir is None:
        build_dir = Path("build")
    build_dir = (root_dir / build_dir).resolve()
    if clear and build_dir.exists():
        shutil.rmtree(build_dir)
    env = os.environ.copy()
    env["LDFLAGS"] = f"/STACK:{32 * 1024 * 1024}"
    env["FFLAGS"] = " ".join(extra_compiler_flags + ["/Qdiag-disable:10448"])
    subprocess.check_call(
        [
            CMAKE_EXE,
            "-S",
            FABM_ROOT / fabm_dir,
            "-B",
            build_dir,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_Fortran_FLAGS_RELWITHDEBINFO_INIT={' '.join(['/debug:inline-debug-info'] + extra_release_flags)}",
            f"-DCMAKE_Fortran_FLAGS_RELEASE_INIT={' '.join(extra_release_flags)}",
            f"-DCMAKE_Fortran_FLAGS_DEBUG_INIT=/check:all",
        ]
        + extra_args,
        cwd=root_dir,
        env=env,
    )
    subprocess.check_call(
        [
            CMAKE_EXE,
            "--build",
            build_dir,
            "--config",
            build_type,
            "--target",
            "test_host",
        ]
    )
    return build_dir / build_type / "test_host.exe"


def profile(
    exe: Path,
    *,
    root_dir: Path = Path("."),
    exp_name: str = "r000hs",
    extra_args: list[str] = [],
) -> Path:
    full_dir = (root_dir / exp_name).resolve()
    # return full_dir
    if full_dir.exists():
        shutil.rmtree(full_dir)
    subprocess.run(
        [
            VTUNE_EXE,
            "-collect",
            "hotspots",
            "-knob",
            "sampling-mode=sw",
            "-knob",
            "enable-stack-collection=true",
            "-search-dir",
            r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib",
            "-finalization-mode=full",
            "-r",
            full_dir,
            "--",
            exe,
            "-s",
        ]
        + extra_args,
        check=True,
        cwd=root_dir,
    )
    return full_dir


def timeit_exe(
    exe: Path,
    *,
    root_dir: Path = Path("."),
    number: int = 1,
    extra_args: list[str] = [],
) -> float:
    def run():
        subprocess.run(
            [exe, "-s"] + extra_args,
            check=True,
            cwd=root_dir,
            creationflags=subprocess.HIGH_PRIORITY_CLASS,
        )

    t = timeit.timeit(run, number=number)
    return t / number


class Node:
    def __init__(self, name: str, time: float, parent: Optional["Node"] = None):
        self.name = name
        self.time = time
        self.children: list[Node] = []
        self.parent = parent
        if parent is not None:
            parent.children.append(self)

    def find(self, name: str) -> Optional["Node"]:
        if self.name == name:
            return self
        for child in self.children:
            n = child.find(name)
            if n is not None:
                return n
        return None

    def collect_endpoints(self, skip_names: Container[str] = ()) -> list["Node"]:
        result: list[Node] = []
        for child in self.children:
            if (
                child.name not in skip_names
                and not child.name.startswith("FABM_WORK_mp_")
                and not child.name.startswith("_")
                and child.time > 0.0
            ):
                result.append(child)
            else:
                result.extend(child.collect_endpoints(skip_names))
        return result


def analyze(dir: Path, title: Optional[str] = None, note: Optional[str] = None):
    p = subprocess.run(
        [
            VTUNE_EXE,
            "-report",
            "top-down",
            "-limit",
            "500",
            "-format",
            "csv",
            "-r",
            dir,
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    # Build tree from indented names
    df = pd.read_csv(io.StringIO(p.stdout), sep="\t")
    stack = []
    for irow, row in df.iterrows():
        name, time = row[:2]
        shortname = name.lstrip()
        depth = len(name) - len(shortname)
        while depth < len(stack):
            stack.pop()
        parent = stack[-1] if stack else None
        node = Node(shortname, time, parent)
        print("  " * depth + f"{shortname}: {time:.3f} %")
        stack.append(node)
    tree = stack[0]

    top_names = [
        "FABM_mp_PROCESS_JOB_EVERYWHERE",  # =prepare_inputs
        "FABM_mp_GET_INTERIOR_SOURCES_RHS",
        "FABM_mp_GET_SURFACE_SOURCES",
        "FABM_mp_GET_BOTTOM_SOURCES_RHS",
        "FABM_mp_GET_VERTICAL_MOVEMENT",
        "FABM_mp_CHECK_INTERIOR_STATE",
        "FABM_mp_CHECK_SURFACE_STATE",
        "FABM_mp_CHECK_BOTTOM_STATE",
        "FABM_mp_FINALIZE_OUTPUTS",
    ]
    top_names.reverse()

    # Find known top-level routines
    top_nodes = []
    for name in top_names:
        node = tree.find(name)
        if node is not None and node.parent is not None:
            node.parent.children.remove(node)
            top_nodes.append(node)
    total = sum(node.time for node in top_nodes)
    print(f"Total in known top-level routines: {total:.3f} %")
    for node in top_nodes:
        print(f"  {node.name}: {node.time:.3f} %")

    # Intermediate (FABM) routines below top-level, which we want to ignore
    skip_names = [
        "FABM_WORK_mp_PROCESS_INTERIOR_SLICE",
        "FABM_WORK_mp_PROCESS_HORIZONTAL_SLICE",
        "FABM_WORK_mp_PROCESS_VERTICAL_SLICE",
        "FABM_WORK_mp_PROCESS_GLOBAL",
        "FABM_mp_PROCESS_JOB_EVERYWHERE",
        "FABM_mp_PROCESS_JOB",
        "PROCESS_JOB",
        "intel_fast_memcpy",
        "FABM_TYPES_mp_BASE_DO_COLUMN",  # forwarding to get_light
    ]

    tab20c = plt.color_sequences["tab10"]
    for i, node in enumerate(top_nodes):
        print(f"{pretty_name(node.name)}: {node.time:.3f} %")
        node.color = tab20c[i % len(tab20c)]
        assert len(node.color) == 3
        node.endpoints = node.collect_endpoints(skip_names)
        endpoint_total = sum(child.time for child in node.endpoints)
        remaining = node.time - endpoint_total
        if remaining > 0.0:
            node.endpoints.append(Node("FABM", remaining))
        for j, child in enumerate(node.endpoints):
            assert not hasattr(child, "color")
            child.color = node.color + (1.0 - 0.75 * (j + 0.5) / len(node.endpoints),)
            print(
                f"  {pretty_name(child.name)}: {child.time:.3f} % ({child.time / node.time:.1%})"
            )

    fig = plot(top_nodes, title=title, note=note)
    png = dir / "profile.png"
    fig.savefig(png, dpi=300)
    print(f"Profile plot saved to {png}")


def pretty_name(name: str) -> str:
    if name.startswith("FABM_mp_"):
        name = name[8:]
    if name.endswith("_RHS"):
        name = name[:-4]
    name = name.replace("_mp_", ":")
    return {"PROCESS_JOB_EVERYWHERE": "PREPARE_INPUTS"}.get(name, name).lower()


def plot(
    top_nodes: list[Node], title: Optional[str] = None, note: Optional[str] = None
):
    fig, ax = plt.subplots(figsize=(10, 12))
    from matplotlib.patches import Rectangle

    x = 0.4
    width = 0.1
    x_text = x + width + 0.05

    y = 0
    texts: list[matplotlib.text.Text] = []
    y_targets: list[float] = []
    for tgt in top_nodes:
        for node in tgt.endpoints:  # sorted(tgt, key=lambda x: -x[1])):
            dy = node.time / 100.0
            ax.add_artist(Rectangle((x, y), width, dy, color=node.color, ec=tgt.color))
            if node.time >= 0.05:
                y_mid = y + 0.5 * dy
                # ax.add_artist(Line2D([x + 0.2, 0.5], [y + 0.5 * dy, y_text], color=c))
                # ax.annotate(pretty_name(n), (x + 0.2, y + 0.5 * dy), (.5, y_text), va="center", arrowprops=dict(arrowstyle="-", color=top_color, relpos=(0.,0.5)), color=top_color)
                t = ax.text(
                    x_text,
                    y_mid,
                    f"{node.time:.1f}% {pretty_name(node.name)}",
                    va="center",
                    color=tgt.color,
                    ha="left",
                    fontdict=dict(size=10),
                )
                y_targets.append(y_mid)
                texts.append(t)
            y += dy

    def shift_text(x, x_text, y, dy):
        newtexts, patches = adjustText.adjust_text(
            texts,
            only_move="y",
            ax=ax,
            ensure_inside_axes=False,
            expand_axes=False,
            min_arrow_len=1000,
            avoid_self=False,
            prevent_crossings=False,
            force_explode=0,
        )  # arrowprops=dict(arrowstyle="-", color=top_color, relpos=(0.,0.5)) max_move=(0,10),

        for t, ty in zip(newtexts, y_targets):
            ax.annotate(
                "",
                xy=(x, ty),
                xytext=(x_text, t.get_position()[1]),
                arrowprops=dict(arrowstyle="-", color=t.get_color(), relpos=(0.0, 0.5)),
                va="center",
                ha="left",
            )

    shift_text(x + width, x_text, texts, y_targets)

    texts.clear()
    y_targets.clear()
    y = 0
    width = 0.01
    x_text = x - width - 0.05
    for node in top_nodes:
        dy = node.time / 100.0
        ax.add_artist(
            Rectangle((x - width, y), width, dy, color=node.color, ec=node.color)
        )
        y_mid = y + 0.5 * dy
        t = ax.text(
            x_text,
            y_mid,
            f"{pretty_name(node.name)} {node.time:.1f}%",
            va="center",
            color=node.color,
            ha="right",
        )
        texts.append(t)
        y_targets.append(y_mid)
        y += dy
    shift_text(x - width, x_text, texts, y_targets)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title, loc="left")
    if note is not None:
        fig.text(0.01, 0.01, note, ha="left", va="bottom")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=models.keys())
    parser.add_argument("exp", default="exp", nargs="?")
    args = parser.parse_args()
    target = models[args.model]

    compiler_flags = []
    # compiler_flags += ["/Ob0"]
    # compiler_flags += ["/check=all"]

    root_dir = Path(args.model)
    exe = compile(root_dir, extra_args=target.cmake_args, build_type="Release")
    tim = timeit_exe(exe, extra_args=target.simulate_args, root_dir=root_dir)

    exe = compile(
        root_dir, extra_args=target.cmake_args, extra_compiler_flags=compiler_flags
    )
    exp_dir = profile(
        exe, root_dir=root_dir, exp_name=args.exp, extra_args=target.simulate_args
    )
    analyze(
        exp_dir,
        title=f"{args.model.upper()} runtime: {target.ndays} days in {tim:.1f} s",
        note=f"cmake args: {' '.join(target.cmake_args)}\nsimulate args: {' '.join(target.simulate_args)}",
    )
