import os
import subprocess
import io
import shutil
import timeit

import pandas as pd
import matplotlib.pyplot as plt
import adjustText

root = "ecosmo"
extra_args = ["-DFABM_NERSC_BASE=nersc", "-DFABM_ERSEM_BASE=ersem"]
host = "hycom"
dt = 600
nx = 15
ny = 14
nz = 50

root = "bfm"
extra_args = ["-DFABM_OGS_BASE=ogs", "-DFABM_SPECTRAL_BASE=spectral", "-DFABM_EXTRA_INSTITUTES=spectral"]
host = "nemo"
dt = 360
nx = 16
ny = 15
nz = 124

args = [
    "--nx",
    str(nx),
    "--ny",
    str(ny),
    "--nz",
    str(nz),
    "--dt",
    str(dt),
    "-n",
    str(int(round(30 * 3600 * 24 / dt))),
    "--nomask",
]

flags = "/Qdiag-disable:10448"
#flags += " /check=all"

vtune_exe = os.path.join(os.environ["VTUNE_PROFILER_2024_DIR"], "bin64", "vtune.exe")


def compile(fabm_dir="fabm3", build_type="RelWithDebInfo", clear=True):
    build_dir = os.path.join(os.getcwd(), root, "build")
    if clear and os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    env = os.environ.copy()
    env["LDFLAGS"] = f"/STACK:{32 * 1024 * 1024}"
    env["FFLAGS"] = flags
    subprocess.run(
        [
            "cmake",
            "-S",
            os.path.join("..", "fabm", fabm_dir),
            "-B",
            "build",
            f"-DFABM_HOST={host}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-DCMAKE_Fortran_FLAGS_RELWITHDEBINFO_INIT=/debug:inline-debug-info /QxHost",
        ]
        + extra_args,
        check=True,
        cwd=root,
        env=env,
    )
    subprocess.run(
        ["cmake", "--build", "build", "--config", build_type, "--target", "test_host"],
        cwd=root,
        check=True,
    )
    return os.path.join(build_dir, build_type, "test_host.exe")


def profile(exe, dir="r000hs"):
    full_dir = os.path.join(os.getcwd(), root, dir)
    #return full_dir
    if os.path.exists(full_dir):
        shutil.rmtree(full_dir)
    subprocess.run(
        [
            vtune_exe,
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
        + args,
        check=True,
        cwd=root,
    )
    return full_dir


def timeit_exe(exe, number=1):
    def run():
        subprocess.run([exe, "-s"] + args, check=True, cwd=root)

    t = timeit.timeit(run, number=number)
    return t / number

def analyze(dir):
    p = subprocess.run(
        [
            vtune_exe,
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
        cwd=root,
        stdout=subprocess.PIPE,
        text=True,
    )

    class Node:
        def __init__(self, name, time):
            self.name = name
            self.time = time
            self.children = []

        def find(self, name):
            if self.name == name:
                return self
            for child in self.children:
                n = child.find(name)
                if n is not None:
                    return n
            return None

        def collect_endpoints(self, skip_names=()):
            result = []
            for child in self.children:
                if (
                    child.name not in skip_names
                    and not child.name.startswith("_")
                    and child.time > 0.0
                ):
                    result.append(child)
                else:
                    result.extend(child.collect_endpoints(skip_names))
            return result

    # Build tree from indented names
    df = pd.read_csv(io.StringIO(p.stdout), sep="\t")
    stack = []
    for irow, row in df.iterrows():
        name, time = row[:2]
        shortname = name.lstrip()
        depth = len(name) - len(shortname)
        node = Node(shortname, time)
        print("  " * depth + f"{shortname}: {time:.3f} %")
        while depth < len(stack):
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        stack.append(node)
    tree = stack[0]

    top_names = [
        "FABM_mp_PROCESS_JOB_EVERYWHERE",
        "FABM_mp_GET_INTERIOR_SOURCES_RHS",
        "FABM_mp_GET_SURFACE_SOURCES",
        "FABM_mp_GET_BOTTOM_SOURCES_RHS",
        "FABM_mp_CHECK_INTERIOR_STATE",
        "FABM_mp_CHECK_SURFACE_STATE",
        "FABM_mp_CHECK_BOTTOM_STATE",
        "FABM_mp_FINALIZE_OUTPUTS",
    ]

    # Find known top-level routines
    top_nodes = []
    for name in top_names:
        node = tree.find(name)
        if node is not None:
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
        "PROCESS_JOB",
        "intel_fast_memcpy",
        "FABM_TYPES_mp_BASE_DO_COLUMN", # forwarding to get_light
    ]

    tab20c = plt.color_sequences["tab10"]
    for i, node in enumerate(top_nodes):
        print(f"{pretty_name(node.name)}: {node.time:.3f} %")
        node.color = tab20c[i % len(tab20c)]
        node.endpoints = node.collect_endpoints(skip_names)
        endpoint_total = sum(child.time for child in node.endpoints)
        remaining = node.time - endpoint_total
        if remaining > 0.0:
            node.endpoints.append(Node("FABM", remaining))
        for j, child in enumerate(node.endpoints):
            child.color = node.color + (1.0 - 0.75 * (j + 0.5) / len(node.endpoints),)
            print(
                f"  {pretty_name(child.name)}: {child.time:.3f} % ({child.time / node.time:.1%})"
            )

    fig = plot(top_nodes)
    fig.savefig("profile.png", dpi=300)


def pretty_name(name):
    if name.startswith("FABM_mp_"):
        name = name[8:]
    if name.endswith("_RHS"):
        name = name[:-4]
    name = name.replace("_mp_", ":")
    return {"PROCESS_JOB_EVERYWHERE": "PREPARE_INPUTS"}.get(name, name).lower()


def plot(top_nodes):
    fig, ax = plt.subplots(figsize=(10, 12))
    from matplotlib.patches import Rectangle

    x = 0.4
    width = 0.1
    x_text = x + width + 0.05

    y = 0
    texts = []
    y_targets = []
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
    return fig


if __name__ == "__main__":
    exe = compile(clear=True)
    #print(timeit_exe(exe))
    dir = profile(exe)
    analyze(dir)
