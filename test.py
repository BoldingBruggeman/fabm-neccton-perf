import argparse
import analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", default=None, nargs="?")
    args = parser.parse_args()
    if args.model is None:
        models = analyze.models.keys()
    else:
        models = [args.model]
    for name in models:
        target = analyze.models[name]
        target.ndays = 1
        exe = analyze.compile(name, extra_args=target.cmake_args, build_type="Debug")
        tim = analyze.timeit_exe(exe, extra_args=target.simulate_args, root_dir=name)
