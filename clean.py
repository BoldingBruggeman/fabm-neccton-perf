import argparse
import shutil
import analyze
import os.path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname")
    args = parser.parse_args()
    for name in analyze.models.keys():
        path = os.path.join(name, args.dirname)
        print(f"Removing {path}...")
        if os.path.exists(path):
            shutil.rmtree(path)