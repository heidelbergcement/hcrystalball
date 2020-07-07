"""Check notebook execution speed"""
import time
import pandas as pd
from pathlib import Path
import subprocess


def bench_notebook(filename):
    cmd = f"jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --execute {filename}"
    print(cmd)
    t = time.time()
    subprocess.call(cmd, shell=True)
    dt = time.time() - t
    return dt


def bench_all_notebooks():
    paths = list(Path("docs/examples").glob("**/*.ipynb"))
    timings = []
    for path in paths:
        time = bench_notebook(path)
        timings.append({"notebook": str(path.name), "walltime": time})

    return pd.DataFrame(timings)


if __name__ == "__main__":
    timings = bench_all_notebooks().sort_values("walltime", ascending=False).round(1)
    print(timings.to_string(index=False))
    timings.to_csv("hcrystalball_notebook_execution_speed.csv", index=False)
