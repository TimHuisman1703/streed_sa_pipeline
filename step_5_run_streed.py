from step_4_generate_settings import parse_settings
import os
from subprocess import Popen, PIPE
import time
from utils import get_feature_meanings

DIRECTORY = os.path.realpath(os.path.dirname(__file__))
EXEC_PATH = f"{DIRECTORY}/streed2/out/build/x64-Release/STREED.exe"

TIME = 600
PARETO_PRUNING = False

def dominates(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2] and a[3] == b[3]

def add_to_front(front, a):
    i = 0

    while i < len(front):
        if dominates(a, front[i]):
            front.pop(i)
        elif dominates(front[i], a):
            return False
        else:
            i += 1

    pareto_front.append(a)
    return True

pareto_front = []

def serialize_tree_with_features(tree, feature_meanings):
    if len(tree) == 1:
        return tree
    else:
        feature, left, right = tree
        feature_meaning = feature_meanings[feature]
        return f"[{feature_meaning},{serialize_tree_with_features(left, feature_meanings)},{serialize_tree_with_features(right, feature_meanings)}]"

def run_streed(params):
    global pareto_front

    args = []
    for key, value in params.items():
        if key == "test-file":
            continue
        args.append(f"-{key}")
        args.append(f"{value}")

    if TIME > 0:
        args.extend(["-time", str(TIME)])
    args.extend([
        "-task", "survival-analysis",
        "-use-lower-bound", "false",
        *["-train-test-split", "0.25"] * (params["mode"] == "hyper"),
    ])

    print(f"\033[35;1m{EXEC_PATH} {' '.join(args)}\033[0m")

    pareto_key = None
    if PARETO_PRUNING:
        pareto_key = [int(j) for j in params['file'].split("_")[-5:-1]]
        pareto_key[2] = params["max-depth"]

        if any(dominates(i, pareto_key) for i in pareto_front):
            print(f"\033[30;1mPareto dominated\033[0m")
            return (-1, -1, -1)

    proc = Popen([EXEC_PATH, *args], stdin=PIPE, stdout=PIPE)
    out, _ = proc.communicate()

    out_lines = [j.strip() for j in out.decode().split("\n")]
    time_line = [j for j in out_lines if j.startswith("CLOCKS FOR SOLVE:")][0]
    tree_line = [j for j in out_lines if j.startswith("Tree 0:")][0]

    time = float(time_line.split()[-1])
    tree = None
    try:
        tree = tree_line.split()[-1]
    except:
        if PARETO_PRUNING:
            add_to_front(pareto_front, pareto_key)

        print(f"\033[31;1mOut of time: {time:.3f} seconds\033[0m")
        return -1, None

    print(tree)
    print(f"\033[34;1mTime: {time:.3f} seconds\033[0m")
    return time, tree

if __name__ == "__main__":
    total_start_time = time.time()

    params_settings = parse_settings(f"{DIRECTORY}/output/settings.txt")

    results = []
    try:
        for params in params_settings:
            filename = params["file"]
            params["file"] = f"{DIRECTORY}/streed2/data/survival-analysis/{filename}_binary.txt"
            test_filename = params["test-file"]
            params["test-file"] = f"{DIRECTORY}/streed2/data/survival-analysis/{test_filename}_binary.txt"

            time_duration, tree = run_streed(params)

            feature_meanings = get_feature_meanings(filename)
            tree = serialize_tree_with_features(eval(tree), feature_meanings)

            results.append((params, time_duration, tree))

            params["file"] = filename
            params["test-file"] = test_filename
    except KeyboardInterrupt:
        print("\033[33;1mHalted program!\033[0m")

    f = open(f"{DIRECTORY}/output/streed_trees.csv", "w")
    f.write("id;settings;time;tree\n")
    print()
    for i, data in enumerate(results):
        f.write(f"{i};" + ";".join(str(j) for j in data) + "\n")
    f.close()

    total_end_time = time.time()
    print(f"\033[34;1mTotal time: {total_end_time - total_start_time:.4f} seconds")

    print("\033[32;1mDone!\033[0m")
