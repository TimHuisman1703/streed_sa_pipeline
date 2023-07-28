import os
import shutil
from step_4_generate_settings import parse_settings
from subprocess import Popen, PIPE
import time
from utils import get_feature_meanings
from utils import DIRECTORY, BINARY_DIRECTORY

EXEC_PATH = f"{DIRECTORY}/streed2/out/build/x64-Release/STREED.exe"
STREED_DIRECTORY = f"{DIRECTORY}/streed2/data/survival-analysis"

TIME = 600

def make_streed_compatible(input_path, output_path):
    f = open(input_path)
    lines = f.read().strip().split("\n")
    f.close()

    feature_names = lines[0].split(",")[2:]
    new_lines = [line.replace(",", " ") for line in lines[1:]]

    f = open(output_path, "w")
    f.write("\n".join(new_lines))
    f.close()

    return feature_names

def run_streed(parameters):
    args = []
    for key, value in parameters.items():
        if key == "test-file":
            continue
        args.append(f"-{key}")
        args.append(f"{value}")

    if TIME > 0:
        args.extend(["-time", str(TIME)])
    args.extend([
        "-task", "survival-analysis",
        "-use-lower-bound", "false",
        *["-train-test-split", "0.25"] * (parameters["mode"] == "hyper"),
    ])

    print(f"\033[35;1m{EXEC_PATH} {' '.join(args)}\033[0m")

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
        return -1, None

    return time, tree

def serialize_tree_with_features(tree, feature_names, feature_meanings):
    if len(tree) == 1:
        return tree
    else:
        feature, left, right = tree
        feature_name = feature_names[feature]
        feature_meaning = feature_meanings.get(feature_name, f"lambda x: x[\"{feature_name}\"]")

        left_child = serialize_tree_with_features(left, feature_names, feature_meanings)
        right_child = serialize_tree_with_features(right, feature_names, feature_meanings)
        return f"[{feature_meaning},{left_child},{right_child}]"

def main():
    total_start_time = time.time()

    shutil.rmtree(STREED_DIRECTORY)
    for section in ["", "/train", "/test"]:
        os.mkdir(f"{STREED_DIRECTORY}{section}")

    params_settings = parse_settings(f"{DIRECTORY}/output/settings.txt")

    results = []
    try:
        for params in params_settings:
            train_filename = params["file"]
            train_path = f"{BINARY_DIRECTORY}/{train_filename}.txt"
            params["file"] = train_path.replace(BINARY_DIRECTORY, STREED_DIRECTORY)
            test_filename = params["test-file"]
            test_path = f"{BINARY_DIRECTORY}/{test_filename}.txt"
            params["test-file"] = test_path.replace(BINARY_DIRECTORY, STREED_DIRECTORY)

            feature_names = make_streed_compatible(train_path, params["file"])
            make_streed_compatible(test_path, params["test-file"])

            time_duration, tree = run_streed(params)

            if time_duration >= 0:
                print(f"\033[33;1m{tree}\033[0m")
                print(f"\033[34mTime: \033[1m{time_duration:.4f}\033[0;34m seconds")
            else:
                print(f"\033[31;1mOut of time: {time_duration:.3f} seconds\033[0m")

            feature_meanings = get_feature_meanings("binary", train_filename)
            tree = serialize_tree_with_features(eval(tree), feature_names, feature_meanings)

            results.append((params, time_duration, tree))

            params["file"] = train_filename
            params["test-file"] = test_filename
    except KeyboardInterrupt:
        print("\033[33;1mHalted program!\033[0m")

    f = open(f"{DIRECTORY}/output/streed_trees.csv", "w")
    f.write("id;settings;time;tree\n")

    for i, data in enumerate(results):
        f.write(f"{i};" + ";".join(str(j) for j in data) + "\n")
    f.close()

    total_end_time = time.time()
    print(f"\033[34mTotal time: \033[1m{total_end_time - total_start_time:.4f}\033[0;34m seconds")

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()