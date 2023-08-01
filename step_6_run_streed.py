import os
import shutil
from step_5_generate_settings import parse_settings
from subprocess import Popen, PIPE
import time
from utils import get_feature_meanings
from utils import DIRECTORY

EXEC_PATH = f"{DIRECTORY}/streed2/out/build/x64-Release/STREED.exe"
STREED_DIRECTORY = f"{DIRECTORY}/streed2/data/survival-analysis"

TIME_OUT_IN_SECONDS = 60

DATASET_TYPE = "binary"

# Takes a comma-separated dataset file and turns it into a file that STreeD can read
# Returns the list of feature names found in the first line of the file
#
# input_path    The path to the comma-separated file
# output_path   The path to the output file
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

# Runs STreeD with a set of parameters
# Returns resulting tree and the time needed to generate it (a negative time indicates a time out)
#
# parameters    The parameters to run the algorithm with
def run_streed(parameters):
    # Convert parameters to arguments
    args = []
    for key, value in parameters.items():
        args.append(f"-{key}")
        args.append(f"{value}")

    # Add addition arguments
    if TIME_OUT_IN_SECONDS > 0:
        args.extend(["-time", str(TIME_OUT_IN_SECONDS)])
    args.extend([
        "-task", "survival-analysis",
        "-use-lower-bound", "1",
        "-use-dataset-caching", "1",
        "-use-branch-caching", "0",
    ])

    # Print executable call for convenience
    print(f"\033[35;1m{EXEC_PATH} {' '.join(args)}\033[0m")

    # Run executable
    proc = Popen([EXEC_PATH, *args], stdin=PIPE, stdout=PIPE)
    out, _ = proc.communicate()

    # Read the output from the console
    out_lines = [j.strip() for j in out.decode().split("\n")]
    time_line = [j for j in out_lines if j.startswith("CLOCKS FOR SOLVE:")][0]
    time = float(time_line.split()[-1])
    if "No tree found" in out_lines:
        return -time, "[None]"
    tree_line = [j for j in out_lines if j.startswith("Tree 0:")][0]
    tree = tree_line.split()[-1]

    return time, tree

# Turn tree structure with numbers into tree structure with lambda's
#
# tree              The tree to convert
# feature_names     The names of the features, in order
# feature_meanings  The meanings of converted binary features
def serialize_tree_with_features(tree, feature_names, feature_meanings):
    if len(tree) == 1:
        # Leaf node
        return tree
    else:
        # Decision node
        # Try to get meaning of binary variable from the feature meanings map
        # If not found, the variable was already binary, so write a simple lambda for it
        feature, left, right = tree
        feature_name = feature_names[feature]
        feature_meaning = feature_meanings.get(feature_name, f"lambda x: x[\"{feature_name}\"]")

        # Serialize children and construct tree
        left_child = serialize_tree_with_features(left, feature_names, feature_meanings)
        right_child = serialize_tree_with_features(right, feature_names, feature_meanings)
        return f"[{feature_meaning},{left_child},{right_child}]"

def main():
    total_start_time = time.time()

    # Clear STreeD-datasets, these will be replaced anyway
    shutil.rmtree(STREED_DIRECTORY)
    for section in ["", "/train", "/test"]:
        os.mkdir(f"{STREED_DIRECTORY}{section}")

    # Read settings
    params_settings = parse_settings(f"{DIRECTORY}/output/settings.txt")

    results = []
    try:
        dataset_directory = f"{DIRECTORY}/datasets/{DATASET_TYPE}"

        for params in params_settings:
            # Change files to new STreeD files
            train_filename = params["file"]
            train_path = f"{dataset_directory}/{train_filename}.txt"
            params["file"] = train_path.replace(dataset_directory, STREED_DIRECTORY)
            test_filename = params["test-file"]
            test_path = f"{dataset_directory}/{test_filename}.txt"
            params["test-file"] = test_path.replace(dataset_directory, STREED_DIRECTORY)

            # Write STreeD files
            feature_names = make_streed_compatible(train_path, params["file"])
            make_streed_compatible(test_path, params["test-file"])

            # Run STreeD
            time_duration, tree = run_streed(params)
            if time_duration >= 0:
                print(f"\033[33;1m{tree}\033[0m")
                print(f"\033[34mTime: \033[1m{time_duration:.3f}\033[0;34m seconds")
            else:
                print(f"\033[31mOut of time: \033[1m{-time_duration:.3f}\033[0;31m seconds\033[0m")

            # Parse tree string to lambda-structure
            feature_meanings = get_feature_meanings(train_filename)
            tree = serialize_tree_with_features(eval(tree), feature_names, feature_meanings)
            results.append((params, time_duration, tree))

            # Reset parameters to write to file nicely
            params["file"] = train_filename
            params["test-file"] = test_filename
    except KeyboardInterrupt:
        print("\033[33;1mHalted program!\033[0m")

    # Write trees to file
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