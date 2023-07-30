import os
from step_4_split_datasets import K
from utils import files_in_directory
from utils import DIRECTORY, ORIGINAL_DIRECTORY

if not os.path.exists(f"{DIRECTORY}/output"):
    os.mkdir(f"{DIRECTORY}/output")

def cartesian_product(items, idx = 0):
    if idx == len(items):
        return [{}]

    key, values = items[idx]
    results = []
    for value in values:
        for result in cartesian_product(items, idx + 1):
            result[key] = value
            results.append(result)
    return results

def prepare_parameters(parameters):
    if "file" not in parameters:
        raise Exception("Forgot to specify file")

    if "test-file" not in parameters:
        parameters["test-file"] = parameters["file"]

    if "max-depth" not in parameters:
        parameters["max-depth"] = 3

    if "max-num-nodes" not in parameters:
        parameters["max-num-nodes"] = "max"
    if parameters["max-num-nodes"] == "max":
        parameters["max-num-nodes"] = 2 ** parameters["max-depth"] - 1

    if "cost-complexity" not in parameters:
        parameters["cost-complexity"] = 0

    if "split" not in parameters:
        parameters["split"] = False
    if parameters.pop("split"):
        parameters["file"] = "train/" + parameters["file"]
        parameters["test-file"] = "test/" + parameters["test-file"]

        result = []
        for idx in range(K):
            p = parameters.copy()
            p["file"] += f"_partition_{idx}"
            p["test-file"] += f"_partition_{idx}"
            result.append(p)

        return result

    return [parameters]

def parse_settings(filename):
    f = open(filename)
    settings = [eval(j) for j in f.read().strip().split("\n")]
    f.close()

    return settings

def main():
    PARAM_OPTIONS = {
        "file": [
            f"{j[:-4]}" for j in files_in_directory(ORIGINAL_DIRECTORY) if not j.startswith("generated_dataset_")
        ],
        "max-depth": [3],
        "max-num-nodes": ["max"],
        "cost-complexity": [0],
        "mode": ["hyper"],
        "split": [True],
    }

    parameter_combinations = cartesian_product([*PARAM_OPTIONS.items()])
    prepared_parameter_combinations = []
    for parameters in parameter_combinations:
        parameters = prepare_parameters(parameters)
        prepared_parameter_combinations.extend(parameters)

    f = open(f"{DIRECTORY}/output/settings.txt", "w")
    for parameters in prepared_parameter_combinations:
        f.write(f"{parameters}\n".replace("'", "\""))
        print(f"\033[35;1m{parameters}\033[0m")
    f.close()

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()
