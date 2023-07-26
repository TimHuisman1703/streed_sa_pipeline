import os

DIRECTORY = os.path.realpath(os.path.dirname(__file__))

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

def prepare_params(params):
    if "file" not in params:
        raise Exception("Forgot to specify file")

    if "test-file" not in params:
        params["test-file"] = params["file"]

    if "max-depth" not in params:
        params["max-depth"] = 3

    if "max-num-nodes" not in params:
        params["max-num-nodes"] = "max"

    if params["max-num-nodes"] == "max":
        params["max-num-nodes"] = 2 ** params["max-depth"] - 1

    if "cost-complexity" not in params:
        params["cost-complexity"] = 0

    if "split" not in params:
        params["split"] = False
    if params.pop("split"):
        params["file"] = "train/" + params["file"]
        params["test-file"] = "test/" + params["test-file"]

        result = []
        for idx in range(5):
            p = params.copy()
            p["file"] += f"_{idx}"
            p["test-file"] += f"_{idx}"
            result.append(p)

        return result

    return [params]

def parse_settings(filename):
    f = open(filename)
    settings = [eval(j) for j in f.read().strip().split("\n")]
    f.close()

    return settings

if __name__ == "__main__":
    PARAM_OPTIONS = {
        "file": [
            f"{j[:-11]}" for j in os.listdir(f"{DIRECTORY}/streed2/data/survival-analysis")
            if j.endswith("_binary.txt") and not j.startswith("generated_dataset_")
            and os.path.isfile(f"{DIRECTORY}/streed2/data/survival-analysis/{j}")
        ],
        "max-depth": [3],
        "max-num-nodes": ["max"],
        "cost-complexity": [0],
        "min-deaths-per-leaf": [1],
        "mode": ["hyper"],
        "split": [True],
    }

    params_settings = cartesian_product([*PARAM_OPTIONS.items()])
    filtered_params_settings = []
    for params in params_settings:
        params = prepare_params(params)
        if params:
            filtered_params_settings.extend(params)

    f = open(f"{DIRECTORY}/output/settings.txt", "w")
    for params in filtered_params_settings:
        f.write(f"{params}\n".replace("'", "\""))
    f.close()

    print("\033[32;1mDone!\033[0m")
