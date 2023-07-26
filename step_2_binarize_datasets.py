import numpy as np
import os
from utils import files_in_directory, parse_line

DIRECTORY = os.path.realpath(os.path.dirname(__file__))
INPUT_DIRECTORY = f"{DIRECTORY}/datasets"
OUTPUT_DIRECTORY = f"{DIRECTORY}/streed2/data/survival-analysis"
METADATA_DIRECTORY = f"{DIRECTORY}/metadata"

MAX_BINARIZATIONS_PER_FEATURE = 10

if __name__ == "__main__":
    # Remove all binarized datasets in STreeD
    for filename in files_in_directory(OUTPUT_DIRECTORY):
        os.remove(f"{OUTPUT_DIRECTORY}/{filename}")

    for input_filename in files_in_directory(INPUT_DIRECTORY):
        instances = []
        feature_names = []
        binary_instances = []
        binary_feature_meanings = []

        f = open(f"{INPUT_DIRECTORY}/{input_filename}")
        lines = f.read().strip()
        if "," not in lines:
            lines = lines.replace(";", ",")
        lines = lines.split("\n")
        f.close()

        feature_names = lines[0].split(",")
        for line in lines[1:]:
            inst = parse_line(line)
            instances.append(inst)
            if inst[0] <= 0:
                inst[0] = 1e-6
            binary_instances.append([inst[0], int(int(inst[1]) == 1)])

        seen_features = set(["0" * len(instances), "1" * len(instances)])
        for i in range(2, len(instances[0])):
            values = set()
            for inst in instances:
                values.add(inst[i])

            if any(type(j) == str for j in values):
                values = sorted([str(j) for j in values])

                for name in values:
                    boolean_values = []
                    for k in range(len(instances)):
                        boolean_values.append(int(instances[k][i] == name))
                    key = "".join(str(k) for k in boolean_values)
                    complement_key = "".join(str(1 - k) for k in boolean_values)

                    if key in seen_features or complement_key in seen_features:
                        continue
                    seen_features.add(key)

                    for k in range(len(instances)):
                        binary_instances[k].append(boolean_values[k])
                    binary_feature_meanings.append(f"lambda x: x['{feature_names[i]}'] == '{name}'")
            else:
                has_nan = any(np.isnan(j) for j in values)
                values = [float("nan")] * has_nan + sorted(j for j in values if not np.isnan(j))

                nan_boolean_values = []
                for k in range(len(instances)):
                    nan_boolean_values.append(int(np.isnan(instances[k][i])))
                key = "".join(str(k) for k in nan_boolean_values)

                if key not in seen_features:
                    seen_features.add(key)

                    for k in range(len(instances)):
                        binary_instances[k].append(nan_boolean_values[k])
                    binary_feature_meanings.append(f"lambda x: np.isnan(x['{feature_names[i]}'])")

                thresholds = [((values[j] + values[j + 1]) / 2, values[j + 1]) for j in range(len(values) - 1)]
                if len(thresholds) > MAX_BINARIZATIONS_PER_FEATURE:
                    indices = [round(((j + 0.5) * (len(thresholds) - 1)) / MAX_BINARIZATIONS_PER_FEATURE) for j in range(MAX_BINARIZATIONS_PER_FEATURE)]
                    thresholds = [thresholds[j] for j in indices]

                for threshold, ceil_value in thresholds:
                    boolean_values = []
                    for k in range(len(instances)):
                        boolean_values.append(int(instances[k][i] > threshold))
                    key = "".join(str(k) for k in boolean_values)

                    if key in seen_features:
                        continue
                    seen_features.add(key)

                    for k in range(len(instances)):
                        binary_instances[k].append(boolean_values[k])
                    binary_feature_meanings.append(f"lambda x: x['{feature_names[i]}'] > {threshold}")

        name = input_filename[:-4]
        f = open(f"{OUTPUT_DIRECTORY}/{name}_binary.txt", "w")
        for bin_inst in binary_instances:
            f.write(" ".join(str(j) for j in bin_inst))
            f.write("\n")
        f.close()

        if not os.path.exists(METADATA_DIRECTORY):
            os.mkdir(METADATA_DIRECTORY)

        f = open(f"{METADATA_DIRECTORY}/{name}_features.txt", "w")
        f.write("\n".join(str(j) for j in binary_feature_meanings))
        f.close()

        assert len(binary_feature_meanings) == len(binary_instances[0]) - 2
        print(f"\033[35;1mBinarized {str(input_filename)+':':<50} {len(binary_instances)} instances, {len(binary_instances[0]) - 2} features\033[0m")

    print("\033[32;1mDone!\033[0m")
