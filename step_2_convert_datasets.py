import os
import shutil
from utils import files_in_directory, parse_line
from utils import ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY, BINARY_DIRECTORY

MAX_BINARIZATIONS_PER_FEATURE = 10

def remove_redundant_binary_features(feature_names, instances):
    # Create new instances with time and event already added
    new_feature_names = ["time", "event"]
    new_instances = [j[:2] for j in instances]

    # The binary value sequences already seen (000...000 is uninformative and therefore already included)
    seen = set([
        "0" * (len(instances[0]) - 1)
    ])

    for i in range(2, len(instances[0])):
        values = [inst[i] for inst in instances]

        # Check whether the variable is binary
        if len(set(values) - set([0, 1])) == 0:
            key = "".join(str(inst[i]) for inst in instances)
            complement_key = "".join(str(1 - inst[i]) for inst in instances)

            # Check whether this binary value sequence already exists in the dataset
            # If so, skip it
            if key in seen or complement_key in seen:
                continue
            seen.add(key)

        # Leave the variable in the dataset
        new_feature_names.append(feature_names[i])
        for k in range(len(new_instances)):
            new_instances[k].append(instances[k][i])

    return new_feature_names, new_instances

def turn_numeric(feature_names, instances):
    new_feature_names = ["time", "event"]
    new_instances = [j[:2] for j in instances]
    new_feature_meanings = {}

    converted_features_amount = 0
    for i in range(2, len(instances[0])):
        values = set([inst[i] for inst in instances])

        # Check whether this variable is not a numeric variable
        if any(type(j) == str for j in values):
            values = sorted([str(j) for j in values])

            # If there are only two options, comparing to one of them will be enough
            if len(values) == 2:
                values = [values[0]]

            for option in values:
                # Create binary variables using this option
                for k in range(len(instances)):
                    new_instances[k].append(int(instances[k][i] == option))

                # Save data
                new_feature_name = f"CatFeat{converted_features_amount}"
                new_feature_names.append(new_feature_name)
                new_feature_meanings[new_feature_name] = f"lambda x: x['{feature_names[i]}'] == '{option}'"
                converted_features_amount += 1
        else:
            # Leave the variable for what it is
            new_feature_names.append(feature_names[i])
            for k in range(len(new_instances)):
                new_instances[k].append(instances[k][i])

    return new_feature_names, new_instances, new_feature_meanings

def turn_binary(feature_names, instances):
    new_feature_names = ["time", "event"]
    new_instances = [j[:2] for j in instances]
    new_feature_meanings = {}

    converted_features_amount = 0
    for i in range(2, len(instances[0])):
        values = set([inst[i] for inst in instances])

        # Check whether this variable is not a binary variable
        if values - set([0, 1]):
            values = sorted([j for j in values])

            # Create a list of thresholds between values
            thresholds = [(values[j] + values[j + 1]) / 2 for j in range(len(values) - 1)]
            if len(thresholds) > MAX_BINARIZATIONS_PER_FEATURE:
                # Reduce the amount of thresholds to a certain maximum
                indices = [round(((j + 0.5) * (len(thresholds) - 1)) / MAX_BINARIZATIONS_PER_FEATURE) for j in range(MAX_BINARIZATIONS_PER_FEATURE)]
                thresholds = [thresholds[j] for j in indices]

            for threshold in thresholds:
                # Create binary variables using this threshold
                for k in range(len(instances)):
                    new_instances[k].append(int(instances[k][i] > threshold))

                # Save data
                new_feature_name = f"NumFeat{converted_features_amount}"
                new_feature_names.append(new_feature_name)
                new_feature_meanings[new_feature_name] = f"lambda x: x['{feature_names[i]}'] > {threshold}"
                converted_features_amount += 1
        else:
            # Leave the variable for what it is
            new_feature_names.append(feature_names[i])
            for k in range(len(new_instances)):
                new_instances[k].append(instances[k][i])

    return new_feature_names, new_instances, new_feature_meanings

def main():
    for output_directory in [NUMERIC_DIRECTORY, BINARY_DIRECTORY]:
        # Remove all previously converted datasets
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        for filename in files_in_directory(output_directory):
            os.remove(f"{output_directory}/{filename}")

        # Create a folder for metadata
        if os.path.exists(f"{output_directory}/feature_meanings"):
            shutil.rmtree(f"{output_directory}/feature_meanings")
        os.mkdir(f"{output_directory}/feature_meanings")

    for input_filename in files_in_directory(ORIGINAL_DIRECTORY):
        name = input_filename[:-4]

        # Read instances from file
        f = open(f"{ORIGINAL_DIRECTORY}/{input_filename}")
        lines = f.read().strip().split("\n")
        f.close()

        instances = []
        feature_names = []
        feature_names = lines[0].split(",")
        for line in lines[1:]:
            inst = parse_line(line)
            instances.append(inst)

        # Convert to numeric instances
        numeric_feature_names, numeric_instances, numeric_feature_meanings = turn_numeric(feature_names, instances)
        numeric_feature_names, numeric_instances = remove_redundant_binary_features(numeric_feature_names, numeric_instances)

        # Write numeric instances
        f = open(f"{NUMERIC_DIRECTORY}/{name}.txt", "w")
        f.write(",".join(numeric_feature_names))
        f.write("\n")
        for num_inst in numeric_instances:
            f.write(",".join(str(j) for j in num_inst))
            f.write("\n")
        f.close()

        # Write numeric feature meanings
        f = open(f"{NUMERIC_DIRECTORY}/feature_meanings/{name}.txt", "w")
        for key, value in numeric_feature_meanings.items():
            f.write(f"{key} = {value}")
            f.write("\n")
        f.close()

        # Convert to binary instances
        binary_feature_names, binary_instances, binary_feature_meanings = turn_binary(numeric_feature_names, numeric_instances)
        binary_feature_names, binary_instances = remove_redundant_binary_features(binary_feature_names, binary_instances)
        for key, value in numeric_feature_meanings.items():
            binary_feature_meanings[key] = value

        # Write binary instances
        f = open(f"{BINARY_DIRECTORY}/{name}.txt", "w")
        f.write(",".join(binary_feature_names))
        f.write("\n")
        for bin_inst in binary_instances:
            f.write(",".join(str(j) for j in bin_inst))
            f.write("\n")
        f.close()

        # Write binary feature meanings
        f = open(f"{BINARY_DIRECTORY}/feature_meanings/{name}.txt", "w")
        for key, value in binary_feature_meanings.items():
            f.write(f"{key} = {value}")
            f.write("\n")
        f.close()

        # Print progress
        print(f"\033[35mConverted \033[1m{str(input_filename)}\033[0;35m (\033[1m{len(binary_instances)}\033[0;35m instances)\033[0m")
        print(f"\033[34m  - Original    \033[1m{len(instances[0]) - 2}\033[0;34m features\033[0m")
        print(f"\033[34m  - Numeric     \033[1m{len(numeric_instances[0]) - 2}\033[0;34m features\033[0m")
        print(f"\033[34m  - Binary      \033[1m{len(binary_instances[0]) - 2}\033[0;34m features\033[0m")

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()