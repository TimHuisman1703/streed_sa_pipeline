import numpy as np
import os
import shutil
from step_1_generate_datasets import ORIGINAL_DIRECTORY
from step_2_binarize_datasets import NUMERIC_DIRECTORY, BINARY_DIRECTORY
from utils import files_in_directory, parse_line

DIRECTORY = os.path.realpath(os.path.dirname(__file__))

SEED = 4136121025
np.random.seed(SEED)

K = 5

def main():
    for directory in [ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY, BINARY_DIRECTORY]:
        for section in ["train", "test"]:
            path = f"{directory}/{section}"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)

    for filename in [j[:-4] for j in files_in_directory(ORIGINAL_DIRECTORY)]:
        f = open(f"{ORIGINAL_DIRECTORY}/{filename}.txt")
        lines = f.read().strip().split("\n")
        f.close()

        instances = []
        for line in lines[1:]:
            inst = parse_line(line)
            instances.append(inst)
        f.close()

        indices_per_event = [[], []]
        for i in range(len(instances)):
            indices_per_event[instances[i][1]].append(i)
        
        partitions = [set() for _ in range(K)]
        for event in range(2):
            curr_indices = indices_per_event[event]
            np.random.shuffle(curr_indices)
            curr_partitions = [{*curr_indices[j * len(curr_indices) // K:(j + 1) * len(curr_indices) // K]} for j in range(K)]
            for i in range(K):
                partitions[i] |= curr_partitions[i]

        for i, partition in enumerate(partitions):
            for directory in [ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY, BINARY_DIRECTORY]:
                f = open(f"{directory}/{filename}.txt")
                lines = f.read().strip().split("\n")
                f.close()

                info_line = lines[0] + "\n"
                data_lines = lines[1:]

                train_lines, test_lines = [], []
                for j, line in enumerate(data_lines):
                    if j in partition:
                        test_lines.append(line)
                    else:
                        train_lines.append(line)

                for section, lines in [("train", train_lines), ("test", test_lines)]:
                    path = f"{directory}/{section}/{filename}_partition_{i}.txt"

                    f = open(path, "w")
                    f.write(info_line)
                    f.write("\n".join(lines))
                    f.close()

        print(f"\033[35mSplit \033[1m{filename}\033[0m")

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()
