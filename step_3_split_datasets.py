import numpy as np
import os
import shutil
from utils import files_in_directory

DIRECTORY = os.path.realpath(os.path.dirname(__file__))
ORIGINAL_DIRECTORY = f"{DIRECTORY}/datasets"
BINARY_DIRECTORY = f"{DIRECTORY}/streed2/data/survival-analysis"

K = 5

SEED = 4136121025
np.random.seed(SEED)

if __name__ == "__main__":
    for directory in [ORIGINAL_DIRECTORY, BINARY_DIRECTORY]:
        for type in ["train", "test"]:
            path = f"{directory}/{type}"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)

    for filename in [j[:-4] for j in files_in_directory(ORIGINAL_DIRECTORY)]:
        f = open(f"{ORIGINAL_DIRECTORY}/{filename}.txt")
        n = f.read().strip().count("\n")
        f.close()

        indices = [*range(n)]
        np.random.shuffle(indices)
        partitions = [{*indices[j * len(indices) // K:(j + 1) * len(indices) // K]} for j in range(K)]

        for i, partition in enumerate(partitions):
            for binary in [False, True]:
                directory = [ORIGINAL_DIRECTORY, BINARY_DIRECTORY][binary]

                f = open(f"{directory}/{filename}{'_binary' * binary}.txt")
                lines = f.read().strip().split("\n")
                f.close()

                info_line = [f"{lines[0]}\n", ""][binary]
                data_lines = lines[not binary:]

                train_lines, test_lines = [], []
                for j, line in enumerate(data_lines):
                    if j in partition:
                        test_lines.append(line)
                    else:
                        train_lines.append(line)

                for type, lines in [("train", train_lines), ("test", test_lines)]:
                    path = f"{directory}/{type}/{filename}_{i}{'_binary' * binary}.txt"

                    f = open(path, "w")
                    f.write(info_line)
                    f.write("\n".join(lines))
                    f.close()

        print(f"\033[35;1mSplit {filename}\033[0m")

    print("\033[32;1mDone!\033[0m")
