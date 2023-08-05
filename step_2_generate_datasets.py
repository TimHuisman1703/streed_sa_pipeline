import numpy as np
import os
from utils import files_in_directory
from utils import ORIGINAL_DIRECTORY

# These are the distributions that can be used in each leaf node
#   - Exponential(lambda)
#   - Weibull(k, lambda)
#   - Lognormal(mu, sigma^2)
#   - Gamma(k, theta)
DISTRIBUTIONS = [
    ("exp", 0.3),
    ("exp", 0.4),
    ("exp", 0.6),
    ("exp", 0.8),
    ("exp", 0.9),
    ("exp", 1.15),
    ("exp", 1.5),
    ("exp", 1.8),
    ("wei", 0.8, 0.4),
    ("wei", 0.9, 0.5),
    ("wei", 0.9, 0.7),
    ("wei", 0.9, 1.1),
    ("wei", 0.9, 1.5),
    ("wei", 1.0, 1.1),
    ("wei", 1.0, 1.9),
    ("wei", 1.3, 0.5),
    ("lgn", 0.1, 1.0),
    ("lgn", 0.2, 0.75),
    ("lgn", 0.3, 0.3),
    ("lgn", 0.3, 0.5),
    ("lgn", 0.3, 0.8),
    ("lgn", 0.4, 0.32),
    ("lgn", 0.5, 0.3),
    ("lgn", 0.5, 0.7),
    ("gam", 0.2, 0.75),
    ("gam", 0.3, 1.3),
    ("gam", 0.3, 2.0),
    ("gam", 0.5, 1.5),
    ("gam", 0.8, 1.0),
    ("gam", 0.9, 1.3),
    ("gam", 1.4, 0.9),
    ("gam", 1.5, 0.7),
]

SEED = 4136121025
np.random.seed(SEED)

# Recursively splits a set of instances and assigns times to them
# If successful it returns true, else false
# This algorithm is based on the algorithm in "Optimal Survival Trees" (Bertsimas et al.), Algorithm 1
#
# depth         The depth of the tree
# instances     The instances to be given a time
# min_bucket    The minimum number of instances that need to be in a leaf
# depth_margin  The amount of layers a subtree is allowed to skim off, in the event that it has too little instances to split
def generate_tree(depth, instances, min_bucket, depth_margin):
    # If there are too little instances, this tree is not okay
    if len(instances) < min_bucket << depth:
        if depth > 0 and depth_margin > 0:
            return generate_tree(depth - 1, instances, min_bucket, depth_margin - 1)
        return -1

    # When no recursions are allowed any longer, use a random distribution to apply to the remaining nodes
    if depth == 0:
        dist = DISTRIBUTIONS[np.random.randint(len(DISTRIBUTIONS))]
        for inst in instances:
            if dist[0] == "exp":
                inst[0] = np.random.exponential(dist[1])
            if dist[0] == "wei":
                inst[0] = np.random.weibull(dist[1]) * dist[2]
            if dist[0] == "lgn":
                inst[0] = np.random.lognormal(dist[1], dist[2])
            if dist[0] == "gam":
                inst[0] = np.random.gamma(dist[1], dist[2])
        return 1

    features = [*range(len(instances[0]) - 2)]
    np.random.shuffle(features)

    for f in features:
        predicates = []
        values = [*{inst[f + 2] for inst in instances}]
        np.random.shuffle(values)
        if type(instances[0][f + 2]) == str:
            predicates = [lambda x: x[f + 2] == v for v in values]
        else:
            predicates = [lambda x: x[f + 2] >= v for v in values]

        for pred in predicates:
            left_instances = []
            right_instances = []
            for inst in instances:
                if not pred(inst):
                    left_instances.append(inst)
                else:
                    right_instances.append(inst)

            left_num_nodes = generate_tree(depth - 1, left_instances, min_bucket, depth_margin)
            if left_num_nodes == -1:
                return -1
            right_num_nodes = generate_tree(depth - 1, right_instances, min_bucket, depth_margin)
            if right_num_nodes == -1:
                return -1

            return left_num_nodes + right_num_nodes

    return generate_tree(depth - 1, instances, min_bucket, depth_margin - 1)

# Generates a dataset according to some contraints
#
# n         The amount of instances in the dataset
# f         How many times to repeat the six preset features
# c         The fraction of instances that should be censored (between 0 and 1)
def generate_dataset(n, f, c):
    while True:
        instances = []
        
        for _ in range(n):
            # Placeholder values for `time` and `event`
            instance = [0, 0]

            # Generate (a total of `f` times):
            #   3 continuous random values between 0 and 1
            #   2 boolean values
            #   A discrete value with 3 options
            #   A discrete value with 5 options
            for _ in range(f):
                instance.append(np.random.rand())
                instance.append(np.random.rand())
                instance.append(np.random.rand())
                instance.append(np.random.randint(2))
                instance.append("XYZ"[np.random.randint(3)])
                instance.append("ABCDE"[np.random.randint(5)])
            instances.append(instance)
        
        # Generate the ground truth tree and sample instance times with it
        num_nodes = generate_tree(4, instances, 1, 1)
        if num_nodes > -1:
            print(num_nodes)
            break

    # Figure out what `k` is needed to censor each particular instance
    ks = []
    for inst in instances:
        inst[0] = max(1e-9, inst[0])
        u = 1 - np.random.random() ** 2

        inst[1] = u
        ks.append(inst[0] / u)

    # Find a `k` such that `100c` percent of the instances is censored
    ks = sorted(ks)
    k = ks[int(n * (1 + 1e-9 - c))] - 1e-9

    # Apply censoring to the chosen instances
    for inst in instances:
        censor = k * inst[1]

        if censor < inst[0]:
            inst[0] = censor
            inst[1] = 0
        else:
            inst[1] = 1

    # Sort the instance based on time
    instances = sorted(instances, key=lambda x: (x[0], x[1]))

    return instances

def main():
    # The settings to generate with
    SETTINGS = [
        (n, f, c, i)
            for n in [100, 200, 500, 1000]
            for f in [1]
            for c in [10, 50, 80]
            for i in range(5)
    ]

    # Create necessary directories
    output_parent_directory = "/".join(ORIGINAL_DIRECTORY.split("/")[:-1])
    if not os.path.exists(output_parent_directory):
        os.mkdir(output_parent_directory)
    if not os.path.exists(ORIGINAL_DIRECTORY):
        os.mkdir(ORIGINAL_DIRECTORY)

    # Remove all previously generated datasets
    for filename in files_in_directory(ORIGINAL_DIRECTORY):
        if filename.startswith("generated_dataset_"):
            os.remove(f"{ORIGINAL_DIRECTORY}/{filename}")

    # Generate a dataset for each setting
    for n, f, c, i in SETTINGS:
        instances = generate_dataset(n, f, c / 100)

        filename = f"generated_dataset_{n:05}_{f}_{c}_{i}"

        file = open(f"{ORIGINAL_DIRECTORY}/{filename}.txt", "w")
        file.write("time,event," + ",".join(f"F{j}" for j in range(len(instances[0]) - 2)))
        file.write("\n")
        for inst in instances:
            file.write(",".join(str(j) for j in inst))
            file.write("\n")
        file.close()

        print(f"\033[35mCreated \033[1m{filename}\033[0m")

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()
