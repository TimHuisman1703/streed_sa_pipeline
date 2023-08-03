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

# Recursively generates a ground truth tree that can be used to distribute the instances along over the leaf nodes
#
# depth     The depth of the tree
# splits    The splits it can still make (a range [a, b] in the case of continuous features, a set of unchosen options in the case of categorical features)
def generate_tree(depth, splits):
    # When no recursions are allowed any longer, return a random distribution
    if depth == 0:
        return DISTRIBUTIONS[np.random.randint(len(DISTRIBUTIONS))]

    # Filter only the possible splits and sample a value from a random one
    allowed_splits = [(i, j) for i, j in enumerate(splits) if len(j) > 1]
    feature, elements = allowed_splits[np.random.randint(len(allowed_splits))]

    # Determine whether the chosen feature is continuous or categorical
    cont = type(elements) != set

    # Randomly decide on a splitting value, and determine the splits allowed afterwards
    left_elements = None
    right_elements = None
    value = 0
    if cont:
        a, b = elements
        value = (b - a) * np.random.rand() + a
        left_elements = (a, value)
        right_elements = (value, b)
    else:
        value = [*elements][np.random.randint(len(elements))]
        left_elements = right_elements = {j for j in elements if j != value}

    # Recurse on the left
    splits[feature] = left_elements
    left_child = generate_tree(depth - 1, splits)
    splits[feature] = elements

    # Recurse on the right
    splits[feature] = right_elements
    right_child = generate_tree(depth - 1, splits)
    splits[feature] = elements

    return (cont, feature, value, left_child, right_child)

# Traverses the ground truth tree with an instance and return random timestamp sample from the distribution
#
# tree      The ground truth tree, as generated by `generate_tree`
# instance  A list of features, the first two being placeholder values for `time` and `event`
def traverse_tree(tree, instance):
    # In the case of a leaf node, sample a time
    if type(tree[0]) == str:
        if tree[0] == "exp":
            return np.random.exponential(tree[1])
        if tree[0] == "wei":
            return np.random.weibull(tree[1]) * tree[2]
        if tree[0] == "lgn":
            return np.random.lognormal(tree[1], tree[2])
        if tree[0] == "gam":
            return np.random.gamma(tree[1], tree[2])

    # Get the decision node data
    cont, feature, value = instance[:3]

    # Check whether the predicate holds
    pred = False
    if cont:
        pred = feature > value
    else:
        pred = feature == value

    # Traverse through the right child
    child = tree[3 + pred]
    return traverse_tree(child, instance)

# Generates a dataset according to some contraints
#
# n         The amount of instances in the dataset
# f         How many times to repeat the six preset features
# c         The fraction of instances that should be censored (between 0 and 1)
def generate_dataset(n, f, c):
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

    # Prepare the splits that can be used for each feature
    allowed_splits = []
    for _ in range(f):
        allowed_splits.append((0, 1))
        allowed_splits.append((0, 1))
        allowed_splits.append((0, 1))
        allowed_splits.append({*range(2)})
        allowed_splits.append({*"XYZ"})
        allowed_splits.append({*"ABCDE"})

    # Generate the ground truth tree
    tree = generate_tree(5, allowed_splits)

    # Figure out what `k` is needed to censor each particular instance
    ks = []
    for inst in instances:
        time = max(traverse_tree(tree, inst), 1e-9)
        u = 1 - np.random.random() ** 2

        inst[0] = time
        inst[1] = u
        ks.append(time / u)

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
            for n in [100,  200, 500, 1000, 2000, 5000, 10000]
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
