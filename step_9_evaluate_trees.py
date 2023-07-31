import numpy as np
from sksurv.metrics import integrated_brier_score
from utils import fill_tree, parse_tree, read_dataset, Tree
from utils import DIRECTORY, ORIGINAL_DIRECTORY

# Calculates the Harrell's C-index (concordance) for a certain tree
#
# root          The tree to evaluate the instances with
# instances     The instances to calculate the concordance with
def calculate_concordance(root, instances):
    # Group the instances together based on their theta
    buckets = {}
    for inst in instances:
        theta, _ = root.classify(inst)
        if theta not in buckets:
            buckets[theta] = [[], [], []]
        buckets[theta][inst.event].append(inst)
        buckets[theta][2].append(inst)
    buckets = [buckets[key] for key in sorted(buckets.keys())]
    for i in range(len(buckets)):
        for j in range(3):
            buckets[i][j] = sorted(buckets[i][j], key=lambda x: x.time)

    # Apply binary search to compute the amount of discordance and concordant pairs efficiently
    cc = tr = dc = 0
    for i in range(len(buckets)):
        for j in range(len(buckets)):
            for inst_i in buckets[i][2]:
                a, b = 0, len(buckets[j][1])
                while a < b:
                    mid = (a + b) // 2
                    if buckets[j][1][mid].time < inst_i.time:
                        a = mid + 1
                    else:
                        b = mid

                if i < j:
                    cc += a
                elif i == j:
                    tr += a
                else:
                    dc += a

    # Return C-index
    if cc + tr + dc:
        return (cc + 0.5 * tr) / (cc + tr + dc)
    else:
        return 1

# Calculates the Integrated Brier Score for a certain tree
#
# root              The tree to evaluate the instances with
# train_instances   The train instances used to create the tree
# test_instances    The train instances used to test the tree
def calculate_integrated_brier_score(root, train_instances, test_instances):
    # Drop test instances that appear after the latest train instance (ideally just a few)
    max_train_time = max(inst.time for inst in train_instances)
    test_instances = [inst for inst in test_instances if inst.time < max_train_time]

    # Get the non-extreme test-instance-times
    times = sorted({inst.time for inst in test_instances})
    q_10 = len(times) // 10
    q_90 = 9 * len(times) // 10
    times = times[q_10:q_90]

    # Create survival distribution estimates for each test instance
    estimates = []
    estimates_per_leaf = {}
    for inst in test_instances:
        _, survival_distribution = root.classify(inst)

        if survival_distribution not in estimates_per_leaf:
            estimate = []
            for t in times:
                estimate.append(survival_distribution(t))
            estimates_per_leaf[survival_distribution] = estimate
        estimates.append(estimates_per_leaf[survival_distribution])

    # Format all instances to structured array
    train_instances_formatted = np.array([(inst.event, inst.time) for inst in train_instances], dtype=[("event", "?"), ("time", "f4")])
    test_instances_formatted = np.array([(inst.event, inst.time) for inst in test_instances], dtype=[("event", "?"), ("time", "f4")])

    # Use sksurv's IBS method
    score = integrated_brier_score(train_instances_formatted, test_instances_formatted, estimates, times)
    return score

def main():
    for algorithm in ["ctree", "ost", "streed"]:
        print(f"\n\033[33;1mEvaluating {algorithm.upper()}'s output...\033[0m")

        # Read trees
        f = open(f"{DIRECTORY}/output/{algorithm}_trees.csv")
        lines = f.read().strip().split("\n")
        f.close()

        new_lines = [";".join(lines[0].split(";")[:-2] + ["results"])]

        for line in lines[1:]:
            # Parse line
            null = None
            id, settings, time_duration, flat_tree = [eval(j) for j in line.split(";")]
            train_filename = settings["file"]
            test_filename = settings["test-file"]

            results = {}

            # Parse given tree and determine labels using training set
            tree = parse_tree(flat_tree)
            train_instances = fill_tree(tree, f"{ORIGINAL_DIRECTORY}/{train_filename}.txt")
            test_instances = read_dataset(f"{ORIGINAL_DIRECTORY}/{test_filename}.txt")
            base_tree = Tree(None, None, None, train_instances)

            # Store time
            results["runtime"] = time_duration

            # Count number of nodes
            results["num_nodes"] = tree.size()

            # Calculate the Integrated Brier Score ratio
            base_tree_ibs = calculate_integrated_brier_score(base_tree, train_instances, test_instances)
            curr_tree_ibs = calculate_integrated_brier_score(tree, train_instances, test_instances)
            ibs_ratio = 1
            if base_tree_ibs > 1e-6:
                ibs_ratio = 1 - curr_tree_ibs / base_tree_ibs
            if abs(ibs_ratio) < 1e-6:
                ibs_ratio = 0
            results["integrated_brier_score_ratio"] = ibs_ratio

            # For both the training set and the testing set
            for name, filename, instances in [("train", train_filename, train_instances), ("test", test_filename, test_instances)]:
                instances = fill_tree(tree, f"{ORIGINAL_DIRECTORY}/{filename}.txt")

                # Fill tree and base tree with instances
                for t in [tree, base_tree]:
                    t.clear_instances()
                    for inst in instances:
                        t.classify(inst, True)
                    t.calculate_error()

                print(tree)

                # Calculate the objective score
                base_tree_error = base_tree.error
                tree_error = tree.error
                objective_score = 1
                if base_tree_error > 1e-6:
                    objective_score = 1 - tree_error / base_tree_error
                if abs(objective_score) < 1e-6:
                    objective_score = 0

                # Calculate Harrell's C-index
                concordance_score = calculate_concordance(tree, instances)

                results[name] = {
                    "objective_score": objective_score,
                    "concordance_score": concordance_score,
                }

            # Push results on a single line
            info_line = ";".join(line.split(";")[:-2])
            results_line = str(results)
            new_line = f"{info_line};{results_line}"
            new_lines.append(new_line)
            print(f"\033[35;1m{info_line}\033[30;1m;\033[34;1m{results_line}\033[0m")

        # Write results to file
        f = open(f"{DIRECTORY}/output/{algorithm}_output.csv", "w")
        f.write("\n".join(new_lines))
        f.close()

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()
