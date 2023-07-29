import numpy as np
from sksurv.metrics import integrated_brier_score
from utils import fill_tree, parse_tree, read_dataset, Tree
from utils import DIRECTORY, ORIGINAL_DIRECTORY

def calculate_concordance(root, instances):
    buckets = {}
    for inst in instances:
        theta, _ = root.traverse(inst)
        if theta not in buckets:
            buckets[theta] = [[], [], []]
        buckets[theta][inst.event].append(inst)
        buckets[theta][2].append(inst)
    buckets = [buckets[key] for key in sorted(buckets.keys())]
    for i in range(len(buckets)):
        for j in range(3):
            buckets[i][j] = sorted(buckets[i][j], key=lambda x: x.time)

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

    if cc + tr + dc:
        return (cc + 0.5 * tr) / (cc + tr + dc)
    else:
        return 1

def calculate_integrated_brier_score(root, instances):
    times = sorted({inst.time for inst in instances})
    q_10 = len(times) // 10
    q_90 = 9 * len(times) // 10
    times = times[q_10:q_90]

    train_instances = root.instances
    ibs_instances, ibs_estimates = [], []
    leaves = root.get_leaves()
    for leaf in leaves:
        leaf_instances = leaf.instances
        survival_function = leaf.distribution

        ibs_instances += leaf_instances

        estimate = []
        for t in times:
            estimate.append(survival_function(t))
        ibs_estimates.extend([estimate for _ in range(len(leaf_instances))])

    train_instances_formatted = np.array([(inst.event, inst.time) for inst in train_instances], dtype=[("event", "?"), ("time", "f4")])
    test_instances_formatted = np.array([(inst.event, inst.time) for inst in ibs_instances], dtype=[("event", "?"), ("time", "f4")])

    score = integrated_brier_score(train_instances_formatted, test_instances_formatted, ibs_estimates, times)

    return score

def main():
    for algorithm in ["ctree", "ost", "streed"]:
        print(f"\n\033[33;1mEvaluating {algorithm.upper()}'s output...\033[0m")

        f = open(f"{DIRECTORY}/output/{algorithm}_trees.csv")
        lines = f.read().strip().split("\n")
        f.close()

        new_lines = [";".join(lines[0].split(";")[:-1] + ["results"])]

        for line in lines[1:]:
            null = None
            id, settings, time_duration, flat_tree = [eval(j) for j in line.split(";")]

            train_filename = settings["file"]
            test_filename = settings["test-file"]

            results = {}

            tree = parse_tree(flat_tree)
            train_instances = fill_tree(tree, f"{ORIGINAL_DIRECTORY}/{train_filename}.txt")
            test_instances = read_dataset(f"{ORIGINAL_DIRECTORY}/{test_filename}.txt")

            results["num_nodes"] = tree.size()

            for name, filename, instances in [("train", train_filename, train_instances), ("test", test_filename, test_instances)]:
                instances = fill_tree(tree, f"{ORIGINAL_DIRECTORY}/{filename}.txt")

                tree.clear_instances()
                for inst in instances:
                    tree.traverse(inst, True)
                tree.calculate_error()
                print(tree)

                base_tree = Tree(-1, None, None, instances)
                base_tree_error = base_tree.error
                curr_tree_error = tree.error
                objective_score = 1
                if base_tree_error > 1e-6:
                    objective_score = 1 - curr_tree_error / base_tree_error
                if abs(objective_score) < 1e-6:
                    objective_score = 0

                concordance_score = calculate_concordance(tree, instances)

                # base_tree_ibs = calculate_integrated_brier_score(base_tree, instances)
                # curr_tree_ibs = calculate_integrated_brier_score(tree, instances)
                ibs_ratio = 1
                # if base_tree_ibs > 1e-6:
                #     ibs_ratio = 1 - curr_tree_ibs / base_tree_ibs
                # if abs(ibs_ratio) < 1e-6:
                #     ibs_ratio = 0

                results[name] = {
                    "objective_score": objective_score,
                    "concordance_score": concordance_score,
                    "integrated_brier_score_ratio": ibs_ratio,
                }

            info_line = ";".join(line.split(";")[:-1])
            results_line = str(results)
            new_line = f"{info_line};{results_line}"
            new_lines.append(new_line)
            print(f"\033[35;1m{info_line}\033[30;1m;\033[34;1m{results_line}\033[0m")

        f = open(f"{DIRECTORY}/output/{algorithm}_output.csv", "w")
        f.write("\n".join(new_lines))
        f.close()

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()
