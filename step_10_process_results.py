import matplotlib.pyplot as plt
import numpy as np
import os

DIRECTORY = os.path.realpath(os.path.dirname(__file__))

if not os.path.exists(f"{DIRECTORY}/plots"):
    os.mkdir(f"{DIRECTORY}/plots")

ALG_INFO = {
    "ctree": ("CTree", "#1F7FDF"),
    "ost": ("OST", "#DF7F1F"),
    "streed": ("STreeD", "#1FBF1F"),
}
TRAIN_TEST_SCORE_TYPES = [
    ("Objective score", "objective_score"),
    ("Concordance score", "concordance_score"),
]

# Makes plots for each of the algorithms by sorting a list of outcomes and laying it out over a [0, 1]-range
#
# data  The output data for each of the algorithms
def plot_sorted_scores(data):
    # Plot num nodes
    plt.clf()
    plt.title(f"Number of nodes")
    for alg, alg_data in data.items():
        title, color = ALG_INFO[alg]
        ys = sorted([j["results"][f"num_nodes"] for j in alg_data])
        xs = [j / (len(ys) - 1) for j in range(len(ys))]
        plt.plot(xs, ys, c=color, label=title, linewidth=3)
    plt.legend()
    plt.savefig(f"{DIRECTORY}/plots/sorted_num_nodes.png")

    # Plot IBS ratio
    plt.clf()
    plt.title(f"Integrated Brier Score Ratio")
    for alg, alg_data in data.items():
        title, color = ALG_INFO[alg]
        ys = sorted([j["results"][f"integrated_brier_score_ratio"] for j in alg_data])
        xs = [j / (len(ys) - 1) for j in range(len(ys))]
        plt.plot(xs, ys, c=color, label=title, linewidth=3)
    plt.legend()
    plt.savefig(f"{DIRECTORY}/plots/sorted_ibs_ratio.png")

    # Plot scores
    for name, attr in TRAIN_TEST_SCORE_TYPES:
        for type in ["train", "test"]:
            plt.clf()
            plt.title(f"{name} ({type})")
            for alg, alg_data in data.items():
                title, color = ALG_INFO[alg]
                ys = sorted([j["results"][f"{type}"][attr] for j in alg_data])
                xs = [j / (len(ys) - 1) for j in range(len(ys))]
                plt.plot(xs, ys, c=color, label=title, linewidth=3)
            plt.legend()
            plt.savefig(f"{DIRECTORY}/plots/sorted_{attr}_{type}.png")

# Compares the results of two algorithms (only if they have been run with the same list of settings)
#
# data  The output data for each of the algorithms
# alg1  The name of algorithm 1
# alg2  The name of algorithm 2
def compare_algs(data, alg1, alg2):
    alg1_data = data[alg1]
    alg2_data = data[alg2]

    # Determine whether the same settigns were used for both algorithms
    correct = True
    if len(alg1_data) != len(alg2_data):
        correct = False
    for line1, line2 in zip(alg1_data, alg2_data):
        if line1["settings"] != line2["settings"]:
            correct = False
            break
    if not correct:
        print("\033[31;1mThe algorithms to compare were not run with the same settings\033[0m")
        return

    def sign(x):
        return (x > 1e-6) - (x < -1e-6)

    # Compare num nodes
    counts = [0, 0, 0]
    diffs = []
    for line1, line2 in zip(alg1_data, alg2_data):
        diff = line1["results"]["num_nodes"] - line2["results"]["num_nodes"]
        counts[1 + sign(diff)] += 1
        diffs.append(diff)
    print("\033[37;1mNum nodes\033[0m")
    print(f"\033[31m  {alg1} < {alg2}:    {counts[0]}\033[0m")
    print(f"\033[33m  {alg1} = {alg2}:    {counts[1]}\033[0m")
    print(f"\033[32m  {alg1} > {alg2}:    {counts[2]}\033[0m")
    print(f"\033[30m  Difference: (μ, σ²) = ({np.average(diffs):.4}, {np.var(diffs):.4})\033[0m")
    print()

    # Compare IBS ratio
    counts = [0, 0, 0]
    diffs = []
    for line1, line2 in zip(alg1_data, alg2_data):
        diff = line1["results"]["integrated_brier_score_ratio"] - line2["results"]["integrated_brier_score_ratio"]
        counts[1 + sign(diff)] += 1
        diffs.append(diff)
    print("\033[37;1mIBS Ratio\033[0m")
    print(f"\033[31m  {alg1} < {alg2}:    {counts[0]}\033[0m")
    print(f"\033[33m  {alg1} = {alg2}:    {counts[1]}\033[0m")
    print(f"\033[32m  {alg1} > {alg2}:    {counts[2]}\033[0m")
    print(f"\033[30m  Difference: (μ, σ²) = ({np.average(diffs):.4}, {np.var(diffs):.4})\033[0m")
    print()

    # Compare scores
    for name, attr in TRAIN_TEST_SCORE_TYPES:
        for type in ["train", "test"]:
            counts = [0, 0, 0]
            diffs = []
            for line1, line2 in zip(alg1_data, alg2_data):
                diff = line1["results"][f"{type}"][attr] - line2["results"][f"{type}"][attr]
                counts[1 + sign(diff)] += 1
                diffs.append(diff)

                if alg1 == "streed" and alg2 == "ost":
                    if type == "train" and attr == "objective_score":
                        if diff < 0:
                            print(line1, line2)

            print(f"\033[37;1m{name} ({type})\033[0m")
            print(f"\033[31m  {alg1} < {alg2}:    {counts[0]}\033[0m")
            print(f"\033[33m  {alg1} = {alg2}:    {counts[1]}\033[0m")
            print(f"\033[32m  {alg1} > {alg2}:    {counts[2]}\033[0m")
            print(f"\033[30m  Difference: (μ, σ²) = ({np.average(diffs):.4}, {np.var(diffs):.4})\033[0m")
            print()

def main():
    # Load data for each algorithm
    data = {}
    for algorithm in ["ctree", "ost", "streed"]:
        f = open(f"{DIRECTORY}/output/{algorithm}_output.csv")
        lines = f.read().strip().split("\n")
        f.close()

        alg_data = []
        for line in lines[1:]:
            _, settings, results = [eval(j) for j in line.split(";")]
            alg_data.append({
                "settings": settings,
                "results": results
            })

        data[algorithm] = alg_data

    # Process the data
    plot_sorted_scores(data)
    compare_algs(data, "streed", "ost")
    print("=" * 42 + "\n")
    compare_algs(data, "streed", "ctree")
    print("=" * 42 + "\n")
    compare_algs(data, "ost", "ctree")

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()
