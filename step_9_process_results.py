import matplotlib.pyplot as plt
import os

DIRECTORY = os.path.realpath(os.path.dirname(__file__))

if not os.path.exists(f"{DIRECTORY}/plots"):
    os.mkdir(f"{DIRECTORY}/plots")

ALG_INFO = {
    "ctree": ("CTree", "#1F7FDF"),
    "ost": ("OST", "#DF7F1F"),
    "streed": ("STreeD", "#1FBF1F"),
}
SCORE_TYPES = [
    ("Objective score", "objective_score"),
    ("Concordance score", "concordance_score"),
    # ("IBS Ratio", "integrated_brier_score_ratio"),
]

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

    # Plot scores
    for name, attr in SCORE_TYPES:
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

def compare_algs(data, alg1, alg2):
    alg1_data = data[alg1]
    alg2_data = data[alg2]

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
        return (x > 0) - (x < 0)

    # Compare num nodes
    counts = [0, 0, 0]
    for line1, line2 in zip(alg1_data, alg2_data):
        counts[1 + sign(line1["results"]["num_nodes"] - line2["results"]["num_nodes"])] += 1
    print("\033[37;1mNum nodes\033[0m")
    print(f"\033[31m  {alg1} < {alg2}:    {counts[0]}\033[0m")
    print(f"\033[33m  {alg1} = {alg2}:    {counts[1]}\033[0m")
    print(f"\033[32m  {alg1} > {alg2}:    {counts[2]}\033[0m")
    print()

    # Compare scores
    for name, attr in SCORE_TYPES:
        for type in ["train", "test"]:
            counts = [0, 0, 0]
            for line1, line2 in zip(alg1_data, alg2_data):
                counts[1 + sign(line1["results"][f"{type}"][attr] - line2["results"][f"{type}"][attr])] += 1
            print(f"\033[37;1m{name} ({type})\033[0m")
            print(f"\033[31m  {alg1} < {alg2}:    {counts[0]}\033[0m")
            print(f"\033[33m  {alg1} = {alg2}:    {counts[1]}\033[0m")
            print(f"\033[32m  {alg1} > {alg2}:    {counts[2]}\033[0m")
            print()

if __name__ == "__main__":
    data = {}
    for algorithm in ["ctree", "ost", "streed"]:
        f = open(f"{DIRECTORY}/output/{algorithm}_output.csv")
        lines = f.read().strip().split("\n")
        f.close()

        alg_data = []
        for line in lines[1:]:
            _, settings, time, results = [eval(j) for j in line.split(";")]
            alg_data.append({
                "settings": settings,
                "time": time,
                "results": results
            })

        data[algorithm] = alg_data

    plot_sorted_scores(data)
    compare_algs(data, "streed", "ost")

    print("\033[32;1mDone!\033[0m")