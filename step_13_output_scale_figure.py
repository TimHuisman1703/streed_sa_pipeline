import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

DIRECTORY = os.path.realpath(os.path.dirname(__file__))

if not os.path.exists(f"{DIRECTORY}/plots"):
    os.mkdir(f"{DIRECTORY}/plots")

def main():

    sns.set_context('paper')
    plt.rc('font', size=10, family='serif')
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('axes', labelsize='small', grid=True)
    plt.rc('legend', fontsize='x-small')
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)
    plt.rc('text', usetex = True)
    sns.set_palette("colorblind")

    data = []
    algorithms = ["streed", "ost", "ctree"]
    algorithm_name = {"ost": "OST", "streed": "SurTree", "ctree": "CTree"}
    for algorithm in algorithms:
        with open(f"{DIRECTORY}/output/{algorithm}_output.csv") as f:
            lines = f.read().strip().split("\n")
            for line in lines[1:]:
                _, settings, _results = [eval(j) for j in line.split(";")]
                if _results["runtime"] >= 600 or _results["runtime"] < -1e6:
                    _results["runtime"] = 1200
                results = {key: _results[key] for key in ["runtime", "num_nodes", "integrated_brier_score_ratio"]}
                results.update({key: settings[key] for key in ["max-depth","max-num-nodes", "mode", "cost-complexity"]})
                results["dataset"] = settings["file"].replace("train/","")
                results["method"] = algorithm_name[algorithm]
                results.update({"train_"+key: _results["train"][key] for key in ["objective_score", "concordance_score"]})
                results.update({"test_"+key: _results["test"][key] for key in ["objective_score", "concordance_score"]})
                data.append(results)

    df = pd.DataFrame.from_dict(data)
    datasets = df["dataset"].unique()

    dataset_info = {}
    for dataset in datasets:
        with open(f"{DIRECTORY}/datasets/binary/{dataset}.txt") as f2:
            lines = f2.readlines()
            n_instances = len(lines) - 1
            n_binary_features = len(lines[0].split(",")) - 2
            censoring = 1.0 - sum([int(line.split(",")[1]) for line in lines[1:]]) / n_instances
        with open(f"{DIRECTORY}/datasets/original/{dataset}.txt") as f2:
            lines = f2.readlines()
            n_features = len(lines[0].split(",")) - 2
        dataset_info[dataset] = {"n_instances": n_instances, "n_features": n_features, "n_binary_features": n_binary_features, "censoring": censoring}

        df.loc[df["dataset"] == dataset, "n_instances"] = n_instances
        df.loc[df["dataset"] == dataset, "n_features"] = n_features
        df.loc[df["dataset"] == dataset, "censoring"] = censoring
        censoring_category = "Moderate"
        if censoring <= 0.25:
            censoring_category = "Low"
        elif censoring >= 0.75:
            censoring_category = "High"
        df.loc[df["dataset"] == dataset, "censoring_category"] = censoring_category


    #df["method_f"] = df[['method', "n_features"]].apply(lambda row: f"{row['method']} ({int(row['n_features']):d} features)", axis=1)
    df["method_f"] = df[['method', "n_features"]].apply(lambda row: f"{row['method']} (f={int(row['n_features']/6):d})", axis=1)

    #method_order = ["CTree (6 features)", "CTree (12 features)", "OST (6 features)", "OST (12 features)", "SurTree (6 features)", "SurTree (12 features)"]
    method_order = ["CTree (f=1)", "CTree (f=2)", "OST (f=1)", "OST (f=2)", "SurTree (f=1)", "SurTree (f=2)"]
    
    plt.figure(figsize=(3.3+0.3, 1.6))

    g = sns.lineplot(data = df, x="max-depth", y='runtime',
                 hue="method_f", hue_order=method_order,
                 style='method_f', style_order=method_order
                )
    g.set_yscale("log")
    g.set_xlim(2, 5)
    g.set_ylabel("Run time (s)")
    g.set_xlabel("Maximum depth")

    plt.subplots_adjust(right=0.6)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), ncol=1, title="", frameon=False)
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"{DIRECTORY}/plots/runtime_d.pdf", bbox_inches="tight", pad_inches = 0)

if __name__ == "__main__":
    main()

