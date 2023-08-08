import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

DIRECTORY = os.path.realpath(os.path.dirname(__file__))

if not os.path.exists(f"{DIRECTORY}/tables"):
    os.mkdir(f"{DIRECTORY}/tables")

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
    algorithms = ["streed", "ost"]
    for algorithm in algorithms:
        with open(f"{DIRECTORY}/output/{algorithm}_output.csv") as f:
            lines = f.read().strip().split("\n")
            for line in lines[1:]:
                _, settings, _results = [eval(j) for j in line.split(";")]
                results = {key: _results[key] for key in ["runtime", "num_nodes", "integrated_brier_score_ratio"]}
                results.update({key: settings[key] for key in ["max-depth","max-num-nodes", "mode", "cost-complexity"]})
                results["dataset"] = "_".join(settings["file"].replace("train/","").split("_")[:-2])
                results["method"] = algorithm
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

    print(df.head(5))

    #fig, ((hc1, hc2, hc3), (ib1, ib2, ib3)) = plt.subplots(2,3)

    
    df2 = df.melt(id_vars=["dataset", "method", "censoring_category", "n_instances"], value_vars=["test_concordance_score", "integrated_brier_score_ratio"])
    print(df2.head(5))

    #sns.lineplot(data=df[df["censoring_category"] == "Moderate"], x="n_instances", y="test_concordance_score", hue='method')
    rel = sns.relplot(data=df2, x="n_instances", y="value", hue='method', col='censoring_category', row='variable', kind='line',
                      height=2.0, aspect = (6.6 / (3 *2.0)))
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.xlim(100, 10000)

    plt.show()

    #sns.relplot(data=df, x="n_instances", y="integrated_brier_score_ratio", hue='method', col='censoring_category', kind='line')

    #plt.show()

if __name__ == "__main__":
    main()
