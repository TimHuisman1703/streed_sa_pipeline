import matplotlib.pyplot as plt
import numpy as np
from utils import Instance, nelson_aalen
from utils import DIRECTORY

def plot_with_theta(hazard_function, theta):
    plt.clf()
    _, ax = plt.subplots(
        figsize=(2, 3),
        dpi=300,
        gridspec_kw=dict(left=0.15, right=0.99, bottom=0.04, top=0.91),
        facecolor="#FFFFFF"
    )
    ax.set_ylim([0, 1.1])
    ax.set_xlim([0, 19.9])

    xs = [0]
    ys = [1]
    prev_y = 1
    for x in np.arange(0, 1000, 0.01):
        curr_y = np.exp(-theta * hazard_function(x))

        xs.extend([x - 1e-6, x])
        ys.extend([prev_y, curr_y])
        prev_y = curr_y

    plt.title(f"θ {'<=>'[int(theta)]} 1")
    plt.plot(xs, ys, color="#DF1F1F", linewidth=2)

    plt.ylabel("Survival rate", fontsize=14, labelpad=-10)
    plt.yticks([0, 1], ["0", "1"])
    plt.xticks([], [])
    plt.savefig(f"{DIRECTORY}/output/theta_{theta:.1f}.svg", dpi=300)
    plt.close()

if __name__ == "__main__":
    instances = [Instance({"time": j, "event": 1}) for j in range(1, 21)]
    hazard_function = nelson_aalen(instances)

    plot_with_theta(hazard_function, 0.5)
    plot_with_theta(hazard_function, 1)
    plot_with_theta(hazard_function, 2)

    print("\033[32;1mDone!\033[0m")
