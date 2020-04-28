import matplotlib.pyplot as plt
import numpy as np


def plot_score_cumulative_distribution(scores, ax=None, target=13):
    if ax is None:
        ax = plt.axes()
    ax.hist(
        scores, color="grey", bins=np.arange(-5, 40, 1), cumulative=True, density=True,
    )
    ax.axvline(13.0, color="orange", label=f"Goal ({target})")
    ax.axvline(
        np.mean(scores), color="green", label=f"Mean ({np.mean(scores)})",
    )
    ax.grid(True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Proportion Scores Less Than Threshold")
    ax.legend()
    ax.set_xlim(-5, 30)
    return ax


def plot_running_scores(scores, running_average=None, ax=None):
    if ax is None:
        ax = plt.axes()
    i_episode = len(scores)
    ax.scatter(np.arange(len(scores)), scores, alpha=0.5)
    if (running_average is not None) and (i_episode > 20):
        ax.plot(
            np.arange(len(running_average))[20:],
            running_average[20:],
            color="red",
            label="Trailing Average",
        )
    ax.axhline(0, color="black")
    ax.axhline(13, color="orange", label="target")
    ax.legend()
    plt.draw()
    return ax


def plot_final_scores(scores, ax=None, avg_size=100):
    if ax is None:
        ax = plt.axes()
    i_episode = len(scores)
    ax.scatter(np.arange(len(scores)), scores, alpha=0.5)
    if i_episode > avg_size:
        running_average = (
            np.cumsum(scores)[avg_size:] - np.cumsum(scores)[:-avg_size]
        ) / avg_size

        ax.plot(
            np.arange(avg_size, len(scores)),
            running_average,
            color="red",
            label="Trailing Average",
        )
    ax.axhline(0, color="black")
    ax.axhline(13, color="orange", label="target")
    ax.set_ylabel("Score")
    ax.set_xlabel("Training Run")
    ax.legend()
    plt.draw()
    return ax
