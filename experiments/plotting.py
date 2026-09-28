"""
Plotting utilities for experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy_vs_eta(
    eta_results: dict[float, list[float]],
    title: str | None = None,
    log_x: bool = False,
    save_path: str | None = None,
) -> None:
    """
    Plot mean test accuracy ± standard deviation
    as a function of eta.

    Parameters
    ----------
    eta_results:
        Mapping:
            eta -> list of test accuracies across seeds

    title:
        Optional plot title.

    log_x:
        Use a logarithmic x-axis.

    save_path:
        Optional path where the figure should be saved.
    """

    etas = sorted(eta_results.keys())

    means = []
    stds = []

    for eta in etas:
        accuracies = np.asarray(
            eta_results[eta],
            dtype=float,
        )

        means.append(accuracies.mean())
        stds.append(accuracies.std())

    plt.figure()

    plt.errorbar(
        etas,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
    )

    plt.xlabel("eta (initial_lr)")
    plt.ylabel("Test accuracy (%)")

    if title is not None:
        plt.title(title)

    if log_x:
        plt.xscale("log")

    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()