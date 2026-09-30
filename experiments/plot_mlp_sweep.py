from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path(
    "results/mlp_optimizer_sweep_eta_seed.csv"
)

OUTPUT_PATH = Path(
    "results/mlp_optimizer_sweep_accuracy.png"
)


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)

    df = df[df["status"] == "ok"].copy()

    summary = (
        df.groupby(
            ["optimizer", "lr"],
            as_index=False,
        )
        .agg(
            mean_test_accuracy=(
                "test_accuracy",
                "mean",
            ),
            std_test_accuracy=(
                "test_accuracy",
                "std",
            ),
            n_seeds=(
                "seed",
                "nunique",
            ),
        )
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    for optimizer_name, optimizer_df in summary.groupby(
        "optimizer"
    ):
        optimizer_df = optimizer_df.sort_values("lr")

        label = {
            "rlbfgs": "Vanilla R-LBFGS",
            "globalized_rlbfgs": "Globalized R-LBFGS",
        }.get(
            optimizer_name,
            optimizer_name,
        )

        ax.errorbar(
            optimizer_df["lr"],
            optimizer_df["mean_test_accuracy"],
            yerr=optimizer_df["std_test_accuracy"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=label,
        )

    ax.set_xscale("log")

    ax.set_xlabel(
        r"Initial step size $\eta_0$"
    )
    ax.set_ylabel(
        "CIFAR-10 test accuracy (%)"
    )

    ax.set_title(
        "MLP: Vanilla vs Globalized R-LBFGS"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved figure to: {OUTPUT_PATH}"
    )

    print("\nAggregated results:")
    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()