from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path(
    "results/mlp_optimizer_sweep_eta_seed.csv"
)

OUTPUT_PATH = Path(
    "results/mlp_optimizer_efficiency.png"
)


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)

    numeric_columns = [
        "lr",
        "seed",
        "test_accuracy",
        "total_time_s",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df[df["status"] == "ok"].copy()

    df = df.dropna(
        subset=[
            "test_accuracy",
            "total_time_s",
        ]
    )

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
            mean_time_s=(
                "total_time_s",
                "mean",
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

    label_map = {
        "rlbfgs": "Vanilla R-LBFGS",
        "globalized_rlbfgs": "Globalized R-LBFGS",
    }

    for optimizer_name, optimizer_df in summary.groupby(
        "optimizer"
    ):
        optimizer_df = optimizer_df.sort_values("lr")

        ax.scatter(
            optimizer_df["mean_time_s"],
            optimizer_df["mean_test_accuracy"],
            s=80,
            label=label_map.get(
                optimizer_name,
                optimizer_name,
            ),
        )

        for _, row in optimizer_df.iterrows():
            ax.annotate(
                f"{row['lr']:.3g}",
                (
                    row["mean_time_s"],
                    row["mean_test_accuracy"],
                ),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel(
        "Mean training time over 3 epochs (s)"
    )

    ax.set_ylabel(
        "CIFAR-10 test accuracy (%)"
    )

    ax.set_title(
        "MLP performance vs training time"
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    ax.legend(
        frameon=False
    )

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

    print("\nEfficiency summary:")
    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()