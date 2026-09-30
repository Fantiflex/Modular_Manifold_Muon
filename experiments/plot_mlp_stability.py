from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path(
    "results/mlp_optimizer_sweep_eta_seed.csv"
)

OUTPUT_PATH = Path(
    "results/mlp_optimizer_sweep_stability.png"
)


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)

    # Keep only successful runs.
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
            min_test_accuracy=(
                "test_accuracy",
                "min",
            ),
            max_test_accuracy=(
                "test_accuracy",
                "max",
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

        ax.plot(
            optimizer_df["lr"],
            optimizer_df["std_test_accuracy"],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.set_xscale("log")

    ax.set_xlabel(
        r"Initial step size $\eta_0$"
    )

    ax.set_ylabel(
        "Std. dev. of test accuracy (%)"
    )

    ax.set_title(
        "MLP sensitivity across random seeds"
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

    print("\nStability summary:")
    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()