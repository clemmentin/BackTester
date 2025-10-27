import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def run_signal_diagnostics(csv_path: str):
    """
    Performs a deep diagnostic on the raw predictive power of alpha signals.
    This script bypasses all complex models to test the fundamental hypothesis:
    Do the raw 'score' and 'confidence' values have any correlation with actual returns?
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at '{csv_path}'")
        return

    print("=" * 80)
    print("      RAW ALPHA SIGNAL QUALITY DIAGNOSTICS")
    print("=" * 80)

    # --- 1. Basic Data Sanity Check ---
    print("\n--- [1. Data Sanity Check] ---")
    required_cols = [
        "factor_source",
        "entry_score",
        "entry_confidence",
        "actual_return",
    ]
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: CSV is missing one of the required columns: {required_cols}")
        return

    print(f"Loaded {len(df)} trade records.")
    print("Checking for variance in key predictive features...")
    for col in ["entry_score", "entry_confidence"]:
        if df[col].nunique() < 5:
            print(
                f"  WARNING: Feature '{col}' has very few unique values. It may lack predictive power."
            )
        else:
            print(f"  - '{col}' seems OK (has {df[col].nunique()} unique values).")

    # --- 2. Direct Correlation Analysis ---
    print("\n--- [2. Direct Correlation: Raw Signal vs. Actual Return] ---")

    # Group by factor source to see if any factor is individually predictive
    for factor, group in df.groupby("factor_source"):
        print(f"\n--- Analyzing Factor: {factor.upper()} ---")

        # Correlation between SCORE and return
        score_corr = group["entry_score"].corr(
            group["actual_return"], method="spearman"
        )
        print(f"  - Spearman Correlation (Score vs. Return): {score_corr:.4f}")

        # Correlation between CONFIDENCE and return
        conf_corr = group["entry_confidence"].corr(
            group["actual_return"], method="spearman"
        )
        print(f"  - Spearman Correlation (Confidence vs. Return): {conf_corr:.4f}")

        if abs(score_corr) < 0.02 and abs(conf_corr) < 0.02:
            print(
                "  >> DIAGNOSIS: Both Score and Confidence show near-zero correlation with returns."
            )
            print("     This suggests the factor is currently generating random noise.")
        elif abs(score_corr) > 0.05 or abs(conf_corr) > 0.05:
            print(
                "  >> DIAGNOSIS: Found a potentially predictive signal! (Correlation > 0.05)"
            )

    # --- 3. Visual Analysis: Scatter Plots ---
    print("\n--- [3. Visual Analysis] ---")
    print(
        "Generating scatter plots to visualize relationships... (Plots will be saved to disk)"
    )

    try:
        # Plot Score vs. Return for each factor
        g = sns.lmplot(
            data=df,
            x="entry_score",
            y="actual_return",
            col="factor_source",
            col_wrap=2,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "red"},
            height=4,
            facet_kws={"sharex": False, "sharey": False},
        )
        g.fig.suptitle("Raw Score vs. Actual Return by Factor", y=1.02)
        plt.savefig("diagnostic_score_vs_return.png")
        plt.close()
        print("  - Saved 'diagnostic_score_vs_return.png'")

        # Plot Confidence vs. Return for each factor
        g = sns.lmplot(
            data=df,
            x="entry_confidence",
            y="actual_return",
            col="factor_source",
            col_wrap=2,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "red"},
            height=4,
            facet_kws={"sharex": False, "sharey": False},
        )
        g.fig.suptitle("Raw Confidence vs. Actual Return by Factor", y=1.02)
        plt.savefig("diagnostic_confidence_vs_return.png")
        plt.close()
        print("  - Saved 'diagnostic_confidence_vs_return.png'")

    except Exception as e:
        print(f"  ERROR creating plots: {e}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE.")
    print("Check the printed correlations and the saved PNG image files.")
    print(
        "A healthy factor should show a discernibly upward-sloping (or downward-sloping) red line in the plots."
    )
    print("A flat red line indicates NO predictive power.")
    print("=" * 80)


if __name__ == "__main__":
    run_signal_diagnostics(csv_path="data/ml_training_data.csv")
