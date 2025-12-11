"""
Basic analysis of merged.csv dataset attributes using Polars
"""

import polars as pl
import pandas as pd
import sys
import os
from pathlib import Path

import phik
import matplotlib.pyplot as plt
import seaborn as sns

from disagreement_model.encoding_utils import fix_windows_encoding

fix_windows_encoding()


def load_data(filepath):
    """Load the CSV file into a Polars DataFrame"""
    try:
        # Define date columns to parse
        date_columns = [
            "date_of_birth",
            "assessment_date",
            "episode_date",
            "request_date",
            "publication_date",
            "report_date",
            "from_date",
            "to_date",
            "completion_date",
            "results_date",
            "evaluation_date",
            "date_of_birth_2",
        ]

        df = pl.read_csv(filepath, try_parse_dates=False)

        # Ensure date columns are parsed as dates
        for col in date_columns:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col)
                    .str.strptime(pl.Date, format="%Y-%m-%d %H:%M:%S", strict=False)
                    .alias(col)
                )

        print(
            f"✓ Successfully loaded data: {df.shape[0]} rows × {df.shape[1]} columns\n"
        )
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)


def basic_info(df):
    """Display basic information about the dataset"""
    print("=" * 80)
    print("BASIC DATASET INFORMATION")
    print("=" * 80)
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"\nMemory usage: {df.estimated_size('mb'):.2f} MB")
    print("\n" + "-" * 80)


def column_info(df):
    """Display information about columns"""
    print("\nCOLUMN INFORMATION")
    print("=" * 80)

    schema_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].null_count()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].n_unique()

        schema_info.append(
            {
                "Column": col,
                "Type": dtype,
                "Nulls": null_count,
                "Null %": f"{null_pct:.1f}",
                "Unique": unique_count,
            }
        )

    schema_df = pl.DataFrame(schema_info)
    print(schema_df)
    print("\n" + "-" * 80)


def numeric_summary(df):
    """Display summary statistics for numeric columns"""
    print("\nNUMERIC COLUMNS SUMMARY")
    print("=" * 80)

    numeric_cols = [
        col
        for col in df.columns
        if df[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]
    ]

    if numeric_cols:
        print(f"Found {len(numeric_cols)} numeric columns\n")
        stats = df.select(numeric_cols).describe()
        print(stats)
    else:
        print("No numeric columns found")

    print("\n" + "-" * 80)


def categorical_summary(df):
    """Display summary for categorical/string columns"""
    print("\nCATEGORICAL COLUMNS SUMMARY (Top 10 by unique values)")
    print("=" * 80)

    categorical_cols = [
        col for col in df.columns if df[col].dtype in [pl.Utf8, pl.Categorical]
    ]

    if categorical_cols:
        print(f"Found {len(categorical_cols)} categorical columns\n")
        for col in categorical_cols[:10]:  # Show first 10 to avoid overwhelming output
            unique_count = df[col].n_unique()
            most_common = (
                df.group_by(col)
                .agg(pl.count().alias("count"))
                .sort("count", descending=True)
                .head(5)
            )
            print(f"\n{col}:")
            print(f"  Unique values: {unique_count}")
            print(f"  Top 5 values:")
            print(most_common)
    else:
        print("No categorical columns found")

    print("\n" + "-" * 80)


def key_fields_analysis(df):
    """Analyze key fields relevant to the study"""
    print("\nKEY FIELDS ANALYSIS")
    print("=" * 80)

    key_fields = [
        "algorithm_recommendation",
        "professional_recommendation_clinician",
        "algorithm_professional_agreement",
        "risk_assessment_algorithm",
        "risk_assessment_psychiatrist",
        "phq9_severity_group_label",
        "gad7_severity_group_label",
        "cssrs_group_2a_label",
    ]

    for field in key_fields:
        if field in df.columns:
            print(f"\n{field}:")
            value_counts = (
                df.group_by(field)
                .agg(pl.count().alias("count"))
                .sort("count", descending=True)
            )
            print(value_counts)
            print(f"  Null values: {df[field].null_count()}")
        else:
            print(f"\n{field}: NOT FOUND in dataset")

    print("\n" + "-" * 80)


def agreement_overview(df):
    """Quick overview of algorithm vs professional agreement"""
    print("\nALGORITHM VS PROFESSIONAL AGREEMENT OVERVIEW")
    print("=" * 80)

    if "algorithm_professional_agreement" in df.columns:
        agreement_dist = (
            df.group_by("algorithm_professional_agreement")
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )
        print("\nAgreement Distribution:")
        print(agreement_dist)

        total = df.shape[0]
        agree_count = (
            df.filter(pl.col("algorithm_professional_agreement") == 1.0).shape[0]
            if 1.0 in df["algorithm_professional_agreement"].to_list()
            else 0
        )
        disagree_count = (
            df.filter(pl.col("algorithm_professional_agreement") == 0.0).shape[0]
            if 0.0 in df["algorithm_professional_agreement"].to_list()
            else 0
        )

        print(
            f"\nAgreement Rate: {(agree_count / total * 100):.1f}% ({agree_count}/{total})"
        )
        print(
            f"Disagreement Rate: {(disagree_count / total * 100):.1f}% ({disagree_count}/{total})"
        )
    else:
        print("Field 'algorithm_professional_agreement' not found")

    print("\n" + "-" * 80)


def generate_phik_correlation_matrix(df):
    """Generate and save phik correlation matrix"""
    print("\nGENERATING PHIK CORRELATION MATRIX")
    print("=" * 80)

    # Create eda_results directory if it doesn't exist
    output_dir = Path("eda_results")
    output_dir.mkdir(exist_ok=True)

    # Convert Polars DataFrame to Pandas for phik compatibility
    print("Converting data to pandas format...")
    df_pandas = df.to_pandas()

    # Calculate phik correlation matrix
    print("Calculating phik correlation matrix...")
    phik_matrix = df_pandas.phik_matrix()

    # Save the correlation matrix as CSV
    csv_path = output_dir / "phik_correlation_matrix.csv"
    phik_matrix.to_csv(csv_path)
    print(f"✓ Saved correlation matrix to: {csv_path}")

    # Create and save heatmap visualization
    print("Generating heatmap visualization...")
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        phik_matrix,
        annot=False,  # Don't show values due to large matrix
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Phik Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()

    heatmap_path = output_dir / "phik_correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved heatmap to: {heatmap_path}")

    # Find and display top correlations (excluding diagonal)
    print("\nTop 20 Correlations (excluding self-correlations):")
    correlations = []
    for i in range(len(phik_matrix.columns)):
        for j in range(i + 1, len(phik_matrix.columns)):
            correlations.append(
                {
                    "Feature 1": phik_matrix.columns[i],
                    "Feature 2": phik_matrix.columns[j],
                    "Phik Correlation": phik_matrix.iloc[i, j],
                }
            )

    corr_df = pd.DataFrame(correlations).sort_values(
        "Phik Correlation", ascending=False
    )
    print(corr_df.head(20).to_string(index=False))

    # Save top correlations to CSV
    top_corr_path = output_dir / "top_phik_correlations.csv"
    corr_df.to_csv(top_corr_path, index=False)
    print(f"\n✓ Saved top correlations to: {top_corr_path}")

    print("\n" + "-" * 80)


def main():
    """Main execution function"""
    filepath = "data/merged.csv"

    print("\n" + "=" * 80)
    print("MERGED DATA ANALYSIS")
    print("=" * 80 + "\n")

    df = load_data(filepath)

    basic_info(df)
    column_info(df)
    numeric_summary(df)
    categorical_summary(df)
    key_fields_analysis(df)
    agreement_overview(df)
    generate_phik_correlation_matrix(df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
