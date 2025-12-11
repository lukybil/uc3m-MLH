"""
Analysis of cases where algorithm_recommendation differs from professional_recommendation_clinician
using Polars
"""

import polars as pl
import sys

# Fix Windows encoding once at entry point
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


def filter_disagreements(df):
    """Filter rows where algorithm and professional recommendations differ"""

    # Check if required columns exist
    if (
        "algorithm_recommendation" not in df.columns
        or "professional_recommendation_clinician" not in df.columns
    ):
        print("✗ Required columns not found in dataset")
        sys.exit(1)

    # Filter where recommendations differ (handling potential nulls)
    disagreements = df.filter(
        (pl.col("algorithm_recommendation").is_not_null())
        & (pl.col("professional_recommendation_clinician").is_not_null())
        & (
            pl.col("algorithm_recommendation")
            != pl.col("professional_recommendation_clinician")
        )
    )

    total = df.shape[0]
    disagree_count = disagreements.shape[0]

    print("=" * 80)
    print("DISAGREEMENT FILTERING")
    print("=" * 80)
    print(f"Total records: {total}")
    print(f"Disagreement cases: {disagree_count} ({(disagree_count/total*100):.2f}%)")
    print(
        f"Agreement cases: {total - disagree_count} ({((total-disagree_count)/total*100):.2f}%)"
    )
    print("-" * 80 + "\n")

    return disagreements


def disagreement_summary(df):
    """Summary statistics for disagreement cases"""
    print("=" * 80)
    print("DISAGREEMENT CASES SUMMARY")
    print("=" * 80)

    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.estimated_size('mb'):.2f} MB\n")

    # Algorithm recommendations in disagreement cases
    print("Algorithm Recommendations:")
    algo_dist = (
        df.group_by("algorithm_recommendation")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )
    print(algo_dist)

    # Professional recommendations in disagreement cases
    print("\nProfessional Recommendations:")
    prof_dist = (
        df.group_by("professional_recommendation_clinician")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )
    print(prof_dist)

    # Cross-tabulation
    print("\nCross-tabulation (Algorithm vs Professional):")
    cross_tab = (
        df.group_by(
            ["algorithm_recommendation", "professional_recommendation_clinician"]
        )
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )
    print(cross_tab)

    print("\n" + "-" * 80)


def demographic_analysis(df):
    """Analyze demographic characteristics of disagreement cases"""
    print("\nDEMOGRAPHIC ANALYSIS OF DISAGREEMENT CASES")
    print("=" * 80)

    # Sex distribution
    if "sex" in df.columns:
        print("\nSex Distribution:")
        sex_dist = (
            df.group_by("sex")
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )
        print(sex_dist)

    # Center distribution
    if "center_id" in df.columns:
        print("\nCenter Distribution:")
        center_dist = (
            df.group_by("center_id")
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )
        print(center_dist)

    # Location of birth
    if "location_of_birth" in df.columns:
        print("\nLocation of Birth:")
        location_dist = (
            df.group_by("location_of_birth")
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )
        print(location_dist)

    # Death status
    if "death_status" in df.columns:
        print("\nDeath Status:")
        death_dist = (
            df.group_by("death_status")
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )
        print(death_dist)

    print("\n" + "-" * 80)


def clinical_scores_analysis(df):
    """Analyze clinical assessment scores in disagreement cases"""
    print("\nCLINICAL SCORES ANALYSIS")
    print("=" * 80)

    clinical_scores = [
        "phq9_total_score",
        "phq9_severity_group_label",
        "gad7_total_score",
        "gad7_severity_group_label",
        "who5_total_score",
        "cssrs_total_score_1",
        "cssrs_group_2a_label",
        "auditc_total_score",
        "dast_total_score",
    ]

    for score in clinical_scores:
        if score in df.columns:
            print(f"\n{score}:")

            if df[score].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
                # Numeric scores - show statistics
                stats = df.select(score).describe()
                print(stats)
            else:
                # Categorical - show distribution
                dist = (
                    df.group_by(score)
                    .agg(pl.count().alias("count"))
                    .sort("count", descending=True)
                )
                print(dist)

    print("\n" + "-" * 80)


def risk_assessment_comparison(df):
    """Compare risk assessments between algorithm and professionals"""
    print("\nRISK ASSESSMENT COMPARISON")
    print("=" * 80)

    if "risk_assessment_algorithm" in df.columns:
        print("\nAlgorithm Risk Assessment:")
        risk_algo = (
            df.group_by("risk_assessment_algorithm")
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )
        print(risk_algo)

    if "risk_assessment_psychiatrist" in df.columns:
        print("\nPsychiatrist Risk Assessment:")
        risk_psych = (
            df.group_by("risk_assessment_psychiatrist")
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )
        print(risk_psych)

    if (
        "risk_assessment_algorithm" in df.columns
        and "risk_assessment_psychiatrist" in df.columns
    ):
        print("\nRisk Assessment Cross-tabulation:")
        risk_cross = (
            df.group_by(["risk_assessment_algorithm", "risk_assessment_psychiatrist"])
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )
        print(risk_cross)

    print("\n" + "-" * 80)


def medication_and_behavior_analysis(df):
    """Analyze medication and behavioral factors"""
    print("\nMEDICATION & BEHAVIORAL FACTORS")
    print("=" * 80)

    factors = [
        "current_psychotropic_medication",
        "current_smoker",
        "cigarettes_per_day",
        "is_pregnant_woman",
    ]

    for factor in factors:
        if factor in df.columns:
            print(f"\n{factor}:")

            if df[factor].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
                stats = df.select(factor).describe()
                print(stats)
            else:
                dist = (
                    df.group_by(factor)
                    .agg(pl.count().alias("count"))
                    .sort("count", descending=True)
                )
                print(dist)

    print("\n" + "-" * 80)


def personality_analysis(df):
    """Analyze Big Five personality traits (BFI-10)"""
    print("\nPERSONALITY TRAITS (BFI-10) ANALYSIS")
    print("=" * 80)

    bfi_scores = [
        "bfi10_extraversion_score",
        "bfi10_agreeableness_score",
        "bfi10_conscientiousness_score",
        "bfi10_emotional_stability_score",
        "bfi10_openness_to_experience_score",
    ]

    bfi_interpretations = [
        "bfi10_agreeableness_interpretation",
        "bfi10_conscientiousness_interpretation",
        "bfi10_emotional_stability_interpretation",
        "bfi10_openness_to_experience_interpretation",
    ]

    print("\nBFI-10 Scores (Numeric):")
    for score in bfi_scores:
        if score in df.columns:
            stats = df.select(score).describe()
            print(f"\n{score}:")
            print(stats)

    print("\nBFI-10 Interpretations (Categorical):")
    for interp in bfi_interpretations:
        if interp in df.columns:
            dist = (
                df.group_by(interp)
                .agg(pl.count().alias("count"))
                .sort("count", descending=True)
            )
            print(f"\n{interp}:")
            print(dist)

    print("\n" + "-" * 80)


def service_and_status_analysis(df):
    """Analyze service types and record statuses"""
    print("\nSERVICE & STATUS ANALYSIS")
    print("=" * 80)

    fields = [
        "requesting_service",
        "response_type",
        "record_status",
        "form_status",
        "service",
        "care_circuit",
        "package_evaluated",
        "evaluating_user",
    ]

    for field in fields:
        if field in df.columns:
            print(f"\n{field}:")
            dist = (
                df.group_by(field)
                .agg(pl.count().alias("count"))
                .sort("count", descending=True)
                .head(10)
            )
            print(dist)

    print("\n" + "-" * 80)


def psychosis_and_self_harm_analysis(df):
    """Analyze psychosis symptoms and self-harm indicators"""
    print("\nPSYCHOSIS & SELF-HARM ANALYSIS")
    print("=" * 80)

    psychosis_fields = [
        "psychosis_1_thoughts_being_interfered_with_or_controlled",
        "psychosis_2_group_conspiring_to_harm_you",
        "psychosis_3_something_very_strange_happening",
        "psychosis_4_heard_voices_with_no_one_there",
    ]

    self_harm_fields = [
        "nssi_ever_intentional_self_injury_without_suicidal_intent",
        "nssi_last_month_frequency",
    ]

    print("\nPsychosis Symptoms:")
    for field in psychosis_fields:
        if field in df.columns:
            dist = (
                df.group_by(field)
                .agg(pl.count().alias("count"))
                .sort("count", descending=True)
            )
            print(f"\n{field}:")
            print(dist)

    print("\nSelf-Harm Indicators:")
    for field in self_harm_fields:
        if field in df.columns:
            if df[field].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
                stats = df.select(field).describe()
                print(f"\n{field}:")
                print(stats)
            else:
                dist = (
                    df.group_by(field)
                    .agg(pl.count().alias("count"))
                    .sort("count", descending=True)
                )
                print(f"\n{field}:")
                print(dist)

    print("\n" + "-" * 80)


def export_disagreements(df, output_path="data/disagreement_cases.csv"):
    """Export disagreement cases to a new CSV file"""
    try:
        df.write_csv(output_path)
        print(f"\n✓ Disagreement cases exported to: {output_path}")
        print(f"  Rows exported: {df.shape[0]}")
    except Exception as e:
        print(f"\n✗ Error exporting data: {e}")


def main():
    """Main execution function"""
    filepath = "data/merged.csv"

    print("\n" + "=" * 80)
    print("DISAGREEMENT ANALYSIS")
    print("Algorithm vs Professional Recommendations")
    print("=" * 80 + "\n")

    # Load data
    df = load_data(filepath)

    # Filter disagreement cases
    disagreements = filter_disagreements(df)

    if disagreements.shape[0] == 0:
        print("No disagreement cases found. Analysis cannot proceed.")
        sys.exit(0)

    # Run analyses
    disagreement_summary(disagreements)
    demographic_analysis(disagreements)
    clinical_scores_analysis(disagreements)
    risk_assessment_comparison(disagreements)
    medication_and_behavior_analysis(disagreements)
    personality_analysis(disagreements)
    service_and_status_analysis(disagreements)
    psychosis_and_self_harm_analysis(disagreements)

    # Export results
    export_disagreements(disagreements)

    print("\n" + "=" * 80)
    print("DISAGREEMENT ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
