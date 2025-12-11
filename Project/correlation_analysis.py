import pandas as pd
import numpy as np
from phik import phik_matrix

# Load the data
df = pd.read_csv("data/merged.csv")

# Select the variables of interest
variables = [
    "phq9_severity_group_label",
    "gad7_severity_group_label",
    "professional_recommendation",
    "professional_recommendation_clinician",
]

# Filter to only rows where all variables are present
df_filtered = df[variables].dropna()

print("=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
print(f"\nNumber of complete observations: {len(df_filtered)}")
print(f"Original dataset size: {len(df)}")
print(f"Rows with missing values: {len(df) - len(df_filtered)}")

# Calculate Phi-K correlation matrix (works for categorical variables)
print("\n" + "=" * 80)
print("PHI-K CORRELATION MATRIX (for categorical variables)")
print("=" * 80)
phik_corr = df_filtered.phik_matrix()
print(phik_corr)

# Calculate correlation between specific pairs
print("\n" + "=" * 80)
print("PAIRWISE CORRELATIONS")
print("=" * 80)

pairs = [
    ("phq9_severity_group_label", "gad7_severity_group_label"),
    ("phq9_severity_group_label", "professional_recommendation"),
    ("phq9_severity_group_label", "professional_recommendation_clinician"),
    ("gad7_severity_group_label", "professional_recommendation"),
    ("gad7_severity_group_label", "professional_recommendation_clinician"),
    ("professional_recommendation", "professional_recommendation_clinician"),
]

for var1, var2 in pairs:
    corr = phik_corr.loc[var1, var2]
    print(f"{var1} <-> {var2}: {corr:.4f}")

# Show value counts for each variable
print("\n" + "=" * 80)
print("VALUE DISTRIBUTIONS")
print("=" * 80)

for var in variables:
    print(f"\n{var}:")
    print(df_filtered[var].value_counts().sort_index())
    print()
