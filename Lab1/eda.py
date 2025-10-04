import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from phik import resources, report
from phik.report import plot_correlation_matrix
from scipy.io import arff

from loader import load_and_clean_data

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

plot_dir = "eda_plots"

CATEGORICAL_COLS = [
    "Recipientgender",
    "Stemcellsource",
    "Donorage35",
    "IIIV",
    "Gendermatch",
    "DonorABO",
    "RecipientABO",
    "RecipientRh",
    "ABOmatch",
    "CMVstatus",
    "DonorCMV",
    "RecipientCMV",
    "Disease",
    "Riskgroup",
    "Txpostrelapse",
    "Diseasegroup",
    "HLAmatch",
    "HLAmismatch",
    "Antigen",
    "Alel",
    "HLAgrI",
    "Recipientage10",
    "Recipientageint",
    "Relapse",
    "aGvHDIIIIV",
    "extcGvHD",
]
NUMERICAL_COLS = [
    "Donorage",
    "Recipientage",
    "CD34kgx10d6",
    "CD3dCD34",
    "CD3dkgx10d8",
    "Rbodymass",
    "ANCrecovery",
    "PLTrecovery",
    "time_to_aGvHD_III_IV",
    "survival_time",
    "survival_status",
]


def explore_data_types(df):
    """
    Explore data types and unique values for each column
    """
    print("\n" + "=" * 80)
    print("DATA TYPES AND UNIQUE VALUES")
    print("=" * 80)

    print("\nData types:")
    print(df.dtypes)

    cat_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    num_cols = [col for col in NUMERICAL_COLS if col in df.columns]

    print(f"\nCategorical features ({len(cat_cols)}): {cat_cols}")
    print(f"\nNumerical features ({len(num_cols)}): {num_cols}")

    print("\nUnique values and counts for categorical features:")
    for col in cat_cols:
        n_unique = df[col].nunique()
        print(f"\n{col}: {n_unique} unique values")
        print(df[col].value_counts().sort_values(ascending=False).head(10))

    print("\nStatistical summary for numerical features:")
    print(df[num_cols].describe().T)

    return cat_cols, num_cols


def analyze_target_distribution(df):
    """
    Analyze the target variable distribution (survival_status)
    """
    print("\n" + "=" * 80)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 80)

    target_col = "survival_status"
    target_counts = df[target_col].value_counts()
    print("\nTarget distribution:")
    print(target_counts)
    for val, count in target_counts.items():
        print(f"Percentage of {val}: {count/len(df)*100:.2f}%")

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=target_col, data=df, palette="viridis")
    plt.title("Distribution of Survival Status", fontsize=16)
    plt.xlabel("Survival Status", fontsize=14)
    plt.ylabel("Count", fontsize=14)

    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            height + 2,
            f"{height} ({height/len(df)*100:.1f}%)",
            ha="center",
            fontsize=12,
        )

    plt.savefig(f"{plot_dir}/target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nTarget distribution plot saved as 'target_distribution.png'")


def analyze_categorical_features(df, cat_cols):
    """
    Analyze categorical features, their distributions and relationship with target
    """
    print("\n" + "=" * 80)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("=" * 80)

    target_col = "survival_status"

    for col in cat_cols:
        print(f"\nAnalyzing {col}:")

        value_counts = df[col].value_counts()
        print(f"Value counts:\n{value_counts}")

        most_common = value_counts.index[0]
        least_common = value_counts.index[-1]
        print(
            f"Most common value: {most_common} ({value_counts[most_common]} occurrences)"
        )
        print(
            f"Least common value: {least_common} ({value_counts[least_common]} occurrences)"
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        sns.countplot(y=col, data=df, order=value_counts.index, ax=ax1)
        ax1.set_title(f"Distribution of {col}", fontsize=16)
        ax1.set_xlabel("Count", fontsize=14)
        ax1.set_ylabel(col, fontsize=14)

        crosstab = pd.crosstab(df[col], df[target_col], normalize="index") * 100
        crosstab.plot(kind="barh", stacked=True, ax=ax2, legend=True)
        ax2.set_title(f"Relationship between {col} and Survival Status", fontsize=16)
        ax2.set_xlabel("Percentage", fontsize=14)
        ax2.set_ylabel(col, fontsize=14)
        ax2.legend(title="Survival Status")

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/categorical_{col}.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Plot saved as 'categorical_{col}.png'")

        from scipy.stats import chi2_contingency

        contingency_table = pd.crosstab(df[col], df[target_col])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(
            f"Chi-square test p-value: {p:.5f} {'(significant relationship)' if p < 0.05 else '(not significant)'}"
        )


def analyze_numerical_features(df, num_cols):
    """
    Analyze numerical features, their distributions, outliers, and relationship with target
    """
    print("\n" + "=" * 80)
    print("NUMERICAL FEATURES ANALYSIS")
    print("=" * 80)

    target_col = "survival_status"

    for col in num_cols:
        if col == target_col:
            continue

        print(f"\nAnalyzing {col}:")

        stats_data = df[col].describe()
        print(f"Statistics:\n{stats_data}")

        Q1 = stats_data["25%"]
        Q3 = stats_data["75%"]
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

        print(
            f"Outliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.2f}% of data)"
        )
        if len(outliers) > 0:
            print(f"Outlier range: {outliers.min()} to {outliers.max()}")

        fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=False)

        value_range = df[col].max() - df[col].min()

        kde = stats.gaussian_kde(df[col].dropna())
        x_vals = np.linspace(df[col].min(), df[col].max(), 1000)
        kde_vals = kde(x_vals)
        sorted_indices = np.argsort(kde_vals)[::-1]
        cumulative_density = np.cumsum(kde_vals[sorted_indices]) / np.sum(kde_vals)
        top_25_indices = sorted_indices[cumulative_density <= 0.25]
        top_25_range = (
            (x_vals[top_25_indices].min(), x_vals[top_25_indices].max())
            if len(top_25_indices) > 0
            else (x_vals[0], x_vals[0])
        )

        if value_range <= 100:
            sns.histplot(df[col], kde=True, stat="count", binwidth=1, ax=axes[0, 0])
        else:
            sns.histplot(df[col], kde=True, ax=axes[0, 0])
        axes[0, 0].axvspan(
            top_25_range[0],
            top_25_range[1],
            color="yellow",
            alpha=0.3,
            label="Top 25% Density",
        )
        axes[0, 0].set_title(f"Distribution of {col}", fontsize=16)
        axes[0, 0].set_xlabel(col, fontsize=14)
        axes[0, 0].set_ylabel("Frequency", fontsize=14)

        sns.boxplot(x=df[col], ax=axes[0, 1])
        axes[0, 1].set_title(f"Box Plot of {col} with Outliers", fontsize=16)
        axes[0, 1].set_xlabel(col, fontsize=14)

        sns.boxplot(x=target_col, y=col, data=df, ax=axes[1, 0])
        axes[1, 0].set_title(f"Distribution of {col} by Survival Status", fontsize=16)
        axes[1, 0].set_xlabel("Survival Status", fontsize=14)
        axes[1, 0].set_ylabel(col, fontsize=14)

        sns.violinplot(x=target_col, y=col, data=df, ax=axes[1, 1])
        axes[1, 1].set_title(f"Violin Plot of {col} by Survival Status", fontsize=16)
        axes[1, 1].set_xlabel("Survival Status", fontsize=14)
        axes[1, 1].set_ylabel(col, fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/numerical_{col}.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Plot saved as 'numerical_{col}.png'")

        for val in df[target_col].unique():
            group = df[df[target_col] == val][col]
            if group.empty:
                continue
        unique_vals = df[target_col].unique()
        if len(unique_vals) == 2:
            group1 = df[df[target_col] == unique_vals[0]][col]
            group2 = df[df[target_col] == unique_vals[1]][col]
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            print(
                f"T-test p-value: {p_value:.5f} {'(significant difference)' if p_value < 0.05 else '(not significant)'}"
            )


def analyze_correlations(df, num_cols):
    """
    Analyze correlations between numerical features
    """
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    corr_matrix = df[num_cols].corr()
    print("\nCorrelation matrix:")
    print(corr_matrix)

    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        mask=mask,
        cbar_kws={"label": "Correlation Coefficient"},
    )
    plt.title("Correlation Matrix of Numerical Features", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    phik_matrix = df.phik_matrix()
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        phik_matrix.values,
        xticklabels=phik_matrix.columns,
        yticklabels=phik_matrix.index,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Phik Correlation Coefficient"},
        annot_kws={"size": 6},
    )
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Phik Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/phik_correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nCorrelation matrix plot saved as 'correlation_matrix.png'")

    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                high_corr.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )

    if high_corr:
        print("\nHighly correlated feature pairs (|r| > 0.5):")
        for feat1, feat2, corr in high_corr:
            print(f"{feat1} and {feat2}: {corr:.3f}")
    else:
        print("\nNo highly correlated feature pairs found (|r| > 0.5).")


def feature_importance(df, cat_cols, num_cols):
    """
    Analyze feature importance using Random Forest
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X = df.drop(["survival_status"], axis=1)
    y = df["survival_status"]

    cat_features = [col for col in X.columns if col in cat_cols]
    num_features = [
        col for col in X.columns if col in num_cols and col != "survival_status"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model.fit(X, y)

    ohe = model.named_steps["preprocessor"].transformers_[1][1]
    cat_feature_names = ohe.get_feature_names_out(cat_features)
    feature_names = np.concatenate([num_features, cat_feature_names])

    importances = model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", ascending=False)

    print("\nTop 15 most important features:")
    print(importance_df.head(15))

    plt.figure(figsize=(14, 10))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(15))
    plt.title("Feature Importance from Random Forest", fontsize=18)
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nFeature importance plot saved as 'feature_importance.png'")


def dimensionality_reduction_analysis(df, num_cols):
    """
    Perform PCA analysis for visualization
    """
    print("\n" + "=" * 80)
    print("DIMENSIONALITY REDUCTION ANALYSIS (PCA)")
    print("=" * 80)

    features = [col for col in num_cols if col not in ["survival_status"]]

    if len(features) < 2:
        print("Not enough numerical features for PCA visualization.")
        return

    data_for_pca = df[features].copy()
    if data_for_pca.isnull().any().any():
        data_for_pca = data_for_pca.fillna(data_for_pca.mean())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_pca)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(
        {
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "survival_status": df["survival_status"],
        }
    )

    print(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_)*100:.2f}%")

    components_df = pd.DataFrame(
        pca.components_.T, columns=["PC1", "PC2"], index=features
    )
    print("\nFeature contributions to principal components:")
    print(components_df)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="survival_status",
        data=pca_df,
        palette="viridis",
        alpha=0.7,
        s=100,
    )
    plt.title("PCA of Numerical Features", fontsize=18)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=14)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=14)
    plt.legend(title="Survival Status")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/pca_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nPCA visualization saved as 'pca_visualization.png'")


def generate_summary(df, cat_cols, num_cols):
    """
    Generate a summary of findings
    """
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)

    print(f"\n1. Dataset Overview:")
    print(f"   - Total records: {df.shape[0]}")
    print(f"   - Total features: {df.shape[1]-1} (excluding target variable)")
    print(f"   - Categorical features: {len(cat_cols)} columns")
    print(f"   - Numerical features: {len(num_cols)-1} columns (excluding target)")

    target_col = "survival_status"
    for val, count in df[target_col].value_counts().items():
        print(f"   - Survival status {val}: {count} ({count/len(df)*100:.2f}%)")

    print("\n3. Key Insights:")

    cat_imbalance = {}
    for col in cat_cols:
        value_counts = df[col].value_counts(normalize=True)
        imbalance = value_counts.max()
        cat_imbalance[col] = imbalance

    most_imbalanced = sorted(cat_imbalance.items(), key=lambda x: x[1], reverse=True)[
        :3
    ]
    print("   a. Most imbalanced categorical features:")
    for col, imbalance in most_imbalanced:
        print(
            f"      - {col}: {imbalance*100:.2f}% values are '{df[col].value_counts().index[0]}'"
        )

    from scipy.stats import chi2_contingency

    cat_p_values = {}
    for col in cat_cols:
        contingency_table = pd.crosstab(df[col], df[target_col])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        cat_p_values[col] = p

    strongest_cat = sorted(cat_p_values.items(), key=lambda x: x[1])[:3]
    print(
        "\n   b. Categorical features with strongest relationship to survival status:"
    )
    for col, p_value in strongest_cat:
        print(f"      - {col}: p-value = {p_value:.5f}")

    num_p_values = {}
    unique_vals = df[target_col].unique()
    if len(unique_vals) == 2:
        for col in num_cols:
            if col == target_col:
                continue
            group1 = df[df[target_col] == unique_vals[0]][col]
            group2 = df[df[target_col] == unique_vals[1]][col]
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            num_p_values[col] = p_value

        strongest_num = sorted(num_p_values.items(), key=lambda x: x[1])[:3]
        print(
            "\n   c. Numerical features with most significant difference between survival status groups:"
        )
        for col, p_value in strongest_num:
            print(f"      - {col}: p-value = {p_value:.5f}")


def main():
    """
    Main function to run all analyses
    """
    print("\n" + "=" * 80)
    print("BONE MARROW TRANSPLANT DATA EXPLORATION")
    print("=" * 80)

    df, cat_cols, num_cols = load_and_clean_data(
        "data/bone-marrow.arff",
        CATEGORICAL_COLS,
        NUMERICAL_COLS,
        handle_missing=False,
        scale=False,
        onehot=False,
        scale_time_based=True,
        return_cols=True,
    )

    # cat_cols, num_cols = explore_data_types(df)

    analyze_target_distribution(df)

    analyze_numerical_features(df, num_cols)

    analyze_categorical_features(df, cat_cols)

    analyze_correlations(df, num_cols)

    feature_importance(df, cat_cols, num_cols)

    dimensionality_reduction_analysis(df, num_cols)

    generate_summary(df, cat_cols, num_cols)

    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
    print("\nAll analysis plots have been saved to the current directory.")


if __name__ == "__main__":
    main()
