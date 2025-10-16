from loader import load_and_clean_data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns

{
    "categorical": [
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
    ],
    "numerical": [
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
    ],
}

removed_attributes_correlation_based = {
    "categorical": [
        "Recipientgender",
        "IIIV",
        "Gendermatch",
        "DonorABO",
        "RecipientABO",
        "ABOmatch",
        "CMVstatus",
        "DonorCMV",
        "Antigen",
        "Alel",
        "HLAgrI",
    ],
    "numerical": [],
}


correlation_based = {
    "categorical": [
        "Stemcellsource",
        "Donorage35",
        "RecipientRh",
        "RecipientCMV",
        "Disease",
        "Riskgroup",
        "Txpostrelapse",
        "Diseasegroup",
        "HLAmatch",
        "HLAmismatch",
        "Recipientage10",
        "Recipientageint",
        "Relapse",
        "aGvHDIIIIV",
        "extcGvHD",
    ],
    "numerical": [
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
    ],
}

removed_removed_redundant = {
    "categorical": [
        "Donorage35",
        "Diseasegroup",
    ],
    "numerical": [
        "Rbodymass",
    ],
}

removed_redundant = {
    "categorical": [
        "Stemcellsource",
        "RecipientRh",
        "RecipientCMV",
        "Disease",
        "Riskgroup",
        "Txpostrelapse",
        "HLAmatch",
        "HLAmismatch",
        "Recipientage10",
        "Recipientageint",
        "Relapse",
        "aGvHDIIIIV",
        "extcGvHD",
    ],
    "numerical": [
        "Donorage",
        "Recipientage",
        "CD34kgx10d6",
        "CD3dCD34",
        "CD3dkgx10d8",
        "ANCrecovery",
        "PLTrecovery",
        "time_to_aGvHD_III_IV",
        "survival_time",
        "survival_status",
    ],
}

df_cleaned = load_and_clean_data(
    "data/bone-marrow.arff",
    removed_redundant["categorical"],
    num_cols=removed_redundant["numerical"],
)

df = df_cleaned.drop(columns=["survival_time", "survival_status"])

# Elbow method for optimal k, when the inertia begins to slow down
inertias = []
k_range = range(1, 16)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=99)
    kmeans.fit(df.values)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(k_range, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.xticks(k_range)
plt.tight_layout()
plt.savefig("results/elbow_method.png")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.values)

n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=99)
clusters = kmeans.fit_predict(X_pca)

df["Cluster_PCA"] = clusters

plt.figure(figsize=(8, 6))
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i in range(n_clusters):
    color = default_colors[i % len(default_colors)]
    plt.scatter(
        X_pca[clusters == i, 0],
        X_pca[clusters == i, 1],
        label=f"Cluster {i}",
        color=color,
    )
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("K-means Clusters (PCA-reduced)")
plt.legend()
plt.tight_layout()
plt.savefig("results/kmeans_pca_clusters.png")
plt.close()

print("Cluster counts (PCA-reduced):")
print(df["Cluster_PCA"].value_counts())

kmeans_orig = KMeans(n_clusters=n_clusters, random_state=99)
clusters_orig = kmeans_orig.fit_predict(df.values)

df["Cluster_Orig"] = clusters_orig

print("Cluster counts (Original features):")
print(df["Cluster_Orig"].value_counts())

df["Cluster"] = df["Cluster_PCA"]
df["survival_time"] = df_cleaned["survival_time"]
df["survival_status"] = df_cleaned["survival_status"]

df.to_csv("data/clustered_bone_marrow.csv", index=False)

df_orig, cat_cols, num_cols = load_and_clean_data(
    "data/bone-marrow.arff",
    removed_redundant["categorical"],
    num_cols=removed_redundant["numerical"],
    handle_missing=False,
    onehot=False,
    scale=False,
    scale_time_based=True,
    return_cols=True,
)

df_orig["Cluster"] = df["Cluster_PCA"]

df_orig.to_csv("data/clustered_bone_marrow_full.csv", index=False)

df_ordered = df_orig.sort_values(by="Cluster")
df_ordered.to_csv("data/clustered_bone_marrow_ordered.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cd",
        dest="cluster_details",
        action="store_true",
        default=True,
        help="Generate cluster details plots (default: True)",
    )
    parser.add_argument(
        "--no-cd",
        dest="cluster_details",
        action="store_false",
        help="Disable cluster details plots",
    )
    parser.add_argument(
        "--cc",
        dest="cluster_compare",
        action="store_true",
        default=True,
        help="Generate cluster comparison plots (default: True)",
    )
    parser.add_argument(
        "--no-cc",
        dest="cluster_compare",
        action="store_false",
        help="Disable cluster comparison plots",
    )
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    num_col_stats = {}
    for col in num_cols:
        vals = df_orig[col].dropna()
        mean = vals.mean()
        std = vals.std()
        y_min = min(mean - std, vals.min(), 0)
        y_max = max(mean + std, vals.max())
        num_col_stats[col] = {"ymin": y_min, "ymax": y_max}

    cat_col_categories = {}
    for col in cat_cols:
        cat_col_categories[col] = sorted(df_orig[col].dropna().unique())

    # plot for each cluster details about all features
    if args.cluster_details:
        for cluster_id in sorted(df_orig["Cluster"].unique()):
            cluster_df = df_orig[df_orig["Cluster"] == cluster_id]
            n_num = len(num_cols)
            n_cat = len(cat_cols)
            n_total = n_num + n_cat

            ncols = 3
            nrows = int(np.ceil(n_total / ncols))

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 2.5 * nrows), constrained_layout=True
            )
            axes = axes.flatten() if n_total > 1 else [axes]

            for idx, col in enumerate(num_cols):
                ax = axes[idx]
                vals = cluster_df[col].dropna()
                mean = vals.mean()
                std = vals.std()
                ax.bar([col], [mean], yerr=[std], capsize=8, color="skyblue")
                ax.set_ylabel("Mean ± Std")
                ax.set_title(f"{col} (Numerical)")
                ax.set_ylim(num_col_stats[col]["ymin"], num_col_stats[col]["ymax"])

            for idx, col in enumerate(cat_cols):
                ax = axes[n_num + idx]
                all_cats = cat_col_categories[col]
                counts = cluster_df[col].value_counts(normalize=True)
                proportions = [counts.get(cat, 0.0) for cat in all_cats]
                ax.bar([str(cat) for cat in all_cats], proportions, color="orange")
                ax.set_ylabel("Proportion")
                ax.set_title(f"{col} (Categorical)")
                ax.set_ylim(0, 1)

            for ax in axes[n_total:]:
                ax.axis("off")

            fig.suptitle(f"Cluster {cluster_id} Details", fontsize=16)
            plt.savefig(f"results/cluster_{cluster_id}_details.png")
            plt.close(fig)

    # compare the worst cluster with the middle one and the best one
    if args.cluster_compare:
        compare_clusters = [7, 5, 3]
        compare_labels = [f"Cluster {cid}" for cid in compare_clusters]
        palette = sns.color_palette("Set2", n_colors=3)

        n_num_features = len(num_cols)
        ncols = 3
        nrows = int(np.ceil(n_num_features / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True
        )
        axes = axes.flatten()
        for idx, col in enumerate(num_cols):
            means = []
            stds = []
            for cid in compare_clusters:
                vals = df_orig[df_orig["Cluster"] == cid][col].dropna()
                means.append(vals.mean())
                stds.append(vals.std())
            x = np.arange(len(compare_clusters))
            axes[idx].bar(
                x,
                means,
                yerr=stds,
                capsize=6,
                color=palette,
                tick_label=compare_labels,
            )
            axes[idx].set_title(col)
            axes[idx].set_ylabel("Mean ± Std")
            axes[idx].set_ylim(num_col_stats[col]["ymin"], num_col_stats[col]["ymax"])
        for ax in axes[n_num_features:]:
            ax.axis("off")
        fig.suptitle("Numerical Features: Clusters 7, 5, 3", fontsize=16)
        plt.savefig("results/compare_clusters_7_5_3_numerical_all.png")
        plt.close(fig)

        all_cats_per_col = [cat_col_categories[col] for col in cat_cols]
        n_cat_features = len(cat_cols)
        ncols = 3
        nrows = int(np.ceil(n_cat_features / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True
        )
        axes = axes.flatten()
        for idx, col in enumerate(cat_cols):
            all_cats = cat_col_categories[col]
            proportions = []
            for cid in compare_clusters:
                counts = df_orig[df_orig["Cluster"] == cid][col].value_counts(
                    normalize=True
                )
                proportions.append([counts.get(cat, 0.0) for cat in all_cats])
            proportions = np.array(proportions)
            x = np.arange(len(all_cats))
            width = 0.22
            for i, (prop, label, color) in enumerate(
                zip(proportions, compare_labels, palette)
            ):
                axes[idx].bar(
                    x + i * width,
                    prop,
                    width=width,
                    label=label if idx == 0 else "",
                    color=color,
                )
            axes[idx].set_xticks(x + width)
            axes[idx].set_xticklabels(
                [str(cat) for cat in all_cats], rotation=30, ha="right"
            )
            axes[idx].set_ylim(0, 1)
            axes[idx].set_title(col)
            if idx % ncols == 0:
                axes[idx].set_ylabel("Proportion")
        for ax in axes[n_cat_features:]:
            ax.axis("off")
        if n_cat_features > 0:
            axes[0].legend()
        fig.suptitle("Categorical Features: Clusters 7, 5, 3", fontsize=16)
        plt.savefig("results/compare_clusters_7_5_3_categorical_all.png")
        plt.close(fig)

        # export comparison data for clusters 7, 5, 3 to csv
        import pandas as pd

        num_data = []
        for col in num_cols:
            for cid in compare_clusters:
                vals = df_orig[df_orig["Cluster"] == cid][col].dropna()
                num_data.append(
                    {
                        "feature": col,
                        "cluster": cid,
                        "mean": vals.mean(),
                        "std": vals.std(),
                    }
                )
        num_df = pd.DataFrame(num_data)
        num_df.to_csv("results/compare_clusters_7_5_3_numerical.csv", index=False)

        cat_data = []
        for col in cat_cols:
            all_cats = cat_col_categories[col]
            for cid in compare_clusters:
                counts = df_orig[df_orig["Cluster"] == cid][col].value_counts(
                    normalize=True
                )
                for cat in all_cats:
                    cat_data.append(
                        {
                            "feature": col,
                            "category": cat,
                            "cluster": cid,
                            "proportion": counts.get(cat, 0.0),
                        }
                    )
        cat_df = pd.DataFrame(cat_data)
        cat_df.to_csv("results/compare_clusters_7_5_3_categorical.csv", index=False)
