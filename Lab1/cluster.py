from loader import load_and_clean_data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

# Elbow method for optimal k
# inertias = []
# k_range = range(1, 16)
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=99)
#     kmeans.fit(df.values)
#     inertias.append(kmeans.inertia_)

# plt.figure(figsize=(7, 4))
# plt.plot(k_range, inertias, marker="o")
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Inertia")
# plt.title("Elbow Method For Optimal k")
# plt.xticks(k_range)
# plt.tight_layout()
# plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.values)

n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=99)
clusters = kmeans.fit_predict(X_pca)

df["Cluster_PCA"] = clusters

plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], label=f"Cluster {i}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("K-means Clusters (PCA-reduced)")
plt.legend()
plt.tight_layout()
plt.savefig("results/kmeans_pca_clusters.png")

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

df_orig = load_and_clean_data(
    "data/bone-marrow.arff",
    correlation_based["categorical"],
    num_cols=correlation_based["numerical"],
    handle_missing=False,
    onehot=False,
    scale=False,
)

df_orig["Cluster"] = df["Cluster_PCA"]

df_orig.to_csv("data/clustered_bone_marrow_full.csv", index=False)

df_ordered = df_orig.sort_values(by="Cluster")
df_ordered.to_csv("data/clustered_bone_marrow_ordered.csv", index=False)
