import pandas as pd
import numpy as np
from loader import load_and_clean_data
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

df = pd.read_csv("data/clustered_bone_marrow.csv")
df_orig = load_and_clean_data(
    "data/bone-marrow.arff",
    cat_cols=[],
    num_cols=[
        "survival_time",
    ],
    handle_missing=False,
    scale=False,
    onehot=False,
)

df["Cluster"] = df["Cluster"].astype(int)
n_clusters = df["Cluster"].nunique()

T = df_orig["survival_time"]
E = df["survival_status"]

# Kaplan-Meier survival curves per cluster
plt.figure(figsize=(8, 6))
kmf = KaplanMeierFitter()
for i in range(n_clusters):
    mask = df["Cluster"] == i
    kmf.fit(T[mask], event_observed=E[mask], label=f"Cluster {i}")
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][
        i % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    ]
    kmf.plot_survival_function(ci_show=False, color=color)
plt.title("Kaplan-Meier Survival Curves by Cluster")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.legend()
plt.tight_layout()
plt.savefig("results/km_survival_curves.png")

# Cox Proportional Hazards model
df_cox = df.copy()
df_cox["Cluster"] = df["Cluster"]
df_cox["survival_time"] = T
df_cox["survival_status"] = E

df_cox = pd.get_dummies(df_cox, columns=["Cluster"], drop_first=True)

cph = CoxPHFitter(penalizer=0.1)
cph.fit(df_cox, duration_col="survival_time", event_col="survival_status")
print("Cox Regression Feature Importances (coefficients):")
print(cph.summary[["coef", "exp(coef)", "p"]])

plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    mean_row = df_cox.mean(numeric_only=True)
    cluster_cols = [col for col in df_cox.columns if col.startswith("Cluster_")]
    mean_row[cluster_cols] = 0
    if i > 0 and f"Cluster_{i}" in mean_row.index:
        mean_row[f"Cluster_{i}"] = 1
    surv = cph.predict_survival_function(mean_row.to_frame().T)
    plt.plot(surv.index, surv.values.flatten(), label=f"Cluster {i}")
plt.title("Cox Model Predicted Survival by Cluster")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.legend()
plt.tight_layout()
plt.savefig("results/cox_predicted_survival.png")
