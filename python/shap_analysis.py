import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

# load the cleaned data
df = pd.read_csv("data/processed/diabetes_clean.csv")

# same preprocessing as R - binary target
df["readmitted_binary"] = df["readmitted_binary"].astype(int)

# separate features and target
X = df.drop(columns=["readmitted_binary"])
y = df["readmitted_binary"]

# one-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# same 70/30 split with same seed as R
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("train size:", X_train.shape)
print("test size: ", X_test.shape)

# manually balance training set - 10k each class
train_df = X_train.copy()
train_df["target"] = y_train.values

no_class  = train_df[train_df["target"] == 0].sample(n=10000, random_state=42)
yes_class = train_df[train_df["target"] == 1].sample(n=10000, random_state=42, replace=True)
train_balanced = pd.concat([no_class, yes_class]).sample(frac=1, random_state=42)

X_bal = train_balanced.drop(columns=["target"])
y_bal = train_balanced["target"]

print("balanced training size:", X_bal.shape)
print("class balance:\n", y_bal.value_counts(normalize=True).round(3))

# train xgboost
dtrain = xgb.DMatrix(X_bal, label=y_bal)
dtest  = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "max_depth":        3,
    "eta":              0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "seed":             42
}

model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)

# test AUC
from sklearn.metrics import roc_auc_score
preds = model.predict(dtest)
auc   = roc_auc_score(y_test, preds)
print(f"\ntest AUC: {auc:.3f}")

# SHAP analysis
print("\ncomputing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 1. summary plot - shows most important features overall
plt.figure()
shap.summary_plot(shap_values, X_test, max_display=15,
                  show=False, plot_type="bar")
plt.title("feature importance - mean SHAP value")
plt.tight_layout()
plt.savefig("outputs/figures/10_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved 10_shap_importance.png")

# 2. beeswarm plot - shows direction and magnitude of each feature
plt.figure()
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.title("SHAP summary - feature impact on readmission")
plt.tight_layout()
plt.savefig("outputs/figures/11_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved 11_shap_beeswarm.png")

# 3. top feature names and mean absolute SHAP values
shap_df = pd.DataFrame({
    "feature":    X_test.columns,
    "mean_shap":  np.abs(shap_values).mean(axis=0)
}).sort_values("mean_shap", ascending=False).head(15)

print("\ntop 15 features by SHAP:")
print(shap_df.to_string(index=False))

shap_df.to_csv("outputs/results/shap_values.csv", index=False)
print("\nSHAP values saved to outputs/results/shap_values.csv")