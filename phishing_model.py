import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Loading large dataset...")
df = pd.read_csv("phishing_dataset.csv")
print(f"Dataset shape: {df.shape}")

X = df.drop("phishing", axis=1)
y = df["phishing"]

# Same normalization as 11k
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Same 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Subset for SVM and KNN
X_small, _, y_small, _ = train_test_split(X_scaled, y, train_size=20000, random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_small, y_small, test_size=0.2, random_state=42)

print(f"Full - Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Small - Train: {len(X_train_s)}, Test: {len(X_test_s)}")

# Same 7 models as paper
full_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}

small_models = {
    "SVM": SVC(kernel="rbf", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

results = {}

for name, model in full_models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, pos_label=1) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred, pos_label=1) * 100, 2),
        "F1 Score":  round(f1_score(y_test, y_pred, pos_label=1) * 100, 2),
        "y_pred":    y_pred,
        "y_test":    y_test
    }
    print(f"  Accuracy:  {results[name]['Accuracy']}%")
    print(f"  Precision: {results[name]['Precision']}%")
    print(f"  Recall:    {results[name]['Recall']}%")
    print(f"  F1 Score:  {results[name]['F1 Score']}%")

for name, model in small_models.items():
    print(f"\nTraining {name} (20k subset)...")
    model.fit(X_train_s, y_train_s)
    y_pred = model.predict(X_test_s)
    results[name] = {
        "Accuracy":  round(accuracy_score(y_test_s, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test_s, y_pred, pos_label=1) * 100, 2),
        "Recall":    round(recall_score(y_test_s, y_pred, pos_label=1) * 100, 2),
        "F1 Score":  round(f1_score(y_test_s, y_pred, pos_label=1) * 100, 2),
        "y_pred":    y_pred,
        "y_test":    y_test_s
    }
    print(f"  Accuracy:  {results[name]['Accuracy']}%")
    print(f"  Precision: {results[name]['Precision']}%")
    print(f"  Recall:    {results[name]['Recall']}%")
    print(f"  F1 Score:  {results[name]['F1 Score']}%")

# Plot 1: Model Comparison
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
model_names = list(results.keys())
x = np.arange(len(metrics))
width = 0.1

fig, ax = plt.subplots(figsize=(16, 7))
colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3"]
for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=name, color=colors[i])
ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison - Large Dataset (88k)")
ax.set_xticks(x + width * 3)
ax.set_xticklabels(metrics)
ax.set_ylim(60, 100)
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("large_model_comparison.png", dpi=150)
print("\nSaved: large_model_comparison.png")

# Plot 2: Accuracy Ranking
acc_series = pd.Series({n: results[n]["Accuracy"] for n in model_names})
acc_series = acc_series.sort_values()
plt.figure(figsize=(10, 6))
bars = plt.barh(acc_series.index, acc_series.values, color="#4C72B0")
plt.xlabel("Accuracy (%)")
plt.title("Model Accuracy Ranking - Large Dataset (88k)")
plt.xlim(60, 100)
for bar, val in zip(bars, acc_series.values):
    plt.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f"{val}%", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("large_accuracy_ranking.png", dpi=150)
print("Saved: large_accuracy_ranking.png")

# Plot 3: Cross Dataset Comparison
kaggle_results = {
    "Logistic Regression": 93.35,
    "Random Forest": 96.92,
    "Decision Tree": 96.02,
    "KNN": 93.98,
    "SVM": 94.98,
    "Gradient Boosting": 94.93,
    "AdaBoost": 93.89,
}
large_results = {name: results[name]["Accuracy"] for name in model_names}
x = np.arange(len(model_names))
width = 0.35
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width/2, [kaggle_results[n] for n in model_names],
       width, label="Kaggle Dataset (11k)", color="#4C72B0")
ax.bar(x + width/2, [large_results[n] for n in model_names],
       width, label="Large Dataset (88k)", color="#DD8452")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Cross Dataset Comparison (11k vs 88k)")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha="right")
ax.set_ylim(60, 100)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("cross_dataset_comparison.png", dpi=150)
print("Saved: cross_dataset_comparison.png")

print("\nAll done!")