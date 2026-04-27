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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

print("Loading Kaggle dataset...")
df = pd.read_csv("phishing_kaggle.csv")
df = df.drop(columns=["Index"], errors="ignore")
print(f"Dataset shape: {df.shape}")

X = df.drop("class", axis=1)
y = df["class"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", random_state=42),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, pos_label=1) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred, pos_label=1) * 100, 2),
        "F1 Score":  round(f1_score(y_test, y_pred, pos_label=1) * 100, 2),
        "y_pred":    y_pred,
        "y_test":    y_test,
        "model":     model
    }
    print(f"  Accuracy:  {results[name]['Accuracy']}%")
    print(f"  Precision: {results[name]['Precision']}%")
    print(f"  Recall:    {results[name]['Recall']}%")
    print(f"  F1 Score:  {results[name]['F1 Score']}%")

# Plot 1: Model Comparison Bar Chart
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
model_names = list(results.keys())
x = np.arange(len(metrics))
width = 0.1

fig, ax = plt.subplots(figsize=(16, 7))
colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3","#8C8C8C"]
for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=name, color=colors[i])
ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison - Kaggle Dataset")
ax.set_xticks(x + width * 3.5)
ax.set_xticklabels(metrics)
ax.set_ylim(50, 100)
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("kaggle_model_comparison.png", dpi=150)
print("\nSaved: kaggle_model_comparison.png")

# Plot 2: Accuracy Ranking
acc_series = pd.Series({n: results[n]["Accuracy"] for n in model_names})
acc_series = acc_series.sort_values()
plt.figure(figsize=(10, 6))
bars = plt.barh(acc_series.index, acc_series.values, color="#4C72B0")
plt.xlabel("Accuracy (%)")
plt.title("Model Accuracy Ranking - Kaggle Dataset")
plt.xlim(50, 100)
for bar, val in zip(bars, acc_series.values):
    plt.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f"{val}%", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("kaggle_accuracy_ranking.png", dpi=150)
print("Saved: kaggle_accuracy_ranking.png")

# Plot 3: Confusion Matrix for Random Forest
best_model = "Random Forest"
cm = confusion_matrix(results[best_model]["y_test"], results[best_model]["y_pred"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Phishing","Legitimate"],
            yticklabels=["Phishing","Legitimate"])
plt.title(f"Confusion Matrix - {best_model} (Kaggle Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("kaggle_confusion_matrix_rf.png", dpi=150)
print("Saved: kaggle_confusion_matrix_rf.png")

# Plot 4: Paper vs Our Results Comparison
paper_results = {
    "Logistic Regression": 92.8,
    "Random Forest": 97.3,
    "Decision Tree": 96.6,
    "KNN": 94.4,
    "SVM": 95.3,
    "Naive Bayes": 86.7,
    "Gradient Boosting": 94.6,
    "AdaBoost": 93.5,
}
our_results = {name: results[name]["Accuracy"] for name in model_names}
x = np.arange(len(model_names))
width = 0.35
fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width/2, [paper_results[n] for n in model_names],
               width, label="Paper Results", color="#4C72B0")
bars2 = ax.bar(x + width/2, [our_results[n] for n in model_names],
               width, label="Our Results", color="#DD8452")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Paper vs Our Results Comparison")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha="right")
ax.set_ylim(50, 100)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("paper_vs_our_results.png", dpi=150)
print("Saved: paper_vs_our_results.png")

print("\nAll done! 4 graphs saved!")