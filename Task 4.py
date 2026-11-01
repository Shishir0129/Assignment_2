"""Task 4: Compare Naive Bayes and Decision Tree across three data splits."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Task 2" / "sand_shale_dataset.csv"
OUTPUT_DIR = BASE_DIR / "Task 4"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the actual dataset
df = pd.read_csv(DATA_FILE)
features = ["f1", "f2", "f3", "f4", "cat"]
X = pd.get_dummies(df[features], columns=["cat"], dtype=float)
y = df["y"].astype(int)

print("Dataset:", len(df), "observations")
print("Class counts:", y.value_counts().sort_index().to_dict())
print("Features:", list(X.columns))

results = []

def evaluate_split(split_name, seed, train_idx, test_idx):
    """Train and evaluate both classifiers on one split."""
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=seed)
    }

    for model_name, model in models.items():
        warning_text = ""

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            if caught:
                warning_text = "; ".join(
                    str(w.message) for w in caught
                )

        error_rate = np.mean(predictions != y_test)

        results.append({
            "split": split_name,
            "seed": seed,
            "model": model_name,
            "train_total": len(train_idx),
            "train_y1": int(y_train.sum()),
            "train_y1_percent": 100 * y_train.mean(),
            "test_total": len(test_idx),
            "test_y1": int(y_test.sum()),
            "test_y1_percent": 100 * y_test.mean(),
            "test_error_rate": error_rate,
            "test_error_percent": 100 * error_rate,
            "warnings": warning_text
        })

        print(
            f"{split_name}, seed={seed}, {model_name}: "
            f"test error={error_rate:.3f}"
        )

# Split A: all minority-class observations in training.
# All remaining observations are placed in the test set.
train_a = np.where(y.to_numpy() == 1)[0]
test_a = np.where(y.to_numpy() == 0)[0]
evaluate_split("A_biased", 0, train_a, test_a)

# Splits B and C: five different random seeds each.
for seed in [1, 2, 3, 4, 5]:
    indices = np.arange(len(df))

    # Split B: random 80/20 split without stratification
    train_b, test_b = train_test_split(
        indices, test_size=0.20, random_state=seed
    )
    evaluate_split("B_random", seed, train_b, test_b)

    # Split C: stratified random 80/20 split
    train_c, test_c = train_test_split(
        indices, test_size=0.20, random_state=seed, stratify=y
    )
    evaluate_split("C_stratified", seed, train_c, test_c)

# Save detailed results
results_df = pd.DataFrame(results)
results_file = OUTPUT_DIR / "task4_results.csv"
results_df.to_csv(results_file, index=False)

print("\nDETAILED RESULTS")
print(results_df[
    ["split", "seed", "model", "train_y1_percent",
     "test_y1_percent", "test_error_percent", "warnings"]
].to_string(index=False))

# Summarise mean and standard deviation across seeds.
# Split A has one run, so its standard deviation is zero.
summary = (
    results_df.groupby(["split", "model"])["test_error_percent"]
    .agg(["mean", "std"])
    .reset_index()
)
summary["std"] = summary["std"].fillna(0)
summary.to_csv(OUTPUT_DIR / "task4_summary.csv", index=False)

print("\nMEAN TEST ERROR (%) AND STANDARD DEVIATION")
print(summary.to_string(index=False))

# Plot test error comparison
split_order = ["A_biased", "B_random", "C_stratified"]
model_order = ["Naive Bayes", "Decision Tree"]

fig, ax = plt.subplots(figsize=(9, 6))
positions = np.arange(len(split_order))
width = 0.35

for i, model_name in enumerate(model_order):
    model_data = (
        summary[summary["model"] == model_name]
        .set_index("split")
        .reindex(split_order)
    )

    ax.bar(
        positions + (i - 0.5) * width,
        model_data["mean"],
        width,
        yerr=model_data["std"],
        capsize=5,
        label=model_name
    )

ax.set_xticks(positions)
ax.set_xticklabels(["A: Biased", "B: Random", "C: Stratified"])
ax.set_ylabel("Test error rate (%)")
ax.set_title("Task 4: Classifier sensitivity to data splitting")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

plot_file = OUTPUT_DIR / "task4_error_comparison.png"
fig.savefig(plot_file, dpi=200)
plt.close(fig)

print("\nSaved files:")
print(results_file)
print(OUTPUT_DIR / "task4_summary.csv")
print(plot_file)
