import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPORT_DIR = "/Users/xiaoyuhe/Recommender_System/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def plot_feature_importance(importance_csv, out_name="feature_importance.png"):
    """
    Draws a barplot of feature importances from a CSV file.
    CSV must have columns: feature, importance
    """
    df = pd.read_csv(importance_csv)
    df = df.sort_values("importance", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=df, palette="viridis")
    plt.title("Feature Importance")
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, out_name)
    plt.savefig(out_path)
    print(f"Saved feature importance plot to {out_path}")
    plt.close()

def plot_error_distribution(errors_csv, out_name="error_distribution.png"):
    """
    Draws a histogram of prediction errors from a CSV file.
    CSV must have a column: error
    """
    df = pd.read_csv(errors_csv)
    plt.figure(figsize=(8, 5))
    sns.histplot(df["error"], bins=30, kde=True)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, out_name)
    plt.savefig(out_path)
    print(f"Saved error distribution plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    # Example usage:
    # plot_feature_importance("feature_importance.csv")
    # plot_error_distribution("errors.csv")
    pass
