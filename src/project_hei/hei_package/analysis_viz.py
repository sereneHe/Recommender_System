import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_and_save_hist(df, col, outdir):
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"{col} 分布")
    plt.savefig(f"{outdir}/{col}_hist.png")
    plt.close()

def plot_and_save_corr(df, outdir):
    plt.figure(figsize=(12,8))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title("特征相关性热力图")
    plt.savefig(f"{outdir}/correlation_heatmap.png")
    plt.close()

def plot_and_save_pred_vs_true(y_true, y_pred, outdir, label):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(f"{label} 预测 vs 真实值")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.savefig(f"{outdir}/{label}_pred_vs_true.png")
    plt.close()