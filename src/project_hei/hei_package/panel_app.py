import panel as pn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pn.extension()
df = pd.read_csv("你的数据路径.csv")

def plot_hist(col):
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"{col} 分布")
    return pn.pane.Matplotlib(plt.gcf(), tight=True)

feature_selector = pn.widgets.Select(name='特征', options=list(df.columns))
interactive_plot = pn.bind(plot_hist, col=feature_selector)
pn.Column(feature_selector, interactive_plot).servable()