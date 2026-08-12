import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("HEI 数据交互式分析")

df = pd.read_csv("你的数据路径.csv")
feature = st.selectbox("选择特征", df.columns)
fig, ax = plt.subplots()
sns.histplot(df[feature].dropna(), kde=True, ax=ax)
st.pyplot(fig)

if st.button("显示相关性热力图"):
    fig2, ax2 = plt.subplots(figsize=(12,8))
    sns.heatmap(df.corr(), cmap='coolwarm', center=0, ax=ax2)
    st.pyplot(fig2)