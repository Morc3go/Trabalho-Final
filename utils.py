import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def process_data(data):
    numeric_data = data.select_dtypes(include=['number'])

    fig = plt.figure(figsize=(10, 8))
    data.hist(bins=30, figsize=(10, 8))
    hist_path = 'datasets/databases/histogram.png'
    fig.savefig(hist_path)

    corr_fig = plt.figure(figsize=(10, 8))
    if not numeric_data.empty:
        corr = numeric_data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        corr_path = 'datasets/databases/correlation.png'
        corr_fig.savefig(corr_path)
    else:
        corr_path = None

    boxplot_fig = plt.figure(figsize=(10, 8))
    if not numeric_data.empty:
        sns.boxplot(data=numeric_data)
        boxplot_path = 'datasets/databases/boxplot.png'
        boxplot_fig.savefig(boxplot_path)
    else:
        boxplot_path = None

    return hist_path, corr_path, boxplot_path
