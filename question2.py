import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

# Assign random centroids initially


def random_centroids(dataSet, k):
    centroids = []
    for i in range(k):
        centroid = dataSet.apply(lambda col: float(col.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

# Get labels for the data points based on centroids


def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data-x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)

# Compute new centroids


def new_centroids(data, labels, k):
    return cleanedDataSet.groupby(labels).apply(lambda x: x.mean()).T

# Plot


def show_plot(centroid, features, data, clusters):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0, clusters):
        cl = data[labels == i]
        ax.scatter(cl.loc[:, features[0]], cl.loc[:, features[1]], cl.loc[:,
                   features[2]], color=sns.color_palette()[i], label=f'Cluster {i}')
        ax.scatter(centroid[i][features[0]], centroid[i][features[1]],
                   centroid[i][features[2]], color="black", label=f'Centroid{i}')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.legend()
    plt.show()

# Dunn index


def dunn_index(centroid, data, labels):
    num = sys.float_info.max
    for i in range(len(centroid.T)):
        for j in range(len(centroid.T)):
            if(i != j):
                dat = data[labels == i]
                dat2 = data[labels == j]
                for index, row in dat.iterrows():
                    for index2, row2 in dat2.iterrows():
                        distance = np.linalg.norm(row-row2)
                        if distance < num:
                            num = distance
    den = sys.float_info.min
    for i in range(len(centroid.T)):
        dat = data[labels == i]
        for index, row in dat.iterrows():
            for index2, row2 in dat.iterrows():
                distance = np.linalg.norm(row-row2)
                if distance > num:
                    den = distance
    return num/den


df = pd.read_csv("flu_data.csv")


# Question 2a
features = ["Risk", "NoFaceContact", "Sick", "HndWshFreq"]
dataset = df[features]
cleanedDataSet = dataset.dropna().drop_duplicates()
cleanedDataSet = ((cleanedDataSet - cleanedDataSet.min()) /
                  (cleanedDataSet.max()-cleanedDataSet.min())) * 9 + 1
max_iterations = 100
c = 3
centroids = random_centroids(cleanedDataSet, c)
old_centroids = pd.DataFrame()
iteration = 1
while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    labels = get_labels(cleanedDataSet, centroids)
    centroids = new_centroids(cleanedDataSet, labels, c)
    iteration += 1
print(
    f'Dunn index for 3 clusters using Risk, NoFaceContact, Sick, HndWshFreq is {dunn_index(centroids, cleanedDataSet, labels)}')

# Question 2b
features = ["Risk", "NoFaceContact", "Sick", "HndWshFreq", "HndWshQual"]
dataset = df[features]
cleanedDataSet = dataset.dropna().drop_duplicates()
cleanedDataSet = ((cleanedDataSet - cleanedDataSet.min()) /
                  (cleanedDataSet.max()-cleanedDataSet.min())) * 9 + 1
max_iterations = 100
c = 3
centroids = random_centroids(cleanedDataSet, c)
old_centroids = pd.DataFrame()
iteration = 1
while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    labels = get_labels(cleanedDataSet, centroids)
    centroids = new_centroids(cleanedDataSet, labels, c)
    iteration += 1
print(
    f'Dunn index for 3 clusters using Risk, NoFaceContact, Sick, HndWshFreq, HndWshQual features is {dunn_index(centroids, cleanedDataSet, labels)}')
