import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from sklearn.decomposition import PCA

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

# Give randomMembership


def random_membership(nClusters, nLen):
    membership = pd.DataFrame(index=range(nLen), columns=range(nClusters))
    for i in range(nLen):
        arr = np.random.rand(nClusters)
        newArr = arr / arr.sum()
        membership.iloc[i] = newArr
    return membership

# Update membership using already obtained centroid


def calculateNewMemberShip(distances, nClusters):
    updatedTestMembership = pd.DataFrame(
        index=range(len(distances)), columns=range(nClusters))
    for index, row in distances.iterrows():
        for index2, value in row.items():
            sum = 0
            for index3, value3 in row.items():
                if(value3 == 0 and value == 0):
                    sum = 1
                    break
                elif(value3 == 0):
                    sum = 0
                    break
                else:
                    sum += (value**2)/(value3)**2
            if(sum != 0):
                sum = 1/sum
            updatedTestMembership.iloc[index, index2] = sum
    return updatedTestMembership

# get labels for centroid


def get_labels(distances):
    return distances.idxmin(axis=1)

# Calculate the centroid using membership matrix and the data


def get_centroid(data, membershipMatrix, nClusters):
    centroid = pd.DataFrame(index=range(nClusters), columns=range(len(data.T)))
    denominatorArr = membershipMatrix.apply(lambda x: np.square(x).sum())
    rows, columns = centroid.shape
    for i in range(0, rows):
        denominator = denominatorArr.iloc[i]
        memberSeries = membershipMatrix.iloc[:, i]
        for j in range(0, columns):
            dataSeries = data.iloc[:, j]
            dat = (memberSeries**2)*dataSeries
            centroid.iloc[i, j] = dat.sum()/denominator
    return centroid


def epsilonCalculator(old_membership, membership):
    if old_membership.empty:
        return True
    squares_old = old_membership.apply(lambda df: (df**2).sum(), axis=1)
    squares_new = membership.apply(lambda df: (df**2).sum(), axis=1)
    diff = np.max(np.abs(squares_old-squares_new))
    if diff < 0.01:
        return False
    return True

# Plot


def visualizeCentroids(fuzzycmeans, kMeans):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(fuzzycmeans.T)
    centroids_2d = pca.fit_transform(kMeans.T)
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], marker="x")
    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], marker="*")
    plt.show()


def calculate(features, data, nClusters, max_iterations):
    memberShip = random_membership(nClusters, len(data))
    centroid = get_centroid(data, memberShip, nClusters)
    old_membership = pd.DataFrame()
    old_centroid = pd.DataFrame()
    iteration = 1
    while iteration < max_iterations and not centroid.equals(old_centroid) and epsilonCalculator(old_membership, memberShip):
        old_centroid = centroid
        old_membership = memberShip
        distances = centroid.T.apply(
            lambda x: np.sqrt(((data-x) ** 2).sum(axis=1)))
        labels = get_labels(distances)
        memberShip = calculateNewMemberShip(distances, nClusters)
        centroid = get_centroid(data, memberShip, nClusters)
        iteration += 1
    data.columns = features
    centroid.columns = features
    centroid = centroid.T
    print("Fuzzy C centroids")
    print(centroid)
    centroids_kmeans_dict = {"Risk": [4.734995,
                                      4.762666, 6.217098], "NoFaceContact": [2.707353, 2.855769, 2.751786],
                             "Sick": [3.064706, 4.538462, 9.935714],
                             "HndWshFreq": [3.594118, 7.884615, 5.821429]}
    centroid_kmeans = pd.DataFrame(centroids_kmeans_dict).T
    print("K Means Centroids")
    print(centroid_kmeans)
    visualizeHelper(centroid, centroid_kmeans)


def calculate3(features, data, nClusters, max_iterations):
    memberShip = random_membership(nClusters, len(data))
    centroid = get_centroid(data, memberShip, nClusters)
    old_membership = pd.DataFrame()
    old_centroid = pd.DataFrame()
    iteration = 1
    while iteration < max_iterations and not centroid.equals(old_centroid) and epsilonCalculator(old_membership, memberShip):
        old_centroid = centroid
        old_membership = memberShip
        distances = centroid.T.apply(
            lambda x: np.sqrt(((data-x) ** 2).sum(axis=1)))
        labels = get_labels(distances)
        memberShip = calculateNewMemberShip(distances, nClusters)
        centroid = get_centroid(data, memberShip, nClusters)
        iteration += 1
    data.columns = features
    centroid.columns = features
    centroid = centroid.T
    print(
        f'Dunn index for fuzzy cmeans using 3 clusters and {features[0]}, {features[1]}, {features[2]}, {features[3]} features is {dunn_index(centroid, data, labels)}')


def calculate2(features, data, nClusters, max_iterations):
    memberShip = random_membership(nClusters, len(data))
    centroid = get_centroid(data, memberShip, nClusters)
    old_membership = pd.DataFrame()
    old_centroid = pd.DataFrame()
    iteration = 1
    while iteration < max_iterations and not centroid.equals(old_centroid) and epsilonCalculator(old_membership, memberShip):
        old_centroid = centroid
        old_membership = memberShip
        distances = centroid.T.apply(
            lambda x: np.sqrt(((data-x) ** 2).sum(axis=1)))
        labels = get_labels(distances)
        memberShip = calculateNewMemberShip(distances, nClusters)
        centroid = get_centroid(data, memberShip, nClusters)
        iteration += 1
    data.columns = features
    centroid.columns = features
    centroid = centroid.T
    print(
        f'Dunn index for fuzzy cmeans using 3 clusters and {features[0]}, {features[1]}, {features[2]}, {features[3]},{features[4]} features is {dunn_index(centroid, data, labels)}')


def visualizeHelper(centroid, centroid_kmeans):
    visualizeCentroids(centroid, centroid_kmeans)


max_iterations = 100
df = pd.read_csv("flu_data.csv")
# Question 3a
features = ["Risk", "NoFaceContact", "Sick", "HndWshFreq"]
data = df[features].dropna().drop_duplicates()
data = ((data - data.min())/(data.max()-data.min())) * 9 + 1
data.index = range(len(data))
data.columns = range(len(features))
nClusters = 3
calculate(features, data, nClusters, max_iterations)

# Question 3b
features = ["Risk", "NoFaceContact", "Sick", "HndWshFreq"]
data = df[features].dropna().drop_duplicates()
data = ((data - data.min())/(data.max()-data.min())) * 9 + 1
data.index = range(len(data))
data.columns = range(len(features))
nClusters = 3
calculate3(features, data, nClusters, max_iterations)

# Question 3c
features = ["Risk", "NoFaceContact", "Sick",  "HndWshFreq", "Vaccin"]
data = df[features].dropna().drop_duplicates()
data = ((data - data.min())/(data.max()-data.min())) * 9 + 1
data.index = range(len(data))
data.columns = range(len(features))
nClusters = 3
calculate2(features, data, nClusters, max_iterations)

features = ["Risk", "NoFaceContact", "Sick", "HndWshQual", "HndWshFreq"]
data = df[features].dropna().drop_duplicates()
data = ((data - data.min())/(data.max()-data.min())) * 9 + 1
data.index = range(len(data))
data.columns = range(len(features))
nClusters = 3
calculate2(features, data, nClusters, max_iterations)

features = ["Risk", "NoFaceContact", "Sick", "SociDist", "HndWshFreq"]
data = df[features].dropna().drop_duplicates()
data = ((data - data.min())/(data.max()-data.min())) * 9 + 1
data.index = range(len(data))
data.columns = range(len(features))
nClusters = 3
calculate2(features, data, nClusters, max_iterations)

features = ["Risk", "NoFaceContact", "Sick", "PersnDist", "HndWshFreq"]
data = df[features].dropna().drop_duplicates()
data = ((data - data.min())/(data.max()-data.min())) * 9 + 1
data.index = range(len(data))
data.columns = range(len(features))
nClusters = 3
calculate2(features, data, nClusters, max_iterations)

features = ["Risk", "NoFaceContact", "Sick", "HandSanit", "HndWshFreq"]
data = df[features].dropna().drop_duplicates()
data = ((data - data.min())/(data.max()-data.min())) * 9 + 1
data.index = range(len(data))
data.columns = range(len(features))
nClusters = 3
calculate2(features, data, nClusters, max_iterations)

features = ["Risk", "NoFaceContact", "Sick", "Complications", "HndWshFreq"]
data = df[features].dropna().drop_duplicates()
data = ((data - data.min())/(data.max()-data.min())) * 9 + 1
data.index = range(len(data))
data.columns = range(len(features))
nClusters = 3
calculate2(features, data, nClusters, max_iterations)
