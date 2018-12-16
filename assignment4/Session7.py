
import numpy as np
import pandas as pd
from colorama import Back, Fore, Style
import time
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
import matplotlib.pyplot as plt

missing_value=['?']
data= pd.read_csv("wiki4HE.csv", na_values=missing_value)
print(data.isnull().sum())

data.dropna(inplace=True)
print(data.shape)
X=data.values
colors = np.array(['green', 'orange', 'blue', ' cyan', 'black'])

#### kmeans algorithm
from sklearn.cluster import KMeans
start = time.time()
kmean = KMeans(n_clusters=3, max_iter=500)
kmean.fit(X)
end = time.time()
print(Fore.BLUE + "k-mean algorithm time is :", end - start)
print(Fore.RESET)

centroids = kmean.cluster_centers_
labels = kmean.labels_
print(centroids)

cluster0 = data.iloc[labels==0, 3]
print('cluster 0: \n', cluster0.value_counts())

print('*' * 50)

cluster1 = data.iloc[labels==1, 3]
print('cluster 1: \n', cluster1.value_counts())

print('*' * 50)

cluster2 = data.iloc[labels==2, 3]
print('cluster 2: \n', cluster2.value_counts())

########## PCA of features for Kmeans
from sklearn.decomposition import PCA
pca_model = PCA(n_components=2)
X_new = pca_model.fit_transform(X)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.scatter(X_new[:, 0], X_new[:, 1],c='green', marker='o', s=10)
ax = fig.add_subplot(122)
ax.scatter(X_new[:, 0], X_new[:, 1], c=colors[kmean.labels_], marker='*')



###### agglomerative algorithm
# linkage : {“ward”, “complete”, “average”}
linkage = "ward"
n_clusters = 2
start = time.time()
model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
model.fit(X)
end = time.time()
print(Fore.BLUE + "aglomerative algorithm time is :", end - start)

print(Fore.RESET)

labels = model.labels_

cluster0 = data.iloc[labels==0, 1]
print('cluster 0: \n', cluster0.value_counts())

print('*' * 50)

cluster1 = data.iloc[labels==1, 1]
print('cluster 1: \n', cluster1.value_counts())

print('*' * 50)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.scatter(X_new[:, 0], X_new[:, 1], c='green', marker='o', s=10)
ax = fig.add_subplot(122)
ax.scatter(X_new[:, 0], X_new[:, 1], c=colors[model.labels_], marker='*')

1+1