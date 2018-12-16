import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
from colorama import Back, Fore, Style
import time
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from colorama import Back, Fore, Style
from sklearn import metrics
colors = np.array(['green', 'orange', 'blue', ' cyan', 'black','red'])


from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv("./buddymove_holidayiq.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')


def get_features():
    features = df.columns.tolist()
    del features[0]
    return features


colors = np.array(['green', 'orange', 'blue', ' cyan', 'black'])




def save_kmeans(X):
    start = time.time()
    kmean = KMeans(n_clusters=6, max_iter=500)
    kmean.fit(X)
    labels = kmean.labels_
    silhouette_score= metrics.silhouette_score(X, labels, metric='euclidean')
    end = time.time()
    print(silhouette_score)
    f = open("./results/kmeans.txt", "w")
    f.write(f"silhouette_score : {silhouette_score}\n")

def save_linkage(X):
    result =""
    linkage_array=["ward", "average", "single", "complete"]
    n_clusters = 2
    for linkage in linkage_array :
        model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
        model.fit(X)
        labels = model.labels_
        silhouette_score= metrics.silhouette_score(X, labels, metric='euclidean')
        result += f"linkage : {linkage}\nsilhouette_score : {silhouette_score}\n\n"
    f = open("./results/linkage.txt", "w")
    f.write(result)

X = df[get_features()]
save_linkage(X)