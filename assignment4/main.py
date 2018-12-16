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
from colorama import Back, Fore, Style
colors = np.array(['green', 'orange', 'blue', ' cyan', 'black','red'])


from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv("./buddymove_holidayiq.csv", sep='\s*,\s*',
                 header=0, encoding='ascii', engine='python')


def get_features():
    features = df.columns.tolist()
    del features[0]
    return features


X = df[get_features()]
colors = np.array(['green', 'orange', 'blue', ' cyan', 'black'])

#### kmeans algorithm
from sklearn.cluster import KMeans
start = time.time()
kmean = KMeans(n_clusters=6, max_iter=500)
kmean.fit(X)
end = time.time()
# print(kmean.labels_)
# print(Fore.BLUE + "k-mean algorithm time is :", end - start)
# print(Fore.RESET)


########## PCA of features for Kmeans
from sklearn.decomposition import PCA
pca_model = PCA(n_components=2)
X_new = pca_model.fit_transform(X)

fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(121)
ax.scatter(X_new[:, 0], X_new[:, 1],c='green', marker='o', s=10)
ax = fig.add_subplot(122)
ax.scatter(X_new[:, 0], X_new[:, 1], c=colors[kmean.labels_], marker='*')
