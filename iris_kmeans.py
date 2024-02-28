import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import susi
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# fetch dataset

n_features = 5
iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.9, random_state=42
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Chani's paper hyperparameters
# m = som.map(xdim = 15, ydim = 10, alg = "som", train=2000)
som = susi.SOMClustering(
    n_rows=5,
    n_columns=10,
    n_iter_unsupervised=1000,
    random_state=0,
)
meow = som.fit_transform(X_train)
for m in meow:
    print(m)
exit()
clusters = np.array(som.get_clusters(X_train))
print(type(clusters))

print(clusters.shape)
kmeans = KMeans(n_clusters=2)
label = kmeans.fit_predict(clusters)
# Getting unique labels
u_labels = np.unique(label)
# plotting the results:
exit()
for i in u_labels:
    plt.scatter(clusters[label == i, 0], clusters[label == i, 1], label=i)
plt.legend()
plt.show()
