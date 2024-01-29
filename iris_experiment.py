import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import susi
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix

# fetch dataset

n_features = 5
iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Chani's paper hyperparameters
# m = som.map(xdim = 15, ydim = 10, alg = "som", train=2000)
som = susi.SOMClassifier(
    n_rows=10,
    n_columns=15,
    n_iter_unsupervised=2000,
    random_state=0,
)
som.fit(X_train, y_train)
plt.show()
u_matrix = som.get_u_matrix()
print(u_matrix)
estimation = som.get_estimation_map()
print(som.predict)
# plot_umatrix(u_matrix, 10, 15)
susi.SOMPlots.plot_estimation_map(estimation)
plt.show()

# TODO
# labels = iris.target
# data = pd.DataFrame(iris.data[::4])
# data.columns = iris.feature_names


# get the data
