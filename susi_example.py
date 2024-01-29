import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# --- for running the script without pip
import sys

sys.path.append("../")
# ---

import susi
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix


X, y = make_blobs(n_samples=100, n_features=2, centers=3)
plt.scatter([x[0] for x in X], [x[1] for x in X], c=y)
plt.show()


som = susi.SOMClustering(n_rows=30, n_columns=30)
som.fit(X)
print("SOM fitted!")


u_matrix = som.get_u_matrix()
plot_umatrix(u_matrix, 30, 30)

plt.show()
