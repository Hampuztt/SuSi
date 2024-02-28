import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
import susi
from susi.SOMPlots import plot_umatrix, plot_estimation_map
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix

SEED_DATA_PATH = "datasets/seeds.txt"


def read_and_process_seeds_data(filepath):
    """
    Reads and processes data from a given file.

    Note: This function also handles cases where the input file contains double tabs,
    which could lead to 'empty' elements in the data. These 'empty' elements are
    removed before appending each line to the data array.
    """
    data = []
    with open(filepath, "r") as file:
        for line in file:
            data.append([float(num) for num in line.split("\t") if num])

    labels = np.array(data)[:, 7].astype(int)
    data = pd.DataFrame(
        np.delete(data, 7, axis=1),
        columns=[
            "area",
            "perimeter",
            "compactness",
            "length of kernel",
            "width of kernel",
            "asymmetry coefficent",
            "length of kernel groove",
        ],
    )

    return data, labels


def read_and_load_iris_data():
    iris = datasets.load_iris()
    labels = iris.target
    data = pd.DataFrame(iris.data[:, :4])
    data.columns = iris.feature_names
    return data, labels


from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score


if __name__ == "__main__":
    data, labels = read_and_process_seeds_data(SEED_DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=1, shuffle=True
    )
    print(data)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # X, y = data.values, labels
    som = susi.SOMClustering(n_rows=30, n_columns=30)
    som.fit(X)
    clusters = som.get_clusters(X)
    # Assuming `clusters` is your 2D array output from the SOM
    # Let's say you decide to further partition these clusters into k clusters
    dbi_scores = []
    n_clusters_options = range(2, 11)  # Testing a range from 2 to 10 clusters
    for i in n_clusters_options:
        kmeans = KMeans(n_clusters=i, random_state=42)
        labels = kmeans.fit_predict(clusters)
        dbi = davies_bouldin_score(clusters, labels)
        dbi_scores.append(dbi)

    # Plotting the DBI scores
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_options, dbi_scores, marker="o")
    plt.title("Davies-Bouldin Index by Number of Clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("DBI Score")
    plt.show()

    # # som.get_clusters(X_train)
    # plt.scatter(x=[c[1] for c in clusters], y=[c[0] for c in clusters], c=y, alpha=1)
    # plt.gca().invert_yaxis()
    # plt.show()
