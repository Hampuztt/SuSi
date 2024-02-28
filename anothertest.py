import pandas as pd
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from pandas._config import display
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
import susi
from susi.SOMPlots import plot_umatrix, plot_estimation_map
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix
from sklearn.datasets import make_classification

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


def displayData(data, labels=None):
    df = pd.DataFrame(data)
    df["label"] = labels  # Add the labels as a new column in the DataFrame
    # Display the first few rows of the DataFrame to check the data
    print(df.shape)
    print(df.head())


def runMiniSom(X_train_scaled):
    som = MiniSom(8, 8, X_train_scaled.shape[1], random_seed=1)
    som.train_random(X_train_scaled, 5000)

    # To simulate `get_clusters`, we manually find the winning position for each sample
    winners = np.array([som.winner(x) for x in X_train_scaled])
    # Visualization: Scatter plot of the winning positions
    # Note: MiniSom coordinates (winning positions) are already in the desired format, no need to invert y-axis
    plt.scatter(winners[:, 1], winners[:, 0], alpha=0.2)
    plt.show()


def runSuSiSom(X_train_scaled):
    som = susi.SOMClustering(
        n_rows=8,
        n_columns=8,
        n_iter_unsupervised=5000,
    )
    som.fit(X_train_scaled)
    # Assuming `clusters` is your 2D array output from the SOM
    # Let's say you decide to further partition these clusters into k clusters
    clusters = som.get_clusters(X_train_scaled)
    for centroid in som.unsuper_som_:
        plt.scatter(
            centroid[:, 0],
            centroid[:, 1],
            marker="x",
            s=80,
            linewidths=35,
            color="k",
            label="centroid",
        )
    # plt.scatter(x=[c[1] for c in clusters], y=[c[0] for c in clusters], alpha=0.2)
    # plt.gca().invert_yaxis()
    plt.show()


def smallSuSi(X_train_scaled):
    som = susi.SOMClustering(
        n_rows=1,
        n_columns=3,
        n_iter_unsupervised=500,
    )
    som.fit(X_train_scaled)

    winner_coordinates = np.array([som.get_bmu(x, som.unsuper_som_) for x in data]).T

    cluster_index = np.ravel_multi_index(winner_coordinates, (1, 3))

    for c in np.unique(cluster_index):
        plt.scatter(
            data[cluster_index == c, 0],
            data[cluster_index == c, 1],
            label="cluster=" + str(c),
            alpha=0.7,
        )

    print(som.unsuper_som_)
    print(som.unsuper_som_.shape)
    exit()
    # plotting centroid
    for centroid in som.unsuper_som_:
        plt.scatter(
            centroid[:, 0],
            centroid[:, 1],
            marker="x",
            s=10,
            linewidths=35,
            color="k",
            label="centroid",
        )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data, labels = read_and_process_seeds_data(SEED_DATA_PATH)
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True
    )
    X_train_scaled = scaler.fit_transform(X_train)
    # displayData(data, labels)

    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
        names=[
            "area",
            "perimeter",
            "compactness",
            "length_kernel",
            "width_kernel",
            "asymmetry_coefficient",
            "length_kernel_groove",
            "target",
        ],
        usecols=[0, 5],
        sep="\t+",
        engine="python",
    )
    # data normalization
    print(data.shape)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    data = data.values
    print(data.shape)
    exit()
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True
    )
    X_train_scaled = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # X, y = data.values, labels
    # runMiniSom(X_train_scaled)
    smallSuSi(X_train_scaled)
    exit()

    # # som.get_clusters(X_train)
    # plt.scatter(x=[c[1] for c in clusters], y=[c[0] for c in clusters], c=y, alpha=1)
    # plt.gca().invert_yaxis()
    # plt.show()
