import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import susi
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix


def getNeuronClasses(som: susi.SOMClustering, X_train, y_train):

    neuron_votes: List[List[Dict[int, int]]] = [
        [{} for _ in range(n_cols)] for _ in range(n_rows)
    ]
    # Iterate through each instance in the training set
    for bmu, y in zip(som.get_bmus(X_train), y_train):
        # Find the BMU for this instance
        # bmu is a tuple (bmu_x, bmu_y) representing the position of the BMU in the grid

        # If the class label y is not yet in the dictionary for this neuron, initialize it
        if y not in neuron_votes[bmu[0]][bmu[1]]:
            neuron_votes[bmu[0]][bmu[1]][y] = 0

        # Increment the vote for this class for the neuron
        neuron_votes[bmu[0]][bmu[1]][y] += 1

    # Decide the class for each neuron by majority voting
    neuron_classes: List[List[Optional[int]]] = [
        [None for _ in range(n_cols)] for _ in range(n_rows)
    ]

    for i in range(n_rows):
        for j in range(n_cols):
            if neuron_votes[i][j]:
                # Find the class with the maximum votes for this neuron
                neuron_classes[i][j] = max(
                    neuron_votes[i][j], key=neuron_votes[i][j].get
                )
    return neuron_classes


def createReportAndConfussionMatrix(
    som, X_test, y_test, data, title: str, filename: str
):
    y_pred = som.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    # report = classification_report(
    #     y_test, y_pred, target_names=data.target_names, output_dict=True
    # )
    accuracy = accuracy_score(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=data.target_names,
        yticklabels=data.target_names,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(
        f"{title}\nAccuracy: {accuracy:.2f}, \nSize: {som.n_columns}x{som.n_rows}"
    )
    plt.savefig("Images/" + filename, bbox_inches="tight")
    # Display metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    plt.show()


def runSom(n_rows: int, n_cols: int, iterations: int):
    # n_features = 5
    iris = datasets.load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.5, random_state=55
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    supervised_som = susi.SOMClassifier(
        n_rows=n_rows,
        n_columns=n_cols,
        n_iter_unsupervised=iterations,
        n_iter_supervised=iterations,
        init_mode_supervised="majority",
    )

    unsupervised_som = susi.SOMClassifier(
        n_rows=n_rows,
        n_columns=n_cols,
        n_iter_unsupervised=iterations,
        n_iter_supervised=0,
        init_mode_supervised="majority",
    )

    # meow_som = susi.SOMClustering(n_rows, n_cols, n_iter_unsupervised=iterations)
    # meow_som.fit(X_train)
    # meow = getNeuronClasses(meow_som, X_train, y_train)
    # bmus = meow_som.get_bmus(X_test)
    #
    # accuracy = accuracy_score(y_test, bmus)
    # print(accuracy)
    # exit()

    unsupervised_som.fit(X_train, y_train)
    supervised_som.fit(X_train, y_train)
    # unsupervised_classes = getNeuronClasses(unsupervised_som, X_train, y_train)
    # supervised_classes = getNeuronClasses(supervised_som, X_train, y_train)

    createReportAndConfussionMatrix(
        supervised_som,
        X_test,
        y_test,
        iris,
        f"supervised_som_{iterations}_iter",
        f"supervised_{iterations}_iter_{n_cols}x{n_rows}",
    )

    createReportAndConfussionMatrix(
        unsupervised_som,
        X_test,
        y_test,
        iris,
        f"unsupervised_som_{iterations}_iter",
        f"unsupervised_{iterations}_iter_{n_cols}x{n_rows}",
    )


if __name__ == "__main__":
    # map_sizes = [(5, 5), (5, 10)]
    # iterations = [1000, 10000]
    map_sizes = [(5, 10)]
    iterations = [10000]
    for n_cols, n_rows in map_sizes:
        for iter in iterations:
            print("wat")
            runSom(n_rows, n_cols, iter)
# TODO
# labels = iris.target
# data = pd.DataFrame(iris.data[::4])
# data.columns = iris.feature_names


# get the data
