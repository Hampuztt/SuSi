import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import susi
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix


def getNeuronClasses(
    som: susi.SOMClustering, X_train: np.ndarray, y_train: np.ndarray
) -> List[List[Optional[int]]]:

    n_cols, n_rows = som.n_columns, som.n_rows
    neuron_votes: List[List[Dict[int, int]]] = [
        [{} for _ in range(n_cols)] for _ in range(n_rows)
    ]
    # Iterate through each instance in the training set
    for bmu, y in zip(som.get_bmus(X_train), y_train):
        # Find the BMU for this instance
        # bmu is a tuple (bmu_x, bmu_y) representing the position of the BMU in the grid
        if y not in neuron_votes[bmu[0]][bmu[1]]:
            neuron_votes[bmu[0]][bmu[1]][y] = 0
        # Increment the vote for this class for the neuron
        neuron_votes[bmu[0]][bmu[1]][y] += 1

    neuron_classes: List[List[Optional[int]]] = [
        [None for _ in range(n_cols)] for _ in range(n_rows)
    ]

    # Decide the class for each neuron by majority voting
    for i in range(n_rows):
        for j in range(n_cols):
            if neuron_votes[i][j]:
                # Find the class with the maximum votes for this neuron
                neuron_classes[i][j] = max(
                    neuron_votes[i][j], key=neuron_votes[i][j].get
                )
    return neuron_classes


def createReportAndConfussionMatrix(
    som: Union[susi.SOMClassifier, susi.SOMClustering],
    X_test,
    y_test,
    data,
    title: str,
    filename: str,
    y_pred=None,
):
    if y_pred is None:
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


def filter_and_predict_test_samples(
    som, X_test, y_test, labeled_neurons
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters test samples based on assigned BMU classes and predicts their classes.

    Parameters:
    - som: The trained SOM model used to find BMUs for the test samples.
    - X_test: The test set features.
    - y_test: The actual labels for the test set.
    - labeled_neurons: A structure containing class labels for each neuron in the SOM.

    Returns:
    - new_x_test: The filtered test set features that have an assigned class.
    - new_y_test: The actual labels corresponding to the filtered test set features.
    - y_pred: The predicted classes for the filtered test set features.
    """
    bmus = som.get_bmus(X_test)
    y_pred = np.array([])
    new_x_test, new_y_test = np.empty((0, X_test.shape[1])), np.array([])
    for i, neuron_pos in enumerate(bmus):
        neuron_class = labeled_neurons[neuron_pos[0]][neuron_pos[1]]
        if neuron_class is None:
            continue
        new_x_test = np.vstack([new_x_test, X_test[i]])
        new_y_test = np.append(new_y_test, y_test[i]).astype(int)
        y_pred = np.append(y_pred, neuron_class).astype(int)
    assert len(new_x_test) == len(
        new_y_test
    ), "Mismatch in filtered test samples and labels length."
    return new_x_test, new_y_test, y_pred


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
        random_state=55,
    )
    supervised_som.fit(X_train, y_train)

    majority_som = susi.SOMClustering(
        n_rows=n_rows, n_columns=n_cols, n_iter_unsupervised=iterations, random_state=55
    )
    majority_som.fit(X_train)

    labeled_neurons = getNeuronClasses(majority_som, X_train, y_train)
    filtered_x_test, filtered_y_test, majority_pred = filter_and_predict_test_samples(
        majority_som, X_test, y_test, labeled_neurons
    )

    createReportAndConfussionMatrix(
        supervised_som,
        filtered_x_test,
        filtered_y_test,
        iris,
        f"supervised_som_{iterations}_iter",
        f"supervised_{iterations}_iter_{n_cols}x{n_rows}",
    )

    createReportAndConfussionMatrix(
        majority_som,
        filtered_x_test,
        filtered_y_test,
        iris,
        f"unsupervised_som_{iterations}_iter_majority_voting",
        f"unsupervised_{iterations}_iter_{n_cols}x{n_rows}",
        majority_pred,
    )


if __name__ == "__main__":
    map_sizes = [(5, 10)]
    iterations = [10000]
    # map_sizes = [(10, 5)]
    # iterations = [1000, 5000, 10000]
    for n_cols, n_rows in map_sizes:
        for iter in iterations:
            runSom(n_rows, n_cols, iter)
