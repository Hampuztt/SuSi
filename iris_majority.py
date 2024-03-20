import numpy as np
from typing import List, Dict, Tuple, Union, Any
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  # type: ignore

# import pandas as pd
import random
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore
from sklearn import datasets  # type: ignore
from sklearn.utils import Bunch  # type: ignore
from scipy.stats import mode  # type: ignore
import susi  # type: ignore
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix  # type: ignore
from enum import Enum, auto


class RejectApproaches(Enum):
    IGNORE = (auto(),)
    RANDOM = (auto(),)
    CLOSEST_NEIGHBOUR = auto()


def load_wheat_data() -> Bunch:
    """
    Reads and processes data from a given file.

    Generates target names based on the labels in the dataset,
    where labels 1, 2, and 3 correspond to 'Kama', 'Rosa', and 'Canadian'.

    Returns a Bunch object similar to the structure of sklearn's dataset loaders.
    """

    data = []
    SEED_DATA_PATH = "datasets/seeds.txt"
    with open(SEED_DATA_PATH, "r") as file:
        for line in file:
            # Remove 'empty' elements caused by double tabs
            cleaned_line = [num for num in line.strip().split("\t") if num]
            # Convert to float and append to data list
            data.append([float(num) for num in cleaned_line])

    # Convert to numpy array for processing
    data_array = np.array(data)

    # Separate labels
    labels = data_array[:, -1].astype(int)  # Assuming the labels are in the last column

    # Data without the labels
    data_without_labels = data_array[:, :-1]
    target_names = ["Class 1", "Class 2", "Class 3"]
    feature_names = [
        "area",
        "perimeter",
        "compactness",
        "length of kernel",
        "width of kernel",
        "asymmetry coefficient",
        "length of kernel groove",
    ]

    # Mapping labels 1, 2, and 3 to 'Kama', 'Rosa', and 'Canadian'
    label_to_name = {1: "Kama", 2: "Rosa", 3: "Canadian"}
    target_names = [label_to_name[label] for label in sorted(label_to_name)]

    # Create a Bunch object to return
    seeds_data = Bunch(
        data=data_without_labels,
        target=labels,
        feature_names=feature_names,
        target_names=target_names,
        DESCR="Seeds Dataset: Features include area, perimeter, etc., with labels for Kama, Rosa, and Canadian.",
    )

    return seeds_data


def getNeuronClasses(
    som: susi.SOMClustering, X_train: np.ndarray, y_train: np.ndarray
) -> np.ndarray:

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

    neuron_classes = np.empty((n_rows, n_cols), dtype=object)

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
    y_test,
    data,
    title: str,
    filename: str,
    y_pred,
):
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
    print(f"Classification Report '{title}':")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    plt.show()


def find_nearest_class_by_distance(
    labeled_neurons: np.ndarray, bmu_position: np.ndarray
) -> int:
    """

    Parameters:
    labeled_neurons (np.ndarray): A 2D array of labeled neurons, where each element
                                  is the class of the neuron or None if not classified.
    bmu_position (np.ndarray): The (x, y) position of the neuron in question.

    Returns:
    int: The imputed class for the neuron based on its neighbors.
    """

    def get_neighbors_at_distance(x: int, y: int, distance: int) -> List[Any]:
        neighbors = []
        for i in range(
            max(0, x - distance), min(x + distance + 1, labeled_neurons.shape[0])
        ):
            for j in range(
                max(0, y - distance), min(y + distance + 1, labeled_neurons.shape[1])
            ):
                if abs(x - i) == distance or abs(y - j) == distance:
                    neighbor_value = labeled_neurons[i, j]
                    if neighbor_value is not None:  # Exclude None values
                        neighbors.append(neighbor_value)
        return neighbors

    for distance in range(1, max(labeled_neurons.shape[0], labeled_neurons.shape[1])):
        neighbors = get_neighbors_at_distance(
            bmu_position[0], bmu_position[1], distance
        )
        if neighbors:
            best_prediction = mode(neighbors)
            best_prediction = np.atleast_1d(best_prediction)
            if best_prediction.size:
                # Uncomment to modify the neuron_clases
                # prediction_grid[x, y] = best_prediction[0]
                return best_prediction[0]
    assert False, "No predictions made at all"


def filter_and_predict_test_samples(
    som, X_test, y_test, labeled_neurons, reject_approach: RejectApproaches
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
    amountOfPredictedClasses = len(np.unique(bmus))

    for i, neuron_pos in enumerate(bmus):
        neuron_class = labeled_neurons[neuron_pos[0]][neuron_pos[1]]
        # If the bmu is a neuron without an assigned class
        if neuron_class is None:
            if reject_approach is RejectApproaches.IGNORE:
                continue
            elif reject_approach is RejectApproaches.RANDOM:
                neuron_class = random.randint(0, amountOfPredictedClasses)
            elif reject_approach is RejectApproaches.CLOSEST_NEIGHBOUR:
                neuron_class = find_nearest_class_by_distance(
                    labeled_neurons, neuron_pos
                )

        # Since x are are arrays we use vstack, y is just an integer
        new_x_test = np.vstack([new_x_test, X_test[i]])
        new_y_test = np.append(new_y_test, y_test[i]).astype(int)
        y_pred = np.append(y_pred, neuron_class).astype(int)
    assert len(new_x_test) == len(
        new_y_test
    ), "Mismatch in filtered test samples and labels length."
    return new_x_test, new_y_test, y_pred


def compareSoms(
    n_rows: int,
    n_cols: int,
    iterations: int,
    dataset: Bunch,
    reject_approach=RejectApproaches.IGNORE,
    random_state=10,
):

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.5, random_state=random_state
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape, X_test.shape)
    supervised_som = susi.SOMClassifier(
        n_rows=n_rows,
        n_columns=n_cols,
        n_iter_unsupervised=iterations,
        n_iter_supervised=iterations,
        random_state=random_state,
    )
    supervised_som.fit(X_train, y_train)
    supervised_y_pred = supervised_som.predict(X_test)

    # Start training unsupervised som
    majority_som = susi.SOMClustering(
        n_rows=n_rows, n_columns=n_cols, n_iter_unsupervised=iterations, random_state=55
    )
    majority_som.fit(X_train)

    labeled_neurons = getNeuronClasses(majority_som, X_train, y_train)  # type: ignore
    # If reject_approach is 'ignore', discard x_test's and y_test's without a matching bmu;
    # otherwise, keep the original test cases stay unchanged.
    filtered_x_test, filtered_y_test, majority_y_pred = filter_and_predict_test_samples(
        majority_som, X_test, y_test, labeled_neurons, reject_approach
    )

    createReportAndConfussionMatrix(
        majority_som,
        filtered_y_test,
        dataset,
        f"unsupervised_som_{iterations}_iter_majority_voting",
        f"unsupervised_{iterations}_iter_{n_cols}x{n_rows}",
        majority_y_pred,
    )

    createReportAndConfussionMatrix(
        supervised_som,
        filtered_y_test,
        dataset,
        f"supervised_som_{iterations}_iter",
        f"supervised_{iterations}_iter_{n_cols}x{n_rows}",
        supervised_y_pred,
    )


def compareAccuracies(
    n_rows: int,
    n_cols: int,
    train_iterations: int,
    dataset: Bunch,
    comparisons: int,
    reject_approach=RejectApproaches.IGNORE,
    random_state=10,
):
    scaler = MinMaxScaler()
    majority_voting_accuracies = []
    supervised_som_accuracies = []
    count = 0
    for i in range(comparisons):
        print(count)
        count += 1
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, test_size=0.5, random_state=random_state + i
        )
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        supervised_som = susi.SOMClassifier(
            n_rows=n_rows,
            n_columns=n_cols,
            n_iter_unsupervised=train_iterations,
            n_iter_supervised=train_iterations,
            random_state=random_state + i,
        )
        supervised_som.fit(X_train, y_train)
        supervised_y_pred = supervised_som.predict(X_test)

        # Start training unsupervised som
        majority_som = susi.SOMClustering(
            n_rows=n_rows,
            n_columns=n_cols,
            n_iter_unsupervised=train_iterations,
            random_state=random_state + i,
        )
        majority_som.fit(X_train)

        labeled_neurons = getNeuronClasses(majority_som, X_train, y_train)  # type: ignore
        # If reject_approach is 'ignore', discard x_test's and y_test's without a matching bmu;
        # otherwise, keep the original test cases stay unchanged.
        filtered_x_test, filtered_y_test, majority_y_pred = (
            filter_and_predict_test_samples(
                majority_som, X_test, y_test, labeled_neurons, reject_approach
            )
        )

        majority_voting_accuracies.append(accuracy_score(y_test, majority_y_pred))
        supervised_som_accuracies.append(accuracy_score(y_test, supervised_y_pred))

    draw_accuracies(majority_voting_accuracies, supervised_som_accuracies)


def draw_accuracies(
    majority_voting_accuracies: list[int], supervised_som_accuracies: list[int]
):
    runs = range(1, len(supervised_som_accuracies) + 1)

    avg_supervised_accuracy = sum(supervised_som_accuracies) / len(
        supervised_som_accuracies
    )

    # First, plot the supervised SOM accuracies
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # First subplot in a 1x2 grid
    plt.plot(
        runs,
        supervised_som_accuracies,
        label="Supervised SOM",
        color="blue",
        marker="o",
    )
    plt.axhline(
        y=avg_supervised_accuracy,
        color="green",
        linewidth=2,
        linestyle=":",
        label="Avg Supervised SOM Accuracy",
    )
    # Annotate the average Supervised SOM accuracy line
    plt.annotate(
        f"Average: {avg_supervised_accuracy:.2f}%",
        xy=(1, avg_supervised_accuracy),
        xycoords=("axes fraction", "data"),
        xytext=(0, 10),
        textcoords="offset points",
        horizontalalignment="right",
        color="green",
    )
    plt.xlabel("Run")
    plt.ylabel("Accuracy")
    plt.title("Supervised SOM Accuracies")
    plt.legend()

    # Then, plot the difference in percentage between the two accuracies
    # Calculate the difference in percentage
    accuracy_differences = [
        (sv - mv) / sv * 100
        for sv, mv in zip(supervised_som_accuracies, majority_voting_accuracies)
    ]

    # Calculate the average accuracy difference
    avg_accuracy_difference = sum(accuracy_differences) / len(accuracy_differences)
    plt.subplot(1, 2, 2)  # Second subplot in a 1x2 grid

    plt.plot(
        runs, accuracy_differences, label="Accuracy Difference", color="red", marker="o"
    )
    plt.axhline(
        y=avg_accuracy_difference,
        color="purple",
        linewidth=2,
        linestyle=":",
        label="Avg Accuracy Difference",
    )
    # Annotate the average accuracy difference line
    plt.annotate(
        f"Average: {avg_accuracy_difference:.2f}%",
        xy=(1, avg_accuracy_difference),
        xycoords=("axes fraction", "data"),
        xytext=(0, -15),
        textcoords="offset points",
        horizontalalignment="right",
        color="purple",
    )
    plt.axhline(
        0, color="black", linestyle="--"
    )  # Adds a horizontal line at 0% difference
    plt.xlabel("Run")
    plt.ylabel("Supervised - Majority Voting Accuracy Difference (%)")
    plt.title("Supervised SOM vs. Majority Voting: Accuracy Difference (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()


"""
5x10 
10x10
10000 ish iterations with wheat
After iris and wheat
Rejection approach of dealing with neurons, (go random), try neighbours, 
"""
if __name__ == "__main__":
    iris_data = datasets.load_iris()
    wheat_data = load_wheat_data()

    datasets = [wheat_data]
    map_sizes = [(5, 10)]
    iterations = [10000]
    # map_sizes = [(10, 5)]
    # iterations = [1000, 5000, 10000]
    for data in datasets:
        for n_cols, n_rows in map_sizes:
            for iter in iterations:
                # compareSoms(
                #     n_rows, n_cols, iter, data, RejectApproaches.CLOSEST_NEIGHBOUR
                # )
                compareAccuracies(
                    n_rows, n_rows, iter, data, 100, RejectApproaches.CLOSEST_NEIGHBOUR
                )
