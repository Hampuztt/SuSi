import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Any
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  # type: ignore
from scipy.io import loadmat
from pprint import pprint

# import pandas as pd import random import seaborn as sns  # type: ignore
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


@dataclass
class ParametersFromPaper:
    init = "random"
    learning_start = 0.7
    learning_end = 0.07
    grid_x = 80
    grid_y = 80
    supervised_iteraetions = 60000
    unsupervised_iterations = 60000
    init_method = "random"
    test_split = 0.7
    # Rest deafult parameters


def load_hyperspectral_data() -> Bunch:
    SPECTRAL_IMAGE_PATH = "datasets/AVIRIS_SalinasValley/Salinas.mat"
    SPECTRAL_GT_PATH = "datasets/AVIRIS_SalinasValley/Salinas_gt.mat"
    data_dict = loadmat(SPECTRAL_IMAGE_PATH)
    gt_dict = loadmat(SPECTRAL_GT_PATH)
    label_to_name = {
        1: "Brocoli_green_weeds_1",
        2: "Brocoli_green_weeds_2",
        3: "Fallow",
        4: "Fallow_rough_plow",
        5: "Fallow_smooth",
        6: "Stubble",
        7: "Celery",
        8: "Grapes_untrained",
        9: "Soil_vinyard_develop",
        10: "Corn_senesced_green_weeds",
        11: "Lettuce_romaine_4wk",
        12: "Lettuce_romaine_5wk",
        13: "Lettuce_romaine_6wk",
        14: "Lettuce_romaine_7wk",
        15: "Vinyard_untrained",
        16: "Vinyard_vertical_trellis",
    }

    target_names = [label_to_name[label] for label in sorted(label_to_name)]
    data = data_dict[list(data_dict.keys())[-1]]  # Assuming the last key holds the data
    gt = gt_dict[list(gt_dict.keys())[-1]]  # Assuming the last key holds the data

    # Reshape the data
    nrows, ncols, nbands = data.shape
    data_reshaped = data.reshape((nrows * ncols, nbands))
    gt_reshaped = gt.flatten()

    return Bunch(
        data=data_reshaped,
        target=gt_reshaped,
        feature_names=[f"Band {i+1}" for i in range(nbands)],
        target_names=target_names,
        DESCR="Hyperspectral Image Dataset",
    )


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
        filename="wheat.cvs",
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
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # report = classification_report(
    #     y_test, y_pred, target_names=data.target_names, output_dict=True
    # )
    accuracy = accuracy_score(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
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
    # print(y_test, y_pr)
    # print(classification_report(y_test, y_pred, target_names=data.target_names))
    plt.show()


def find_nearest_class_by_distance_optimized(
    labeled_neurons: np.ndarray, bmu_position: np.ndarray
) -> int:
    max_distance = max(labeled_neurons.shape)

    # Create a masked array where unclassified neurons are masked out
    masked_labeled_neurons = np.ma.masked_array(
        labeled_neurons, mask=labeled_neurons == None
    )

    for distance in range(1, max_distance):
        min_row = max(0, bmu_position[0] - distance)
        max_row = min(bmu_position[0] + distance + 1, labeled_neurons.shape[0])
        min_col = max(0, bmu_position[1] - distance)
        max_col = min(bmu_position[1] + distance + 1, labeled_neurons.shape[1])

        # Extracting edge elements at a given "Manhattan distance" from bmu_position
        top_edge = masked_labeled_neurons[min_row:max_row, min_col].compressed()
        bottom_edge = masked_labeled_neurons[min_row:max_row, max_col - 1].compressed()
        left_edge = masked_labeled_neurons[
            min_row, min_col + 1 : max_col - 1
        ].compressed()
        right_edge = masked_labeled_neurons[
            max_row - 1, min_col + 1 : max_col - 1
        ].compressed()

        neighbors = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])

        if neighbors.size > 0:
            best_prediction = mode(neighbors, nan_policy="omit").mode
            if best_prediction.size:
                return best_prediction[0]

    assert False, "No predictions made at all"


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
    # y_pred = np.array([])
    # new_x_test, new_y_test = np.empty((0, X_test.shape[1])), np.array([])
    new_x_test_list, new_y_test_list, y_pred_list = [], [], []

    if reject_approach is RejectApproaches.RANDOM:
        predicted_classes = [
            labeled_neurons[bmu[0]][bmu[1]]
            for bmu in bmus
            if labeled_neurons[bmu[0]][bmu[1]] is not None
        ]
        amountOfPredictedClasses = len(np.unique(predicted_classes))

    print(len(bmus))
    one_percent = len(bmus) // 100
    count = 0
    for i, neuron_pos in enumerate(bmus):
        # if i % one_percent == 0:
        #     count += 1
        # print(f"Finished {count} % of filter loop")
        neuron_class = labeled_neurons[neuron_pos[0]][neuron_pos[1]]
        # If the bmu is a neuron without an assigned class
        if neuron_class is None:
            if reject_approach is RejectApproaches.IGNORE:
                continue
            elif reject_approach is RejectApproaches.RANDOM:
                neuron_class = random.randint(1, amountOfPredictedClasses)
            elif reject_approach is RejectApproaches.CLOSEST_NEIGHBOUR:
                neuron_class = find_nearest_class_by_distance(
                    labeled_neurons, neuron_pos
                )

        new_x_test_list.append(X_test[i])
        new_y_test_list.append(int(y_test[i]))
        y_pred_list.append(int(neuron_class))
    new_x_test = np.array(new_x_test_list)
    new_y_test = np.array(new_y_test_list, dtype=int)
    y_pred = np.array(y_pred_list, dtype=int)
    assert len(new_x_test_list) == len(
        new_y_test_list
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
    parameters = ParametersFromPaper()
    LEARN_START = 0.7
    LEARN_END = 0.07

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=parameters.test_split,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape, X_test.shape)
    supervised_som = susi.SOMClassifier(
        learning_rate_start=parameters.learning_start,
        learning_rate_end=parameters.learning_end,
        n_rows=n_rows,
        n_columns=n_cols,
        n_iter_unsupervised=iterations,
        n_iter_supervised=iterations,
        random_state=random_state,
    )
    supervised_som.fit(X_train, y_train)
    print("supervised finish")
    supervised_y_pred = supervised_som.predict(X_train)
    print(supervised_som.score(X_test, y_test) * 100)

    # Start training unsupervised som
    majority_som = susi.SOMClustering(
        learning_rate_start=LEARN_START,
        learning_rate_end=LEARN_END,
        n_rows=n_rows,
        n_columns=n_cols,
        n_iter_unsupervised=iterations,
        random_state=55,
    )
    majority_som.fit(X_train)
    print("unsuper finish")

    labeled_neurons = getNeuronClasses(majority_som, X_train, y_train)  # type: ignore
    print("Labeled neurons found")
    # If reject_approach is 'ignore', discard x_test's and y_test's without a matching bmu;
    # otherwise, keep the original test cases stay unchanged.
    filtered_x_test, filtered_y_test, majority_y_pred = filter_and_predict_test_samples(
        majority_som, X_train, y_train, labeled_neurons, reject_approach
    )
    print("filter finish")

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

    draw_accuracies(
        majority_voting_accuracies, supervised_som_accuracies, dataset["filename"]
    )


def draw_accuracies(
    majority_voting_accuracies: list[int],
    supervised_som_accuracies: list[int],
    dataset_name="",
):
    if dataset_name:
        dataset_name = dataset_name.split(".")[0].upper()
    runs = range(1, len(supervised_som_accuracies) + 1)

    avg_supervised_accuracy = sum(supervised_som_accuracies) / len(
        supervised_som_accuracies
    )
    avg_majority_voting_accuracy = sum(majority_voting_accuracies) / len(
        majority_voting_accuracies
    )

    # First, plot the supervised SOM accuracies
    # Plot supervised and majority voting accuracies
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 1, 1)  # First subplot in a 1x2 grid
    plt.plot(
        runs,
        supervised_som_accuracies,
        label="Supervised SOM",
        color="blue",
        marker="o",
    )
    plt.plot(
        runs,
        majority_voting_accuracies,
        label="Majority Voting",
        color="orange",
        marker="x",
    )
    plt.axhline(
        y=avg_supervised_accuracy,
        color="green",
        linewidth=2,
        linestyle=":",
        label=f"Avg Supervised SOM Accuracy ({100*avg_supervised_accuracy:.2f}%)",
    )
    plt.axhline(
        y=avg_majority_voting_accuracy,
        color="red",
        linewidth=2,
        linestyle=":",
        label=f"Avg Majority Voting Accuracy ({100*avg_majority_voting_accuracy:.2f}%)",
    )

    plt.xlabel("Run")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracies: Supervised SOM and Majority Voting ({dataset_name})")
    plt.legend()

    plt.savefig("Images/" + "Accuracies_" + dataset_name, bbox_inches="tight")
    plt.show()

    # Then, plot the difference in percentage between the two accuracies
    # Calculate the difference in percentage
    # Then, plot the difference in accuracy
    accuracy_differences = [
        (sv - mv)
        for sv, mv in zip(supervised_som_accuracies, majority_voting_accuracies)
    ]
    avg_accuracy_difference = sum(accuracy_differences) / len(accuracy_differences)
    plt.subplot(1, 1, 1)  # Second subplot in a 1x2 grid
    plt.plot(
        runs, accuracy_differences, label="Accuracy Difference", color="red", marker="o"
    )
    plt.axhline(
        y=avg_accuracy_difference,
        color="purple",
        linewidth=2,
        linestyle=":",
        label=f"Avg Accuracy Difference ({100*avg_accuracy_difference:.2f}%)",
    )
    plt.axhline(
        0, color="black", linestyle="--"
    )  # Adds a horizontal line at 0% difference
    plt.xlabel("Run")
    plt.ylabel("Accuracy Difference (%)")
    plt.title(
        f"Accuracy Difference: Supervised SOM vs Majority Voting ({dataset_name})"
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig("Images/" + "Accuracies_compare" + dataset_name, bbox_inches="tight")
    plt.show()


def soildata():
    data = loadmat("datasets/AVIRIS_SalinasValley/Salinas_gt.mat")
    print(data.keys())

    # for key in data:
    #     print(key, data[key])
    # pprint(data["salinas_gt"])
    for i in data["salinas_gt"]:
        print(i)

    print(len(data["salinas_gt"]))
    print(len(data["salinas_gt"][0]))
    # pprint(data)


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
    spectral_data = load_hyperspectral_data()
    # print(f"meow: {spectral_data.target_names}")
    # print(f"meow: {spectral_data.feature_names}")
    # print(f"meow: {wheat_data.target_names}")
    # print(f"meow: {wheat_data.feature_names}")

    # soildata()
    # exit()
    datasets = [spectral_data]
    map_sizes = [(10, 10)]
    iterations = [1000]
    # map_sizes = [(10, 5)]
    # iterations = [1000, 5000, 10000]

    for data in datasets:
        for n_cols, n_rows in map_sizes:
            for iter in iterations:
                compareSoms(
                    n_rows, n_cols, iter, data, RejectApproaches.CLOSEST_NEIGHBOUR
                )
                # compareAccuracies(
                #     n_rows, n_rows, iter, data, 10, RejectApproaches.CLOSEST_NEIGHBOUR
                # )
