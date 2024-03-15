import numpy as np
import unittest
from typing import List, Any
from scipy.stats import mode
from iris_majority import impute_prediction_with_distance


class TestImputePredictionWithDistance(unittest.TestCase):
    def test_imputation_with_direct_neighbors(self):
        grid = np.array([[None, 1, 3], [1, None, 1], [None, None, 2]], dtype=object)
        result = impute_prediction_with_distance(grid, np.array([1, 1]))
        self.assertEqual(
            result, 1, "Should impute with the most common direct neighbor"
        )

    def test_imputation_with_no_direct_neighbors(self):
        grid = np.array([[1, None, 3], [None, None, None], [2, None, 2]], dtype=object)
        result = impute_prediction_with_distance(grid, np.array([1, 1]))
        self.assertEqual(
            result,
            2,
        )

    def test_imputation_with_further_neighbors(self):
        grid = np.array([[None, None, 3], [None, None, 3], [2, 5, 4]], dtype=object)
        result = impute_prediction_with_distance(grid, np.array([0, 0]))
        self.assertEqual(
            result,
            3,
            "Should impute based on further neighbors if direct ones are undefined",
        )

    def test_assertion_without_any_neighbors(self):
        grid = np.array([[None]], dtype=object)
        with self.assertRaises(AssertionError):
            impute_prediction_with_distance(grid, np.array([0, 0]))


if __name__ == "__main__":
    unittest.main()
