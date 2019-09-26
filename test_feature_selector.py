import unittest
from feature_selector import FeatureSelector
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import numpy as np


class TestFeatureSelector(unittest.TestCase):
    def setUp(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.model = RandomForestClassifier()
        self.n_iter = 100
        self.threshold = 0.7
        self.min_features = 4
        self.X_noise = np.concatenate(
            (np.random.rand(150, 1), self.X, np.random.rand(150, 1)), axis=1
        )

        def acc(y, preds):
            return np.mean(y == preds)
        self.metric = acc

    def test_drops_noisy_pollute_random_k(self):
        """Should drop noisy features"""
        selector = FeatureSelector(
            self.model,
            n_iter=self.n_iter,
            pollute_type="random_k",
            drop_features=True,
            performance_threshold=self.threshold,
            performance_function=self.metric,
            min_features=self.min_features,
        )
        X_dropped = selector.fit_transform(self.X_noise, self.y)
        self.assertEqual(self.X.shape, X_dropped.shape)
        self.assertNotIn(0, selector.retained_features_)
        self.assertNotIn(5, selector.retained_features_)

    def test_drops_noisy_pollute_all(self):
        """Should drop noisy features"""
        selector = FeatureSelector(
            self.model,
            n_iter=self.n_iter,
            pollute_type="all",
            drop_features=True,
            performance_threshold=self.threshold,
            performance_function=self.metric,
            min_features=self.min_features,
        )
        X_dropped = selector.fit_transform(self.X_noise, self.y)
        self.assertEqual(self.X.shape, X_dropped.shape)
        self.assertNotIn(0, selector.retained_features_)
        self.assertNotIn(5, selector.retained_features_)

    def test_relevant_on_iris_without_drops(self):
        """Should find at least 3 relevant features on Iris"""
        selector = FeatureSelector(
            self.model,
            n_iter=self.n_iter,
            pollute_type="random_k",
            drop_features=False,
            performance_threshold=self.threshold,
            performance_function=self.metric,
        )
        selector.fit(self.X_noise, self.y)
        important_features = np.sum(selector.feature_importances_ > 0.7)
        self.assertTrue(important_features >= 3)

    def test_relevant_on_iris_with_drops(self):
        """Should find at least 3 relevant features on Iris"""
        selector = FeatureSelector(
            self.model,
            n_iter=self.n_iter,
            pollute_type="random_k",
            drop_features=True,
            performance_threshold=self.threshold,
            performance_function=self.metric,
            min_features=self.min_features,
        )
        selector.fit(self.X_noise, self.y)
        important_features = np.sum(selector.feature_importances_ > 0.7)
        self.assertTrue(important_features >= 3)

    def test_sklearn_interface(self):
        selector = FeatureSelector(
            self.model,
            n_iter=self.n_iter,
            pollute_type="random_k",
            drop_features=True,
            performance_threshold=self.threshold,
            performance_function=self.metric,
            min_features=self.min_features,
        )
        self.assertTrue(hasattr(selector, "fit"))
        self.assertTrue(hasattr(selector, "transform"))
        self.assertTrue(hasattr(selector, "fit_transform"))


if __name__ == '__main__':
    unittest.main()
