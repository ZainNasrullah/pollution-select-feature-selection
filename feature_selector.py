from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.utils.estimator_checks import check_estimator
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Any


class FeatureSelector:
    def __init__(
        self,
        model,
        performance_threshold,
        performance_metric,
        min_features=None,
        drop_every_n_iters=5,
        n_splits=3,
        n_iter=100,
    ):
        self.drop_every_n_iters = drop_every_n_iters
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.model = model
        self.metric = performance_metric
        self.threshold = performance_threshold
        self.min_features = min_features

        # check_estimator(self.model)

    def fit(self, X, y):

        n_samples, n_features = X.shape
        mask_array = np.zeros(n_features)
        cv = KFold(n_splits=self.n_splits, shuffle=True)

        self.X_orig = X
        self.retained_features_ = list(np.arange(n_features))
        self.dropped_features_ = list()
        self.train_scores_ = []
        self.test_scores_ = []
        self.failures_ = 0
        self.iters_ = 0
        self.feature_importances_ = np.zeros(n_features)

        for iter_idx in range(self.n_iter):
            train_cv_scores = []
            test_cv_scores = []
            X = self.X_orig[:, self.retained_features_]

            for train_idx, test_idx in cv.split(X, y):

                pre_n_features = X.shape[1]
                X_pollute = self._pollute_data(X)
                X_train, y_train = X_pollute[train_idx], y[train_idx]
                X_test, y_test = X_pollute[test_idx], y[test_idx]

                train_score, test_score = self._fit_predict_score(
                    X_test, X_train, y_test, y_train
                )

                if test_score >= self.threshold:
                    mask = self._compare_to_pollution_mask(X_pollute, pre_n_features)
                    mask_array[self.retained_features_] += mask
                    self.iters_ += 1
                else:
                    print(
                        f"Did not meet test threshold: {self.metric}>{self.threshold}"
                    )
                    self.failures_ += 1

                train_cv_scores.append(train_score)
                test_cv_scores.append(test_score)

            self.train_scores_.append(np.mean(train_cv_scores))
            self.test_scores_.append(np.mean(test_cv_scores))

            if (
                iter_idx >= 5
                and iter_idx % self.drop_every_n_iters == 0
                and self.min_features
                and len(self.retained_features_) > self.min_features
            ):
                min_feature = np.argmin(self.feature_importances_[self.retained_features_])
                print("Dropping feature with index: ", min_feature)
                self.dropped_features_.append(self.retained_features_.pop(min_feature))

            self.feature_importances_[self.retained_features_] = (
                    mask_array[self.retained_features_] / self.iters_
            )


    def _compare_to_pollution_mask(self, X_pollute, n_features):
        orig_idx = np.arange(0, n_features)
        pollute_idx = np.arange(n_features, X_pollute.shape[1])[:-2]
        importances = self.model.feature_importances_
        mask = (
            (importances[orig_idx] > importances[pollute_idx])
            & (importances[orig_idx] > importances[-1])
            & (importances[orig_idx] > importances[-2])
        )
        return mask

    def _fit_predict_score(self, X_test, X_train, y_test, y_train):
        self.model.fit(X_train, y_train)
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        train_score = self.metric(y_train, train_preds)
        test_score = self.metric(y_test, test_preds)
        return train_score, test_score

    def _pollute_data(self, X):

        n_samples = X.shape[0]
        pollute_shape = X.shape[1] + 2
        X_pollute = np.zeros(shape=(n_samples, pollute_shape))

        for i in range(X.shape[1]):
            X_pollute[:, i] = np.random.permutation(X[:, i])

        X_pollute[:, i + 1] = np.random.randint(0, 2, n_samples)
        X_pollute[:, i + 2] = np.random.rand(n_samples)

        return np.concatenate((X, X_pollute), axis=1)


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_noise = np.concatenate((np.random.rand(150,1),X, np.random.rand(150, 1)), axis=1)

    def acc(y, preds):
        return np.mean(y == preds)

    model = RandomForestClassifier()
    selector = FeatureSelector(
        model, performance_threshold=0.7, performance_metric=acc, min_features=4
    )

    selector.fit(X_noise, y)
    print(selector.feature_importances_)
