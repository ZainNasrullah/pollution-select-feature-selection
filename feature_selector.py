from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
        pollute_type="random_k",
        pollute_k=1,
        additional_pollution=True,
        min_features=None,
        drop_every_n_iters=5,
        n_splits=3,
        n_iter=100,
        feature_names=None,
        drop_features=False,
        subsample_ratio=0.5,
    ):
        self.drop_features = drop_features
        self.additional_pollution = additional_pollution
        self.pollute_k = pollute_k
        self.pollute_type = pollute_type
        self.drop_every_n_iters = drop_every_n_iters
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.model = model
        self.metric = performance_metric
        self.threshold = performance_threshold
        self.min_features = min_features
        self._additional_pollute_k = 2
        self.subsample_ratio = subsample_ratio

        if feature_names:
            assert type(feature_names) == list, "Feature names is not a list."
            self._name_mapping = {idx: name for idx, name in enumerate(feature_names)}

        if pollute_type:
            assert type(pollute_k) == int, "pollute k must be an integer"

        self._test_size = 0.3

        # check_estimator(self.model)

    def fit(self, X, y):

        self._init_fit_params(X)
        n_features = X.shape[1]
        mask_array = np.zeros(n_features)
        assert self.pollute_k < n_features, "Pollute k must be less than # of features"

        for iter_idx in range(self.n_iter):

            _, X_sample, _, y_sample = train_test_split(
                X, y, test_size=self.subsample_ratio, stratify=y, shuffle=True
            )

            X_sample = X_sample[:, self.retained_features_]
            n_features = X_sample.shape[1]

            X_pollute = self._pollute_data(X_sample)
            X_train, X_test, y_train, y_test = train_test_split(
                X_pollute, y_sample, test_size=self._test_size, stratify=y_sample
            )

            train_score, test_score = self._fit_predict_score(
                X_test, X_train, y_test, y_train
            )

            if test_score >= self.threshold:
                mask = self._compare_to_pollution_mask(n_features)
                mask_array[self.retained_features_] += mask
                self.iters_ += 1

                self.feature_importances_[self.retained_features_] = (
                    mask_array[self.retained_features_] / self.iters_
                )

            else:
                print(f"Did not meet test threshold: {self.metric}>{self.threshold}")
                self.failures_ += 1

            self._record_iter(test_score, train_score)
            if self.drop_features:
                self._eval_dropping_conditions(iter_idx)

    def transform(self, X):
        assert (
            self.retained_features_
        ), "Selector has not been fit or retained feature list is empty."
        return X[:, self.retained_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _record_iter(self, test_score, train_score):
        self.train_scores_.append(train_score)
        self.test_scores_.append(test_score)
        self._features_at_iter.append(self.retained_features_.copy())
        self._importances_at_iter.append(np.copy(self.feature_importances_))

    def _init_fit_params(self, X_sample):
        self.X_orig = X_sample
        n_features = X_sample.shape[1]
        self.retained_features_ = list(np.arange(n_features))
        self.feature_importances_ = np.zeros(n_features)
        self.dropped_features_ = list()
        self.train_scores_ = []
        self.test_scores_ = []
        self.failures_ = 0
        self.iters_ = 0
        self._features_at_iter = list()
        self._importances_at_iter = list()

    def _eval_dropping_conditions(self, iter_idx):
        dropping_condition = (
            iter_idx >= 5
            and iter_idx % self.drop_every_n_iters == 0
            and self.min_features
            and len(self.retained_features_) > self.min_features
        )

        if dropping_condition:
            min_feature = np.argmin(self.feature_importances_[self.retained_features_])
            if hasattr(self, "_name_mapping"):
                print("Dropping: ", self._name_mapping[min_feature])
            else:
                print(
                    "Dropping feature with index:", self.retained_features_[min_feature]
                )
            self.dropped_features_.append(self.retained_features_.pop(min_feature))

    def _compare_to_pollution_mask(self, n_features):

        pollute_shape = (
            self.pollute_k + 2 if self.additional_pollution else self.pollute_k
        )
        pollute_idx = np.arange(n_features, n_features + pollute_shape)
        importances = self.model.feature_importances_

        mask = np.zeros(n_features)
        for i in range(n_features):
            mask[i] = np.all(importances[i] > importances[pollute_idx])

        return mask

    def _fit_predict_score(self, X_test, X_train, y_test, y_train):
        self.model.fit(X_train, y_train)
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        train_score = self.metric(y_train, train_preds)
        test_score = self.metric(y_test, test_preds)
        return train_score, test_score

    def _pollute_data(self, X):

        n_samples, n_features = X.shape

        if self.pollute_type == "all":
            pollute_shape = X.shape[1]
        elif self.pollute_type == "random_k":
            pollute_shape = self.pollute_k
        else:
            raise Exception("pollute type not in ('all', 'random_k')")

        pollute_shape += self._additional_pollute_k
        X_pollute = np.zeros(shape=(n_samples, pollute_shape))
        random_feats = np.random.choice(
            n_features, pollute_shape - self._additional_pollute_k, replace=None
        )

        for i, feat_idx in enumerate(random_feats):
            X_pollute[:, i] = np.random.permutation(X[:, feat_idx])
        X_pollute[:, i + 1] = np.random.randint(0, 2, n_samples)
        X_pollute[:, i + 2] = np.random.rand(n_samples)

        return np.concatenate((X, X_pollute), axis=1)

    def plot_test_scores_by_iters(self):
        assert hasattr(self, "feature_importances_"), "Model has not been fit yet."
        plt.plot(range(len(self.test_scores_)), self.test_scores_, marker="o")
        plt.title("Test scores (average across k-folds) by iterations.")
        plt.show()

    def plot_test_scores_by_features(self):
        assert hasattr(self, "feature_importances_"), "Model has not been fit yet."
        feature_lens = [len(l) for l in self._features_at_iter]
        scores_ = self.test_scores_
        test_df = pd.DataFrame({"n_features": feature_lens, "test_score": scores_})
        test_df_agg = test_df.groupby("n_features").mean()
        plt.plot(test_df_agg.index, test_df_agg.test_score, marker="o")
        plt.title("Test scores (average across k-folds) by number of features.")
        plt.show()


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_noise = np.concatenate(
        (np.random.rand(150, 1), X, np.random.rand(150, 1)), axis=1
    )

    def acc(y, preds):
        return np.mean(y == preds)

    model = RandomForestClassifier()
    selector = FeatureSelector(
        model,
        n_iter=100,
        performance_threshold=0.7,
        performance_metric=acc,
        min_features=4,
    )

    X_dropped = selector.fit_transform(X_noise, y)
    print(selector.feature_importances_)

    selector.plot_test_scores_by_iters()
    selector.plot_test_scores_by_features()
