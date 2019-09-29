from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from numba import njit, prange
import seaborn as sns
import multiprocessing as mp


def acc(y, preds):
    return np.mean(y == preds)


class PollutionSelect:
    """ Pollution-Select Feature Selection Algorithm

    Parameters
    ----------
    model : base estimator with scikit-learn style API
        Must have fit and predict method implemented as well as a
        feature_importances_ attribute after fitting.

    performance_function : function(y_true, y_predictions)
        Function which measures performance. After fitting the model, it gets
        against this function during cross-validation.

    performance_threshold : float
        Target performance that must be achieved by the model to update feature
        importance values. If not met, the number of failures is incremented.

    n_iter : int, optional (default=100)
        Number of iterations to run the feature selection for. Larger values
        will result in more accurate estimates of importances.

    subsample_ratio : float, optional (default=0.5)
        Ratio to subsample the dataset at each iteration for computational reasons.
        Also introduces some randomness in the process.

    pollute_type : str, optional (default="random_k")
        Method used for generating the polluted features.

        - if `all`, creates shadow features from every feature as in boruta
        - if `random_k`, creates shadow features out of k random where where k is
          defined by `pollute_k`

    pollute_k : int, optional (default=1)
        Number of features to pollute when pollute_type == `random_k`

    additional_pollution : bool, optional (default=True)
        Whether to include additional random noise features

        - Random normal (mean 0, variance 1)
        - Randomly drawn samples from [0, 1] using a discrete uniform distribution

    drop_features : bool, optional (default=False)
        Whether to drop features. By default performs a check every
        `drop_every_n_iters` iterations and only drops the feature with consistently
        minimum importance until `min features` remain. Starts after 5 iterations.

    min_features : int or None, optional (default=None)
        Whether to include a lower bound on the number of features; only relevant
        when `drop_features` is True.

    drop_every_n_iters : int, optional (default=5)
        Drop a feature after every `drop_every_n_iters` iterations. If drop_features
        is false, this value is ignored / not used.

    use_predict_proba : bool, optional (default=True)
        Sets the prediction mode

        - if True, use the predict_proba() method instead
        - else, use the predict() method

    feature_names : list of strings, optional (default=None)
        Names for the features in the dataset. Primarily used to understand which
        features are being dropped.

    verbose : bool, optional (default=False)
        Whether to print out features that are being dropped.

    Attributes
    ----------
    feature_importances_ : np array of shape = [n_features]
        Importances of each feature after running fit

    retained_features_ : list
        Features retained after fitting; does not change from the original n_features
        if the object has drop_features=False.

    dropped_features_ : list
        Features dropped during fitting; empty if drop_features=False.

    train_scores_ :  array of shape = [self.n_iters]
        Scores on the training set during fitting

    test_scores_ :  array of shape = [self.n_iters]
        Scores on the test set during fitting

    failures_ : int
        Number of times the test score failed to meet self.threshold

    successes_ : int
        Number of times the test score succeeded in meeting self.threshold
    """

    def __init__(
        self,
        model,
        performance_function,
        performance_threshold,
        n_iter=100,
        subsample_ratio=0.5,
        pollute_type="random_k",
        pollute_k=1,
        additional_pollution=True,
        drop_features=False,
        min_features=None,
        drop_every_n_iters=5,
        use_predict_proba=False,
        feature_names=None,
        verbose=False,
    ):
        self.model = model
        self.metric = performance_function
        self.threshold = performance_threshold
        self.n_iter = n_iter
        self.subsample_ratio = subsample_ratio
        self.pollute_type = pollute_type
        self.pollute_k = pollute_k
        self.additional_pollution = additional_pollution
        self.drop_features = drop_features
        self.min_features = min_features
        self.drop_every_n_iters = drop_every_n_iters
        self.use_predict_proba = use_predict_proba
        self.verbose = verbose

        if feature_names:
            if type(feature_names) != list:
                raise TypeError("Feature names is not a list.")
            self._name_mapping = {idx: name for idx, name in enumerate(feature_names)}

        if pollute_type and type(pollute_k) != int:
            raise TypeError("Pollute k must be an integer.")

        if not drop_features:
            if min_features:
                warnings.warn(
                    "Min features specified but drop_features is false. Ignoring."
                )
            if drop_every_n_iters != 5:
                warnings.warn(
                    "drop_every_n_iters modified but drop_features is false. Ignoring."
                )
        else:
            if not min_features:
                raise TypeError("Min features not defined for drop_features.")

        # parameters set using intuition
        self._additional_pollute_k = 2 if self.additional_pollution else 0
        self._drop_start_iter = 5

        if not hasattr(model, "fit"):
            raise TypeError("Model does not have a fit() method")
        if not hasattr(model, "predict"):
            raise TypeError("Model does not have a predict() method")

    def fit(self, X, y):
        """Find consistently useful features

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            Input samples

        y : array-like of shape [n_samples,]
            Target labels

        Returns
        -------
        self : object

        """

        self._init_fit_params(X)
        n_features = X.shape[1]
        mask_array = np.zeros(n_features)
        if self.pollute_k > n_features:
            raise ValueError("Pollute k must be less than or equal # of features")

        num_workers = mp.cpu_count()
        pool = mp.Pool(num_workers)

        results = []
        for iter_idx in range(self.n_iter):
            results.append(pool.apply_async(self._find_pollute_mask, args=(X, y)))
        pool.close()
        pool.join()

        results = [result.get() for result in results]
        masks, train_scores, test_scores = list(zip(*results))

        successes = [m for m in masks if m is not None]
        self.successes_ += len(successes)
        self.failures_ += len([m for m in masks if m is None])
        mask_array += np.sum(successes, axis=0)

        self.feature_importances_ = mask_array / self.successes_

        return self

    def _find_pollute_mask(self, X, y):
        X_sample = X[:, self.retained_features_]
        n_features = X_sample.shape[1]
        X_pollute = self._pollute_data(X_sample)
        X_train, X_test, y_train, y_test = train_test_split(
            X_pollute,
            y,
            test_size=(1 - self.subsample_ratio),
            stratify=y,
            shuffle=True,
        )
        train_score, test_score = self._fit_predict_score(
            X_train, X_test, y_train, y_test
        )
        if test_score >= self.threshold:
            pollute_shape = self._get_pollute_shape()
            importances = self.model.feature_importances_
            mask = self._create_mask_from_pollution(
                n_features, pollute_shape, importances
            )
            return mask, train_score, test_score
        else:
            if self.verbose:
                print(
                    f"Did not meet tests threshold: {self.metric}>{self.threshold}"
                )
            return None, train_score, test_score

    def transform(self, X):
        """Return the dataset with the selected features

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            Input samples

        Returns
        -------
        X_transformed : array-like of shape = [n_samples, self.retained_features_]
            Input samples with a subset of features selected

        """
        if not hasattr(self, "retained_features_"):
            raise NotFittedError(
                "Selector has not been fit or retained feature list is empty."
            )
        return X[:, self.retained_features_]

    def fit_transform(self, X, y):
        """

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            Input samples

        y : array-like of shape [n_samples,]
            Target labels

        Returns
        -------
        X_transformed : array-like of shape = [n_samples, self.retained_features_]
            Input samples with a subset of features selected
        """
        self.fit(X, y)
        return self.transform(X)

    def _record_iter(self, iter_idx, train_score, test_score):
        """Track scores, features and importances at each iteration in the object"""
        self.train_scores_[iter_idx] = train_score
        self.test_scores_[iter_idx] = test_score
        self._features_at_iter.append(self.retained_features_.copy())
        self._importances_at_iter[iter_idx, :] = self.feature_importances_

    def _init_fit_params(self, X_sample):
        """Initialize model attributes defined during predict"""
        self.X_orig = X_sample
        n_features = X_sample.shape[1]
        self.retained_features_ = list(np.arange(n_features))
        self.feature_importances_ = np.zeros(n_features)
        self.dropped_features_ = list()
        self.train_scores_ = np.zeros(self.n_iter)
        self.test_scores_ = np.zeros(self.n_iter)
        self.failures_ = 0
        self.successes_ = 0
        self._features_at_iter = list()
        self._importances_at_iter = np.zeros(shape=(self.n_iter, n_features))

    def _eval_dropping_conditions(self, iter_idx):
        """Evaluate conditions for dropping features and drop the min importance"""
        dropping_condition = (
            iter_idx >= self._drop_start_iter
            and iter_idx % self.drop_every_n_iters == 0
            and self.min_features
            and len(self.retained_features_) > self.min_features
        )

        if dropping_condition:
            min_feature = np.argmin(self.feature_importances_[self.retained_features_])
            if hasattr(self, "_name_mapping") and self.verbose:
                print("Dropping: ", self._name_mapping[min_feature])
            else:
                if self.verbose:
                    print(
                        "Dropping feature with index:",
                        self.retained_features_[min_feature],
                    )
            self.dropped_features_.append(self.retained_features_.pop(min_feature))

    @staticmethod
    @njit(parallel=True)
    def _create_mask_from_pollution(n_features, pollute_shape, importances):
        """Create mask by comparing original feature importances to polluted features"""
        pollute_idx = np.arange(n_features, n_features + pollute_shape)

        mask = np.zeros(n_features)
        for i in prange(n_features):
            mask[i] = np.all(importances[i] > importances[pollute_idx])

        return mask

    def _get_pollute_shape(self):
        """Create pollution shape"""
        pollute_shape = (
            self.pollute_k + self._additional_pollute_k
            if self.additional_pollution
            else self.pollute_k
        )
        return pollute_shape

    def _fit_predict_score(self, X_train, X_test, y_train, y_test):
        """fit and predict model, returning scores on defined metric"""
        self.model.fit(X_train, y_train)
        if not hasattr(self.model, "feature_importances_"):
            raise NotFittedError("Selector not fitted (no feature importances).")

        if not self.use_predict_proba:
            train_preds = self.model.predict(X_train)
            test_preds = self.model.predict(X_test)
        else:
            train_preds = self.model.predict_proba(X_train)
            test_preds = self.model.predict_proba(X_test)

        train_score = self.metric(y_train, train_preds)
        test_score = self.metric(y_test, test_preds)
        return train_score, test_score

    def _pollute_data(self, X):
        """Create polluted feature representation (adds noisy features)"""

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

        if self.additional_pollution:
            X_pollute[:, i + 1] = np.random.randint(0, 2, n_samples)
            X_pollute[:, i + 2] = np.random.rand(n_samples)

        return np.concatenate((X, X_pollute), axis=1)

    def plot_test_scores_by_iters(self):
        """Plot tests scores against feature importances"""
        if not hasattr(self, "feature_importances_"):
            raise NotFittedError("Model has not been fit yet.")

        plt.plot(range(len(self.test_scores_)), self.test_scores_, marker="o")
        plt.title("Test scores (average across k-folds) by iterations.")
        plt.show()

    def plot_test_scores_by_n_features(self):
        """Plot tests scores against n_features"""
        if not hasattr(self, "feature_importances_"):
            raise NotFittedError("Model has not been fit yet.")
        feature_lens = [len(l) for l in self._features_at_iter]
        scores_ = self.test_scores_
        test_df = pd.DataFrame({"n_features": feature_lens, "test_score": scores_})
        test_df_agg = test_df.groupby("n_features").mean()
        plt.plot(test_df_agg.index, test_df_agg.test_score, marker="o")
        plt.title("Test scores (average across k-folds) by number of features.")
        plt.show()


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=10000, n_features=20, n_informative=10, n_redundant=5
    )

    model = RandomForestClassifier()
    selector = PollutionSelect(
        model,
        n_iter=100,
        pollute_type="random_k",
        drop_features=False,
        performance_threshold=0.7,
        performance_function=acc,
        min_features=4,
    )

    import time
    start = time.time()
    X_dropped = selector.fit_transform(X, y)
    end = time.time()
    print(end - start)
    print(selector.feature_importances_)

