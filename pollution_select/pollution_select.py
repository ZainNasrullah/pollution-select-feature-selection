# Authors: Zain Nasrullah <zain.nasrullah.zn@gmail.com>
#
#
# License: MIT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from numba import njit, prange
import multiprocessing as mp
from typing import List, Union, Tuple, Optional, Callable
import collections.abc
from sklearn.utils import shuffle
import inspect


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

    data_subsample_ratio : float, optional (default=0.5)
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
        performance_function: Callable[[np.ndarray, np.ndarray], float],
        performance_threshold: float,
        n_iter: int = 100,
        data_subsample_ratio: float = 0.5,
        pollute_type: str = "random_k",
        pollute_k: int = 1,
        additional_pollution: bool = True,
        mask_type: str = "binary",
        drop_features: bool = False,
        min_features: int = 1,
        drop_every_n_iters: int = 5,
        use_predict_proba: bool = False,
        feature_names: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.metric = performance_function
        self.threshold = performance_threshold
        self.n_iter = n_iter
        self.data_subsample_ratio = data_subsample_ratio
        self.pollute_type = pollute_type
        self.pollute_k = pollute_k
        self.additional_pollution = additional_pollution
        self.mask_type = mask_type
        self.drop_features = drop_features
        self.min_features = min_features
        self.drop_every_n_iters = drop_every_n_iters
        self.use_predict_proba = use_predict_proba
        self.verbose = verbose

        # check model
        if not hasattr(self.model, "fit"):
            raise TypeError("Model does not have a fit() method")
        if not hasattr(self.model, "predict"):
            raise TypeError("Model does not have a predict() method")

        # check metric
        if not len(inspect.signature(self.metric).parameters) == 2:
            raise TypeError(
                "metric should only take in two parameters,"
                " e.g., acc(y_true, y_preds)"
            )

        # check threshold
        if not isinstance(self.threshold, (int, float)):
            raise TypeError("self.threshold is not an int or float")

        # check iters
        if n_iter <= 0:
            raise ValueError(
                "Number of iterations cannot be less than or equal to zero."
            )
        elif n_iter <= 10:
            warnings.warn("Less than 10 iters doesn't give good importance scores.")

        # check data subsample
        if not (0 < self.data_subsample_ratio < 1):
            raise ValueError("Data subsample should be greater than 0 and less than 1.")

        # check pollute type and k
        if pollute_type not in ["random_k", "all"]:
            raise ValueError("Pollution type should be 'random_k' or 'all'")
        if pollute_type == "random_k" and (
            not isinstance(pollute_k, int) or pollute_k <= 0
        ):
            raise TypeError("Pollute k must be a positive integer greater than 0.")

        # check additional pollution
        if not isinstance(self.additional_pollution, bool):
            raise TypeError("Additional pollution is not a boolean.")

        # check mask type
        if mask_type not in ["binary", "delta_weighted", "negative_score"]:
            raise ValueError("Mask type should be 'binary' or 'weighted'")

        # check feature dropping
        if not isinstance(self.drop_features, bool):
            raise TypeError("drop features is not a boolean.")

        if drop_features:
            if self.min_features == 1:
                warnings.warn(
                    "Minimum number of features is set to 1 "
                    "and this can lead to almost every feature being dropped."
                    " Please modify this value for the desired behavior."
                )
            if not isinstance(self.min_features, int) and self.min_features < 1:
                raise TypeError("min features is not a positive integer.")

            if not isinstance(drop_every_n_iters, int) and self.min_features < 1:
                raise TypeError("drop_every_n_iters is not a positive integer.")

        # check predict proba
        if not isinstance(self.use_predict_proba, bool):
            raise TypeError("use_predict_proba is not a boolean.")

        # check feature names
        if feature_names:
            if not isinstance(feature_names, collections.abc.Sequence) or isinstance(
                feature_names, str
            ):
                raise TypeError("Feature names is not a list.")

            self._name_mapping = {idx: name for idx, name in enumerate(feature_names)}

        # check verbosity
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose is not a boolean.")

        # parameters set using intuition
        self._additional_pollute_k = 2 if self.additional_pollution else 0
        self._drop_start_iter = 5

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
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
        if not isinstance(X, np.ndarray):
            raise TypeError("X is not a numpy array.")

        if not isinstance(y, np.ndarray):
            raise TypeError("y is not a numpy array.")

        n_features = X.shape[1]
        if hasattr(self, "_name_mapping") and len(self._name_mapping) != X.shape[1]:
            raise ValueError(
                f"Length of feature names {len(self._name_mapping)} don't match number of features {n_features}."
            )

        if self.pollute_k > n_features:
            raise ValueError(
                f"Pollute k must be less than or equal # of features {n_features}"
            )
        mask_array = np.zeros(n_features)
        self._init_fit_params(X)
        self.parallel = True

        if self.parallel:
            num_workers = mp.cpu_count()
            pool = mp.Pool(num_workers)
            iters = int(self.n_iter / self.drop_every_n_iters)
        else:
            iters = self.n_iter

        for iter_idx in range(iters):
            X_sample = X[:, self.retained_features_]

            # number of features may change as features are dropped across iters
            n_features = X_sample.shape[1]
            X_pollute = self._pollute_data(
                X_sample, self.pollute_type, self.pollute_k, self._additional_pollute_k
            )

            if self.parallel:
                train_scores = []
                test_scores = []
                for _ in range(self.drop_every_n_iters):
                    mask, test_score, train_score = self._train_model(
                        X_pollute, y, n_features
                    )
                    train_scores.append(train_score)
                    test_scores.append(test_score)
            else:
                mask, test_score, train_score = self._train_model(
                    X_pollute, y, n_features
                )
                train_scores = [train_score]
                test_scores = [test_score]

            for train_score, test_score in zip(train_scores, test_scores):
                if test_score >= self.threshold:
                    mask_array[self.retained_features_] += mask
                    self.successes_ += 1
                    self.feature_importances_[self.retained_features_] = (
                        mask_array[self.retained_features_] / self.successes_
                    )
                else:
                    if self.verbose:
                        print(
                            f"Did not meet tests threshold: {self.metric}>{self.threshold}"
                        )
                    self.failures_ += 1

            self._record_iter(iter_idx, train_score, test_score)
            if self.drop_features:
                self._eval_dropping_conditions(iter_idx)

        if self.parallel:
            pool.close()
            pool.join()

        return self

    def _train_model(self, X_pollute, y, n_features):
        X_train, X_test, y_train, y_test = train_test_split(
            X_pollute,
            y,
            test_size=(1 - self.data_subsample_ratio),
            stratify=y,
            shuffle=True,
        )
        train_score, test_score = self._fit_predict_score(
            X_train, X_test, y_train, y_test
        )

        pollute_shape = self._get_pollute_count()
        mask = self._create_mask_from_pollution(
            n_features=n_features,
            pollute_shape=pollute_shape,
            importances=self.model.feature_importances_,
        )
        return mask, test_score, train_score

    def transform(self, X: np.ndarray) -> np.ndarray:
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

        if not isinstance(X, np.ndarray):
            raise TypeError("X is not a numpy array.")

        if X.shape[1] != self.X_orig.shape[1]:
            raise ValueError(
                f"n_features do not match original data.\n"
                f"X: {X.shape[1]} != X_orig {self.X_orig.shape[1]}"
            )

        return X[:, self.retained_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
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

    def _record_iter(
        self, iter_idx: int, train_score: float, test_score: float
    ) -> None:
        """Track scores, features and importances at each iteration in the object"""
        self.train_scores_[iter_idx] = train_score
        self.test_scores_[iter_idx] = test_score
        self._features_at_iter.append(self.retained_features_.copy())
        self._importances_at_iter[iter_idx, :] = self.feature_importances_

    def _init_fit_params(self, X_sample: np.ndarray) -> None:
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

    def _eval_dropping_conditions(self, iter_idx: int) -> None:
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

    def _create_mask_from_pollution(self, **kwargs) -> np.ndarray:

        if self.mask_type == "binary":
            mask = self._create_binary_mask_from_pollution(**kwargs)
        elif self.mask_type == "delta_weighted":
            mask = self._create_weighted_mask_from_pollution(**kwargs)
        elif self.mask_type == "negative_score":
            mask = self._create_negative_weight_mask_from_pollution(**kwargs)
        else:
            mask = None

        return mask

    @staticmethod
    def _create_binary_mask_from_pollution(
        n_features: int, pollute_shape: int, importances: np.ndarray
    ) -> np.ndarray:
        """Create binary mask by comparing original feature importances
         to polluted features and scoring 1 only if feature is more
         important than every noisy feature."""
        pollute_idx = np.arange(n_features, n_features + pollute_shape)

        mask = np.zeros(n_features)
        for i in range(n_features):
            mask[i] = np.all(importances[i] > importances[pollute_idx])

        return mask

    @staticmethod
    def _create_weighted_mask_from_pollution(
        n_features: int, pollute_shape: int, importances: np.ndarray
    ) -> np.ndarray:
        """ Extends the binary mask by additionally weighting by differences in
         feature importance and then normalizing into [0,1]"""
        pollute_idx = np.arange(n_features, n_features + pollute_shape)

        mask = np.zeros(n_features)
        deltas = np.zeros(n_features)
        for i in range(n_features):
            mask[i] = np.all(importances[i] > importances[pollute_idx])
            deltas[i] = importances[i] - np.mean(importances[pollute_idx])

        weighted_mask = mask * deltas
        max_element = np.max(weighted_mask)
        min_element = np.min(weighted_mask)
        weighted_mask_norm = (weighted_mask - min_element) / (max_element - min_element)

        return weighted_mask_norm

    @staticmethod
    def _create_negative_weight_mask_from_pollution(
        n_features: int, pollute_shape: int, importances: np.ndarray
    ) -> np.ndarray:
        """ Extends the binary mask by assigning an original features which fails
         to beat a noisy feature a negative score """
        pollute_idx = np.arange(n_features, n_features + pollute_shape)
        scaling = 1 / pollute_shape

        mask = np.zeros(n_features)
        for i in range(n_features):
            feat_comparison = importances[i] > 2 * importances[pollute_idx]
            scaled_comparison = np.where(feat_comparison, scaling, -1 * scaling)
            mask[i] = np.sum(scaled_comparison)

        return mask

    def _get_pollute_count(self) -> int:
        """Create pollution shape"""
        pollute_shape = (
            self.pollute_k + self._additional_pollute_k
            if self.additional_pollution
            else self.pollute_k
        )
        return pollute_shape

    def _fit_predict_score(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[float, float]:
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

        if not isinstance(train_score, (int, float)):
            raise TypeError("self.metric did not return an integer or float")

        if not isinstance(test_score, (int, float)):
            raise TypeError("self.metric did not return an integer or float")

        return train_score, test_score

    @staticmethod
    @njit
    def _pollute_data(
        X: np.ndarray, pollute_type: str, pollute_k: int, additional_pollute_k: int
    ) -> np.ndarray:
        """Create polluted feature representation (adds noisy features)"""

        n_samples, n_features = X.shape

        if pollute_type == "all":
            pollute_shape = X.shape[1]
        elif pollute_type == "random_k":
            pollute_shape = pollute_k
        else:
            raise Exception("pollute type not in ('all', 'random_k')")

        pollute_shape += additional_pollute_k
        X_pollute = np.zeros(shape=(n_samples, pollute_shape))
        random_feats = np.arange(n_features)
        random_feats = np.random.permutation(random_feats)[
            0 : pollute_shape - additional_pollute_k
        ]

        # add permuted features
        for i, feat_idx in enumerate(random_feats):
            X_pollute[:, i] = np.random.permutation(X[:, feat_idx])

        # add random noise
        if additional_pollute_k == 2:
            X_pollute[:, i + 1] = np.random.randint(0, 2, n_samples)
            X_pollute[:, i + 2] = np.random.rand(n_samples)

        return np.concatenate((X, X_pollute), axis=1)

    def plot_test_scores_by_iters(self) -> None:
        """Plot tests scores against feature importances"""
        if not hasattr(self, "feature_importances_"):
            raise NotFittedError("Model has not been fit yet.")

        plt.plot(range(len(self.test_scores_)), self.test_scores_, marker="o")
        plt.title("Test scores (average across k-folds) by iterations.")
        plt.show()

    def plot_test_scores_by_n_features(self) -> None:
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

    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    import time

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    def acc(y, preds):
        return np.mean(y == preds)

    X, y = make_classification(
        n_samples=10000, n_features=15, n_informative=5, n_redundant=5, shuffle=False
    )
    X, y = shuffle(X, y)

    model = RandomForestClassifier()

    print("Binary Mask:")
    selector = PollutionSelect(
        model,
        n_iter=100,
        pollute_type="random_k",
        drop_features=False,
        performance_threshold=0.7,
        performance_function=acc,
        mask_type="binary",
    )

    start = time.time()
    X_dropped = selector.fit_transform(X, y)
    end = time.time()
    print("{0:.2f}s".format(end - start))
    print("Relevant:", selector.feature_importances_[:5])
    print("Redundant:", selector.feature_importances_[5:10])
    print("Noise:", selector.feature_importances_[10:])

    print("\nDelta Weighted Mask:")
    selector = PollutionSelect(
        model,
        n_iter=100,
        pollute_type="random_k",
        drop_features=False,
        performance_threshold=0.7,
        performance_function=acc,
        mask_type="delta_weighted",
    )

    start = time.time()
    X_dropped = selector.fit_transform(X, y)
    end = time.time()
    print("{0:.2f}s".format(end - start))
    print("Relevant:", selector.feature_importances_[:5])
    print("Redundant:", selector.feature_importances_[5:10])
    print("Noise:", selector.feature_importances_[10:])

    print("\nNegative Score Mask:")
    selector = PollutionSelect(
        model,
        n_iter=100,
        pollute_type="random_k",
        drop_features=False,
        performance_threshold=0.7,
        performance_function=acc,
        mask_type="negative_score",
    )

    start = time.time()
    X_dropped = selector.fit_transform(X, y)
    end = time.time()
    print("{0:.2f}s".format(end - start))
    print("Relevant:", selector.feature_importances_[:5])
    print("Redundant:", selector.feature_importances_[5:10])
    print("Noise:", selector.feature_importances_[10:])
