import numpy as np

from sklearn.utils import shuffle
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time

import sys
sys.path.append('..')
from pollution_select import PollutionSelect


def acc(y, preds):
    return np.mean(y == preds)


def run_pollution_select(X, y, model, mask_type, n_iter, pollute_k):
    print('\n' + mask_type)
    selector = PollutionSelect(
        model,
        n_iter=n_iter,
        pollute_type="random_k",
        pollute_k=pollute_k,
        drop_features=False,
        performance_threshold=0.7,
        performance_function=acc,
        mask_type=mask_type,
        parallel=True,
    )
    start = time.time()
    X_dropped = selector.fit_transform(X, y)
    end = time.time()
    print("{0:.2f}s".format(end - start))
    print("Relevant:", selector.feature_importances_[:5])
    print("Redundant:", selector.feature_importances_[5:10])
    print("Noise:", selector.feature_importances_[10:])

    return selector


if __name__ == "__main__":

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    X, y = make_classification(
        n_samples=1000, n_features=15, n_informative=5, n_redundant=5, shuffle=False
    )
    X, y = shuffle(X, y)

    # model = RandomForestClassifier()
    model = GradientBoostingClassifier()

    binary_params = dict(mask_type="binary", n_iter=100, pollute_k=5)
    run_pollution_select(X, y, model, **binary_params)

    delta_weighted_params = dict(mask_type="delta_weighted", n_iter=100, pollute_k=5)
    run_pollution_select(X, y, model, **delta_weighted_params)

    negative_score_params = dict(mask_type="negative_score", n_iter=100, pollute_k=5)
    run_pollution_select(X, y, model, **negative_score_params)

    delta_negative_params = dict(mask_type="delta_negative", n_iter=100, pollute_k=5)
    run_pollution_select(X, y, model, **delta_negative_params)

    test_weighted_params = dict(mask_type="test_weighted", n_iter=100, pollute_k=5)
    run_pollution_select(X, y, model, **test_weighted_params)
