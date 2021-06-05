import numpy as np

from sklearn.utils import shuffle
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time

import sys

sys.path.append("..")
from pollution_select import PollutionSelect


def acc(y, preds):
    return np.mean(y == preds)


def run_pollution_select(X, y, model, mask_type, n_iter, pollute_k):
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
        verbose=True,
    )
    start = time.time()
    selector.fit_transform(X, y)
    end = time.time()

    file = open("outputs/compare_masks.txt", 'a')
    print("\n" + mask_type)
    print("{0:.2f}s".format(end - start))
    print("Relevant:", selector.feature_importances_[:5])
    print("Noise:", selector.feature_importances_[5:])
    file.write(mask_type)
    file.write("\n{0:.2f}s".format(end - start))
    file.write(f"\nRelevant: {selector.feature_importances_[:5]}")
    file.write(f"\nNoise: {selector.feature_importances_[5:]}\n\n")
    file.close()

    return selector


if __name__ == "__main__":

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    file = open("outputs/compare_masks.txt", 'w')

    random_state = 42
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=5,
        n_redundant=0,
        shuffle=False,
        random_state=random_state,
    )
    X, y = shuffle(X, y, random_state=random_state)

    model = RandomForestClassifier(random_state=random_state)

    binary_params = dict(mask_type="binary", n_iter=100, pollute_k=1)
    run_pollution_select(X, y, model, **binary_params)

    delta_weighted_params = dict(mask_type="delta", n_iter=100, pollute_k=1)
    run_pollution_select(X, y, model, **delta_weighted_params)

    negative_score_params = dict(mask_type="negative", n_iter=100, pollute_k=1)
    run_pollution_select(X, y, model, **negative_score_params)

    delta_negative_params = dict(mask_type="delta_negative", n_iter=100, pollute_k=1)
    run_pollution_select(X, y, model, **delta_negative_params)

    delta_negative_params = dict(
        mask_type="extreme_delta_negative", n_iter=100, pollute_k=1
    )
    run_pollution_select(X, y, model, **delta_negative_params)
