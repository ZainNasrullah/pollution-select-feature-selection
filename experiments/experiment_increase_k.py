import numpy as np
import pandas
import matplotlib.pyplot as plt

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
    selector.fit(X, y)

    return selector


if __name__ == "__main__":

    n_iterations = 100
    n_features = 100
    n_relevant = 50
    n_redundant = 0
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    model = RandomForestClassifier()

    X, y = make_classification(
        n_samples=10000,
        n_features=n_features,
        n_informative=n_relevant,
        n_redundant=n_redundant,
        shuffle=False,
    )
    X, y = shuffle(X, y)

    importance_list = []
    for pollute_k in range(1, n_features + 1):
        start = time.time()
        binary_params = dict(mask_type="delta_negative", n_iter=100, pollute_k=pollute_k)
        selector = run_pollution_select(X, y, model, **binary_params)

        relevant_imp = np.max(selector.feature_importances_[:n_relevant])
        noisy_imp = np.max(selector.feature_importances_[n_relevant:])
        importance_list.append((relevant_imp, noisy_imp))
        end = time.time()
        print(f"{pollute_k}/{n_features} -- {end - start:.2f}s")

    relevant, redundant, noise = zip(*importance_list)
    x = range(n_features)
    plt.plot(x, relevant, label="relevant")
    plt.plot(x, noise, label="noise")
    plt.xlabel("Pollute K")
    plt.ylabel("Max Feature Importance")
    plt.legend()
    plt.title("Feature Importance by Pollute K")

    plt.show()
