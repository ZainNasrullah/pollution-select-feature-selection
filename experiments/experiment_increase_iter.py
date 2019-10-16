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
    )
    X_dropped = selector.fit_transform(X, y)

    return selector


if __name__ == "__main__":

    n_iterations = 100
    n_features = 150
    n_relevant = 50
    n_redundant = 50
    n_useful = n_relevant + n_redundant
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
    for iter_idx in range(1, n_iterations + 1):
        start = time.time()
        binary_params = dict(mask_type="binary", n_iter=iter_idx, pollute_k=3)
        selector = run_pollution_select(X, y, model, **binary_params)

        relevant_imp = np.mean(selector.feature_importances_[:n_relevant])
        redundant_imp = np.mean(selector.feature_importances_[n_relevant:n_useful])
        noisy_imp = np.mean(selector.feature_importances_[n_useful:])
        importance_list.append((relevant_imp, redundant_imp, noisy_imp))
        end = time.time()
        print(f"{iter_idx}/{n_iterations} -- {end - start:.2f}s")

    relevant, redundant, noise = zip(*importance_list)
    x = range(n_iterations)
    plt.plot(x, relevant, label="relevant")
    plt.plot(x, redundant, label="redundant")
    plt.plot(x, noise, label="noise")
    plt.xlabel("Iterations")
    plt.ylabel("Avg Feature Importance")
    plt.legend()
    plt.title("Feature Importance by Iterations")

    plt.show()
