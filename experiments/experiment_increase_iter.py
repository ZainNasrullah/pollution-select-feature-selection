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
    print("\n" + mask_type)
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

    n_iterations = 100
    n_features = 15
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    model = RandomForestClassifier()

    X, y = make_classification(
        n_samples=1000,
        n_features=n_features,
        n_informative=5,
        n_redundant=5,
        shuffle=False,
    )
    X, y = shuffle(X, y)

    importance_list = []
    for iter in range(1, n_iterations + 1):
        binary_params = dict(mask_type="binary", n_iter=iter, pollute_k=3)
        selector = run_pollution_select(X, y, model, **binary_params)

        relevant_imp = np.mean(selector.feature_importances_[:5])
        redundant_imp = np.mean(selector.feature_importances_[5:10])
        noisy_imp = np.mean(selector.feature_importances_[10:])
        importance_list.append((relevant_imp, redundant_imp, noisy_imp))

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
