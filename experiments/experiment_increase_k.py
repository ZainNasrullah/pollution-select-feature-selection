import numpy as np
import pandas
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
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
    n_features = 15
    n_relevant = 5
    n_redundant = 0
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    random_state = 42
    model = RandomForestClassifier(random_state=random_state)

    X, y = make_classification(
        n_samples=1000,
        n_features=n_features,
        n_informative=n_relevant,
        n_redundant=n_redundant,
        shuffle=False,
        random_state=random_state
    )
    X, y = shuffle(X, y, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    mask_type = "delta_negative"

    importance_list = []
    for pollute_k in range(1, n_features + 1):
        start = time.time()
        binary_params = dict(
            mask_type=mask_type, n_iter=100, pollute_k=pollute_k
        )
        selector = run_pollution_select(X_train, y_train, model, **binary_params)
        mask = selector.feature_importances_ > 0
        model.fit(X_train[:, mask], y_train)
        preds = model.predict(X_test[:, mask])
        result = acc(y_test, preds)

        mean_relevant_imp = np.mean(selector.feature_importances_[:n_relevant])
        max_relevant_imp = np.max(selector.feature_importances_[:n_relevant])
        min_relevant_imp = np.min(selector.feature_importances_[:n_relevant])
        mean_noisy_imp = np.mean(selector.feature_importances_[n_relevant:])
        max_noisy_imp = np.max(selector.feature_importances_[n_relevant:])
        min_noisy_imp = np.min(selector.feature_importances_[n_relevant:])
        importance_list.append(
            (
                mean_relevant_imp,
                max_relevant_imp,
                min_relevant_imp,
                mean_noisy_imp,
                max_noisy_imp,
                min_noisy_imp,
                result,
            )
        )
        end = time.time()
        print(f"{pollute_k}/{n_features} -- {end - start:.2f}s")

    mean_relevant, max_relevant, min_relevant, mean_noise, max_noise, min_noise, results = zip(
        *importance_list
    )

    model.fit(X_train[:, :n_relevant], y_train)
    preds = model.predict(X_test[:, :n_relevant])
    relevant_result = acc(y_test, preds)

    x = range(n_features)
    plt.plot(x, mean_relevant, label="mean_relevant", c="b", linestyle="-")
    plt.plot(x, max_relevant, label="max/min relevant", c="b", linestyle=":")
    plt.plot(x, min_relevant, c="b", linestyle=":")
    plt.fill_between(x=x, y1=max_relevant, y2=min_relevant, color='b', alpha=0.15)
    plt.plot(x, mean_noise, label="mean_noise", c="orange", linestyle="-")
    plt.plot(x, max_noise, label="max/min relevant", c="orange", linestyle=":")
    plt.plot(x, min_noise, c="orange", linestyle=":")
    plt.fill_between(x=x, y1=max_noise, y2=min_noise, color='orange', alpha=0.15)
    plt.plot(x, results, label="test_acc", c="g", linestyle="-", linewidth=2)
    plt.axhline(relevant_result, label="relevant_test_acc", c="g", linestyle=":")
    plt.xlabel("Pollute K")
    plt.ylabel("Feature Importance")
    plt.legend(fancybox=True, loc=7).get_frame().set_alpha(0.4)
    plt.title(f"{mask_type} Feature Importance by Pollute K")
    plt.savefig("outputs/imp_by_k.png")
    plt.show()

