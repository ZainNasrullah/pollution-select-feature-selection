.. highlight:: rst

----------
Background
----------

Pollution Select is a feature selection algorithm method based on ideas from
boruta and other iterative selection methods. It finds features that consistently achieve
a desired performance criteria and are more important than random noise in
monte carlo cross-validation. 

---------
Algorithm
---------

* As input, Pollution Select receives a model, a performance evaluation function and a threshold.
* For n_iters:

  - Generate k+2 polluted features by permuting k random features and creating two noisy features
  - Train the model on a polluted training set with d + k + 2 features and checks that the desired performance threshold is met on the test set (else skip iteration)
  - Compare the importance of each original feature to every polluted feature. Assigns each feature a score of 1 for the iteration if its importance is greater than every noisy feature
  - Update the overall importance of each feature as cumulative_score / n_iterations

-------
Install
-------

The simplest way to install right now is to clone this repo and do a local install:

    pip install .

-----------
Quick Start
-----------

Simple example without dropping any features:

.. code-block:: python

   import numpy as np
   from sklearn.datasets import load_iris
   from sklearn.ensemble import RandomForestClassifier
   from pollution_select import PollutionSelect

   iris = load_iris()
   X = iris.data
   y = iris.target
   X_noise = np.concatenate(
       (np.random.rand(150, 1), X, np.random.rand(150, 1)), axis=1
   )

   def acc(y, preds):
       return np.mean(y == preds)

   selector = PollutionSelect(
       RandomForestClassifier(),
       performance_function=acc,
       performance_threshold=0.7,
   )

   selector.fit_transform(X_noise, y)
   print(selector.feature_importances_)


More complex example with feature dropping:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from pollution_select import PollutionSelect

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_redundant=5
    )

    def acc(y, preds):
        return np.mean(y == preds)

    selector = PollutionSelect(
        RandomForestClassifier(),
        n_iter=100,
        pollute_type="random_k",
        drop_features=True,
        performance_threshold=0.7,
        performance_function=acc,
        min_features=4,
    )

    print(selector.retained_features_)
    print(selector.dropped_features_)
    print(selector.feature_importances_)

    selector.plot_test_scores_by_iters()
    selector.plot_test_scores_by_n_features()
