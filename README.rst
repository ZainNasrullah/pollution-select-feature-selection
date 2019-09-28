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

    - generate k+2 polluted features by permutating k random features and creating two noisy features
    - trains the model on a polluted training set with d + k + 2 features and checks that the desired performance threshold is met on the test set (else skip iteration)
    - Compares the importance of each original feature to every polluted feature. Assigns each feature a score of 1 for the iteration if its importance is greater than every noisy feature
    - Update the overall importance of each feature as cumulative_score / n_iterations 

-------
Install
-------

-----------
Quick Start
-----------
