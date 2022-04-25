import numpy as np
from useful_functions import *
import matplotlib.pyplot as plt

np.random.seed(1234)
iters = 10000

def _test_accuracy(n, m_vec, sd_vec):
    """
    Returns whether a simulated binary trial overestimated or underestimated the probability of success
    for the predicted optimal arm.
    :param n: Number of trial individuals per arm
    :param m_vec: Vector of means for the arms
    :param sd_vec: Vector of standard deviations for the arms
    :return: -1 if underestimated, 0 if accurate, 1 if overestimated
    """
    assert np.shape(m_vec) == np.shape(sd_vec)
    outcome_vec = np.random.normal(m_vec,sd_vec, (n,np.size(m_vec)))
    mean_vec = np.mean(outcome_vec, 0)
    predicted_best_arg = randargmax(mean_vec)
    predicted_best_m = mean_vec[predicted_best_arg]
    actual_m_at_best_arg = m_vec[predicted_best_arg]
    if predicted_best_m < actual_m_at_best_arg:  # Underestimation
        return -1, predicted_best_arg, predicted_best_m
    elif predicted_best_m > actual_m_at_best_arg:  # Overestimation
        return 1, predicted_best_arg, predicted_best_m
    else:  # Accurate estimation
        return 0, predicted_best_arg, predicted_best_m


def test_accuracy(n, m_vec, sd_vec, iters=iters):
    """
    Wrapper of _test_accuracy which simulates 'iters' trials
    :param n: Number of trial individuals per arm
    :param m_vec: Vector of means for the arms
    :param sd_vec: Vector of standard deviations for the arms
    :param iters: Number of simulated trials
    :return: numpy array of -1 if underestimated, 0 if accurate, 1 if overestimated
    """
    outcomes = np.zeros(iters)
    predicted_best_args, predicted_best_ms = np.zeros(iters), np.zeros(iters)
    for i in range(iters):
        outcomes[i],predicted_best_args[i], predicted_best_ms[i] = _test_accuracy(n, m_vec, sd_vec)
    return np.asarray(outcomes), predicted_best_args, predicted_best_ms


ns = [
    10, 100, 10, 100,
    10, 100, 10, 100,
    10, 100
]

m_vecs = [
    [10, 10], [10, 10], [10, 9], [10, 9],
    [10, 10], [10, 10], [10, 9], [10, 9],
    [50, 20], [50, 20]
]

sd_vecs = [
    [2, 2], [2, 2], [2, 2], [2, 2],
    [4, 4], [4, 4], [4, 4], [4, 4],
    [2, 2], [2, 2]
]

for n, m_vec, sd_vec in zip(ns, m_vecs, sd_vecs):
    outcome,predicted_best_args, predicted_best_ms = test_accuracy(n, m_vec, sd_vec)
    beautify_output_m(n, m_vec, sd_vec, outcome)

    xs = rand_jitter(predicted_best_args)
    plt.scatter(xs, predicted_best_ms)
    plt.scatter(range(len(m_vec)), m_vec)
    plt.show()

