import numpy as np
from useful_functions import *

np.random.seed(1234)
iters = 10000

def _test_accuracy(n, p_vec):
    """
    Returns whether a simulated binary trial overestimated or underestimated the probability of success
    for the predicted optimal arm.
    :param n: Number of trial individuals 
    :param p_vec: Vector of success probabilities
    :return: -1 if underestimated, 0 if accurate, 1 if overestimated
    """
    success_vec = np.random.binomial(n, p_vec)
    predicted_best_arg = randargmax(success_vec)
    predicted_best_p = success_vec[predicted_best_arg]/n
    actual_p_at_best_arg = p_vec[predicted_best_arg]
    if predicted_best_p < actual_p_at_best_arg: # Underestimation
        return -1
    elif predicted_best_p > actual_p_at_best_arg: # Overestimation
        return 1
    else: # Accurate estimation
        return 0

def test_accuracy(n, p_vec, iters=iters):
    """
        Wrapper of _test_accuracy which simulates 'iters' trials
        :param n: Number of trial individuals 
        :param p_vec: Vector of success probabilities
        :param iters: Number of simulated trials
        :return: numpy array of -1 if underestimated, 0 if accurate, 1 if overestimated
        """
    outcomes = np.zeros(iters)
    for i in range(iters):
        outcomes[i] = _test_accuracy(n, p_vec)
    return np.asarray(outcomes)


ns = [
    10,100,10,100,
    10,100,10,100,
    20,10,10
]

p_vecs = [
    [.5,.5],[.5,.5],[.6,.7],[.6,.7],
    [.1,.9],[.1,.9],[.5,.5,.5],[.5,.5,.5],
    [.5,.6,.7],[.1,.1,.5],[0.0,.0,.5]
]

for n, p_vec in zip(ns, p_vecs):
    outcome = test_accuracy(n, p_vec)
    beautify_output_p(n, p_vec, outcome)
