import numpy as np
import scipy.stats as sts

def randargmax(vec):
    """
    A tie breaking argmax that breaks ties at random
    :param vec: vector of values to choose maximum argument of
    :return: Maximum Argument of vec
    """
    return np.argmax(np.random.random(vec.shape) * (vec == np.max(vec)))


def beautify_output_p(n, p_vec, outcome):
    """
    Prints a nice description of outcome data for the binomial trials
    """
    len_outcome = np.size(outcome)
    print(f'For n = {n}, probability vector = {p_vec}')
    print(f'    Underestimated: {np.count_nonzero(outcome == -1)/len_outcome} = '
          f'{np.count_nonzero(outcome == -1)}/{len_outcome}')
    print(f'    Accurately estimated: {np.count_nonzero(outcome == 0) / len_outcome} = '
          f'{np.count_nonzero(outcome == 0)}/{len_outcome}')
    print(f'    Overestimated: {np.count_nonzero(outcome == 1) / len_outcome} = '
          f'{np.count_nonzero(outcome == 1)}/{len_outcome}')

def beautify_output_m(n, m_vec, sd_vec, outcome):
    """
    Prints a nice description of outcome data for the Continuous trials
    """
    len_outcome = np.size(outcome)
    print(f'For n = {n}, mean vector = {m_vec}, deviation vector = {sd_vec}')
    print(f'    Underestimated: {np.count_nonzero(outcome == -1)/len_outcome} = '
          f'{np.count_nonzero(outcome == -1)}/{len_outcome}')
    print(f'    Accurately estimated: {np.count_nonzero(outcome == 0) / len_outcome} = '
          f'{np.count_nonzero(outcome == 0)}/{len_outcome}')
    print(f'    Overestimated: {np.count_nonzero(outcome == 1) / len_outcome} = '
          f'{np.count_nonzero(outcome == 1)}/{len_outcome}')


def rand_jitter(arr):
    """
    Used to jitter values for nice plotting 
    """
    stdev = .01
    return arr + np.random.randn(len(arr)) * stdev


def scaled_peaking(independant, params):
    """
    The peaking dose utility curve used for the model example
    :param independant: The 'dose' values
    :param params: height, mid, spread which are the maximum utility, 
    best dose,
    and spread of the dose utility curve respectively
    :return: the 'utility' values'
    """
    height, mid, spread= params
    max = sts.norm.pdf(mid, mid, spread)
    return height * sts.norm.pdf(independant, mid, spread)/max

def optimise(query_doses,params):
    """
    
    :param query_doses: The dose values to choose the optimal of.
    :param params:  height, mid, spread which are the maximum utility, 
    best dose,
    and spread of the dose utility curve respectively
    :return: tuple of (the best dose according to these parameters, the utility at that dose)
    """
    utility = scaled_peaking(query_doses, params)
    index = np.argmax(utility)
    optimal_dose = query_doses[index]
    optimal_response = utility[index]
    return optimal_dose, optimal_response
