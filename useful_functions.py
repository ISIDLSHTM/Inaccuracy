import numpy as np
import scipy.stats as sts

def randargmax(vec):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(vec.shape) * (vec == np.max(vec)))


def beautify_output_p(n, p_vec, outcome):
    len_outcome = np.size(outcome)
    print(f'For n = {n}, probability vector = {p_vec}')
    print(f'    Underestimated: {np.count_nonzero(outcome == -1)/len_outcome} = '
          f'{np.count_nonzero(outcome == -1)}/{len_outcome}')
    print(f'    Accurately estimated: {np.count_nonzero(outcome == 0) / len_outcome} = '
          f'{np.count_nonzero(outcome == 0)}/{len_outcome}')
    print(f'    Overestimated: {np.count_nonzero(outcome == 1) / len_outcome} = '
          f'{np.count_nonzero(outcome == 1)}/{len_outcome}')

def beautify_output_m(n, m_vec, sd_vec, outcome):
    len_outcome = np.size(outcome)
    print(f'For n = {n}, mean vector = {m_vec}, deviation vector = {sd_vec}')
    print(f'    Underestimated: {np.count_nonzero(outcome == -1)/len_outcome} = '
          f'{np.count_nonzero(outcome == -1)}/{len_outcome}')
    print(f'    Accurately estimated: {np.count_nonzero(outcome == 0) / len_outcome} = '
          f'{np.count_nonzero(outcome == 0)}/{len_outcome}')
    print(f'    Overestimated: {np.count_nonzero(outcome == 1) / len_outcome} = '
          f'{np.count_nonzero(outcome == 1)}/{len_outcome}')


def rand_jitter(arr):
    stdev = .01
    return arr + np.random.randn(len(arr)) * stdev


def scaled_peaking(independant, params):
    height, mid, spread= params
    max = sts.norm.pdf(mid, mid, spread)
    return height * sts.norm.pdf(independant, mid, spread)/max

def optimise(query_doses,params):
    utility = scaled_peaking(query_doses, params)
    index = np.argmax(utility)
    optimal_dose = query_doses[index]
    optimal_response = utility[index]
    return optimal_dose, optimal_response
