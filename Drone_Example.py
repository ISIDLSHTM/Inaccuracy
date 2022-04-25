import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.proportion as ssp

def calc_s(b,f,m,g,a):
    """
    Calculates the distance a drone will fly with parameters b,f,g,m,a
    :param b: Battery Size
    :param f: Forward Force
    :param m: Mass (excluding battery)
    :param g: Gravity
    :param a: Drain Rate of battery
    :return: Distance
    """
    top = (f-(g*(m+b)))*b
    bottom = (m+b)*a
    return top/bottom

def get_optimal_b(f, m, g, a):
    """
    Finds the optimal battery size assuming parameters f,g,m,a
    :param f: Forward Force
    :param m: Mass (excluding battery)
    :param g: Gravity
    :param a: Drain Rate of battery
    :return: b: Battery Size
    """
    first = -m + math.sqrt(f*m/g)
    second = -m - math.sqrt(f*m/g)
    return max(first,second)


def _test_accuracy(true_params, error_scale):
    est_params = np.random.normal(true_params, error_scale*true_params)
    est_params[est_params<0.01] = 0.01 # Just done to ensure that model is well defined.
    exp_f, exp_m, exp_g, exp_a = est_params

    exp_b = get_optimal_b(exp_f, exp_m, exp_g, exp_a)
    exp_s = calc_s(exp_b, exp_f, exp_m, exp_g, exp_a)
    true_s = calc_s(exp_b, true_f, true_m, true_g, true_a)


    if exp_s == true_s:
        print('exact', exp_b, exp_s, true_s, e)
        return 0.5
    elif exp_s > true_s:
        return 1
    return 0

def test_accuracy(true_params, error_scale, iters):
    outcomes = np.zeros(iters)
    for i in range(iters):
        outcomes[i] = _test_accuracy(true_params, error_scale)
    return np.asarray(outcomes)



true_f = 250.0
true_m = 1.0
true_g = 10.0
true_a = 1.0
error_rate = .5
true_params = np.asarray([true_f, true_m, true_g, true_a])

iters = 10000

### Effect of increasing error scale


overs = []
es = np.linspace(0.01,2,51)
for e in es:
    outcomes = test_accuracy(true_params, e, iters)
    overs.append(np.mean(outcomes))


lcb = []
ucb = []
for over in overs:
    l, u = ssp.proportion_confint(int(over*iters), iters)
    lcb.append(l)
    ucb.append(u)

plt.plot(es, overs)
plt.plot(es, lcb, c = 'blue', ls = '--')
plt.plot(es, ucb, c = 'blue', ls = '--')
plt.hlines(.5, 0, 2)
plt.ylim(0, 1)
plt.xlabel('Error Scale')
plt.ylabel('Proportion of trials overestimated')
plt.title('Effect of parameter estimation variance on overestimation bias')
plt.show()
