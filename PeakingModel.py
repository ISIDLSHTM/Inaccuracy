import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.proportion as ssp
from useful_functions import *

iters = 10000
np.random.seed(1234)


def _test_accuracy(query_doses,true_params, error_scale):
    est_params = true_params + np.random.normal(0,[error_scale,error_scale, error_scale*.1], 3)
    est_params[est_params<0.01] = 0.01 # Just done to ensure that model is well defined. This effects ~2% of simulations for error_scale = 0.5
    optimal_dose, optimal_response = optimise(query_doses,est_params)
    act_at_pred = scaled_peaking(optimal_dose, true_params)
    if act_at_pred < optimal_response: # Overestimation
        return 1, optimal_dose, optimal_response
    else:
        return -1, optimal_dose, optimal_response

def test_accuracy(query_doses,true_params, error_scale, iters=iters):
    outcomes = np.zeros(iters)
    predicted_best_dose, predicted_best_r= np.zeros(iters), np.zeros(iters)
    for i in range(iters):
        outcomes[i],predicted_best_dose[i], predicted_best_r[i] = _test_accuracy(query_doses,true_params, error_scale)
    return np.asarray(outcomes), predicted_best_dose, predicted_best_r


### Set up true values
true_height = 3
true_mid = 5
true_scale = 1
true_params = true_height, true_mid, true_scale
query_doses = np.linspace(0,10,101)
true_curve = scaled_peaking(query_doses, true_params)
true_d, true_r = optimise(query_doses, true_params)

error_scale = 0.5


## Visualise True Curve
plt.plot(query_doses,true_curve)
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.scatter(true_d, true_r, c='black')
plt.show()

### 6 Directions visualisation

est_params = true_height - error_scale , true_mid, true_scale
est_curve = scaled_peaking(query_doses,est_params)
est_d, est_r = optimise(query_doses,est_params)
plt.plot(query_doses,true_curve)
plt.plot(query_doses,est_curve)
plt.scatter(est_d, est_r, c='red')
plt.plot([est_d, est_d], [scaled_peaking(est_d, true_params), est_r], ls = '--', c='black')
plt.title('Effect of underestimating maximum ')
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.legend(['True', 'Estimated', 'Inaccuracy', 'Predicted Optimal'])
plt.show()


est_params = true_height + error_scale , true_mid, true_scale
est_curve = scaled_peaking(query_doses,est_params)
est_d, est_r = optimise(query_doses,est_params)
plt.plot(query_doses,true_curve)
plt.plot(query_doses,est_curve)
plt.scatter(est_d, est_r, c='red')
plt.plot([est_d, est_d], [scaled_peaking(est_d, true_params), est_r], ls = '--', c='black')
plt.title('Effect of overestimating maximum ')
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.legend(['True', 'Estimated', 'Inaccuracy', 'Predicted Optimal'])
plt.show()

est_params = true_height, true_mid - error_scale , true_scale
est_curve = scaled_peaking(query_doses,est_params)
est_d, est_r = optimise(query_doses,est_params)
plt.plot(query_doses,true_curve)
plt.plot(query_doses,est_curve)
plt.scatter(est_d, est_r, c='red')
plt.plot([est_d, est_d], [scaled_peaking(est_d, true_params), est_r], ls = '--', c='black')
plt.title('Effect of underestimating midpoint ')
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.legend(['True', 'Estimated', 'Inaccuracy', 'Predicted Optimal'])
plt.show()

est_params = true_height, true_mid + error_scale , true_scale
est_curve = scaled_peaking(query_doses,est_params)
est_d, est_r = optimise(query_doses,est_params)
plt.plot(query_doses,true_curve)
plt.plot(query_doses,est_curve)
plt.scatter(est_d, est_r, c='red')
plt.plot([est_d, est_d], [scaled_peaking(est_d, true_params), est_r], ls = '--', c='black')
plt.title('Effect of overestimating midpoint ')
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.legend(['True', 'Estimated', 'Inaccuracy', 'Predicted Optimal'])
plt.show()

est_params = true_height , true_mid, true_scale - error_scale
est_curve = scaled_peaking(query_doses,est_params)
est_d, est_r = optimise(query_doses,est_params)
plt.plot(query_doses,true_curve)
plt.plot(query_doses,est_curve)
plt.scatter(est_d, est_r, c='red')
plt.plot([est_d, est_d], [scaled_peaking(est_d, true_params), est_r], ls = '--', c='black')
plt.title('Effect of underestimating scale ')
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.legend(['True', 'Estimated', 'Inaccuracy', 'Predicted Optimal'])
plt.show()


est_params = true_height , true_mid, true_scale + error_scale
est_curve = scaled_peaking(query_doses,est_params)
est_d, est_r = optimise(query_doses,est_params)
plt.plot(query_doses,true_curve)
plt.plot(query_doses,est_curve)
plt.scatter(est_d, est_r, c='red')
plt.plot([est_d, est_d], [scaled_peaking(est_d, true_params), est_r], ls = '--', c='black')
plt.title('Effect of overestimating scale ')
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.legend(['True', 'Estimated', 'Inaccuracy', 'Predicted Optimal'])
plt.show()



# Single Error Scale

outcomes, predicted_best_dose, predicted_best_r = test_accuracy(query_doses,true_params, error_scale)

overestmiated = outcomes == 1
overestimated_d, overestimated_r = predicted_best_dose[overestmiated], predicted_best_r[overestmiated]
underestimated = outcomes == -1
underestimated_d, underestimated_r = predicted_best_dose[underestimated], predicted_best_r[underestimated]




plt.plot(query_doses, scaled_peaking(query_doses, true_params))
plt.scatter([],[], alpha = 1, c='blue')
plt.scatter([],[], alpha = 1, c='red')
plt.scatter(np.mean(predicted_best_dose), np.mean(predicted_best_r), c='black')
plt.scatter(underestimated_d, underestimated_r, alpha = .05, c='blue')
plt.scatter(overestimated_d, overestimated_r, alpha = .05, c='red')
plt.scatter(np.mean(predicted_best_dose), np.mean(predicted_best_r), c='black')
plt.title('Display of overestimation bias')
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.legend(['True', 'Underestimated Predicted Optimal', 'Overestimated Predicted Optimal', 'Mean of Predicted Optimals'])
plt.show()


### Effect of increasing error scale


overs = []
es = np.linspace(0.01,1,11)
for e in es:
    print(e)
    outcomes, _, _ = test_accuracy(query_doses, true_params, e)
    overs.append(np.count_nonzero(outcomes==1)/iters)


lcb = []
ucb = []
for over in overs:
    l, u = ssp.proportion_confint(int(over*iters), iters)
    lcb.append(l)
    ucb.append(u)

plt.plot(es, overs)
plt.plot(es, lcb, c = 'blue', ls = '--')
plt.plot(es, ucb, c = 'blue', ls = '--')
plt.ylim(0, 1)
plt.ylabel('Overestimation probability')
plt.xlabel('σ')
plt.title('Effect of parameter estimation variance on overestimation bias')
plt.show()
