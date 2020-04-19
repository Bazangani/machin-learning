import numpy as np
from sklearn.mixture import GaussianMixture

# output: a predicted value mixtures of Gaussian


def predict_exe(execution_array):
    distribution_data = []
    execution_array = np.array(execution_array)

    for number in execution_array:
        try:
            val = float(number)
        except ValueError:
            return 'string cant consider'

    # E0 error of empty array
    if (len(execution_array)) == 0:
        return 'empty array'

    if (len(execution_array)) == 2 or (len(execution_array)) == 1:
        execution = max(execution_array)
        return execution

    if (len(execution_array)) > 1000:
        index_last = len(execution_array)
        execution_array = execution_array[(index_last-100):index_last]

    # Expected maximization with mixture of Gaussian (number_of_Gaussian=3)
    mix_gaussian_model = GaussianMixture(n_components=3, max_iter=100, tol=0.01)
    mix_gaussian_model.fit(np.expand_dims(execution_array, 1))
    # 9 feature as the feature of tests Distribution
    for mu, sd, p in zip(mix_gaussian_model.means_.flatten(), np.sqrt(mix_gaussian_model.covariances_.flatten()), mix_gaussian_model.weights_):
        distribution_data.append([mu, sd, p])
    execution_time = (distribution_data[0][0] * distribution_data[0][2]) + (
            distribution_data[1][0] * distribution_data[1][2]) + (
            distribution_data[2][0] * distribution_data[2][2])

    return int(execution_time)
