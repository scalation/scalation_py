"""
__author__ = "Mohammed Aldosari"
__date__ = 4/4/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import numpy as np
import scipy.stats
from typing import List
def prediction_intervals(actual: np.array, forecasts: np.array, main_output: str,
                         start_index_test: int, horizon: int, alphas: List[float], model: str):
    """
    A function for calculating prediction intervals.

    Parameters
    ----------
    actual: np.array
        a numpy ndarray containing the observed actual values
    forecasts: np.array
        a numpy ndarray containing the forecasts given by a model
    main_output: str
        main output
    start_index_test: int
        start index of the testset
    horizon: int
        forecasting horizon
    alphas: List[float]
        significance levels
    model: str
        model name

    Returned Values
    ----------
    lower_bounds: np.array
    upper_bounds: np.array

    For more information on prediction intervals, please refer to:
    https://otexts.com/fpp2/prediction-intervals.html
    """
                             
    errors = actual.loc[start_index_test+horizon:, main_output] - forecasts[:, horizon]
    errors_std = np.std(errors)
    if model == 'RandomWalk':
        forecasts_std_h = errors_std * np.sqrt(horizon + 1)
    else:
        raise ValueError('prediction_intervals supports RandomWalk only. '
                         'Please change forecasts_std_h for your own model.')
    lower_bounds = []
    upper_bounds = []
    for alpha in alphas:
        critical_value = scipy.stats.norm.ppf(alpha)
        sigma_hat_h = critical_value * forecasts_std_h
        lower_bound = forecasts[:, horizon] - sigma_hat_h
        upper_bound = forecasts[:, horizon] + sigma_hat_h
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    return lower_bounds, upper_bounds
