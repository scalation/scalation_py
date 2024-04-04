"""
__author__ = "Mohammed Aldosari"
__date__ = 4/4/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import numpy as np
import scipy.stats 
def prediction_interval(actual, forecasts, t, h, percentile):
    """
    https://otexts.com/fpp2/prediction-intervals.html
    """
    errors = actual[t+h-1:] - forecasts[:,h-1]
    errors_std = np.std(errors) 
    forecasts_std_h = errors_std * np.sqrt(h)
    critical_value = scipy.stats.norm.ppf(percentile) 
    return forecasts_std_h * critical_value
