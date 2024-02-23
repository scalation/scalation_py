"""
__author__ = "Mohammed Aldosari"
__date__ = 2/22/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import numpy as np
from util.metrics import mse, mae, smape


def getmetrics(actual: np.array, forecasts: np.array) -> (float, float, float):
    """
    A function used for returning the MSE, MAE, sMAPE metrics.

    Arguments
    ----------
    actual: np.array
        the observed test set values aligned with the forecasts
    forecasts: np.array | List[floats]
        the output from a forecasting model, i.e. RandomWalk

    Returned Values
    ----------
    mse: float
    mae: float
    smape: float

    """
    return mse(actual, forecasts), mae(actual, forecasts), smape(actual, forecasts)
