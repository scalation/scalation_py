"""
__author__ = "Mohammed Aldosari"
__date__ = 2/22/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import numpy as np
from typing import Any, Union
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit

from util.data_loading import load_data, plot_train_test
from util.data_transforms import data_transform_std
from util.data_splitting import train_test_split

def SimpleMovingAverage(file_name: str, training_ratio: float, horizon: int, main_output: str, normalization: bool,
                        window: int, model: str) -> (np.array, np.array):
    """
    A function used for producing forecasts by taking the mean of previous values according to a given window (w = 1 is the Random Walk model).

    Arguments
    ----------
    file_name: str
        the file path for csv data file.
    training_ratio: float
        the training ratio used for splitting the dataset into train and test
    horizon: int
        how many time steps ahead to make the forecasts
    main_output: str
        the main output column/feature, e.g. '% WEIGHTED ILI'
    normalization: bool
        specifies whether the data is normalized or original
    model: str
        model name

    Returned Values
    ----------
    actual: ndarray[Any, dtype[Union[floating[_64Bit], float_]]]
    forecasts: ndarray[Any, dtype[Union[floating[_64Bit], float_]]]
    """
    horizon = horizon - 1
    data = load_data(file_name, main_output=main_output)
    train_size = int(training_ratio * len(data))
    if normalization:
        scaled_mean_std, data = data_transform_std(data, train_size)

    train_data, val_data, test_data = train_test_split(data, train_ratio=training_ratio)  # No validation data for the SMA model.
    train_data_MO = train_data[[main_output]]  # Train set for main output column.
    test_data_MO = test_data[[main_output]]  # Test set for main output column.
    actual: ndarray[Any, dtype[Union[floating[_64Bit], float_]]] = np.zeros(shape=(len(test_data_MO) - horizon, horizon + 1))  # Make an initital array for storing the actual values.
    forecasts: ndarray[Any, dtype[Union[floating[_64Bit], float_]]] = np.zeros(shape=(len(test_data_MO) - horizon, horizon + 1))   # Make an initital array for storing the forecasts values.
    for i in range(len(test_data_MO) - horizon):
        for j in range(horizon + 1):
            actual[i, j] = float(data.iloc[train_size + i + j, :]['new_deaths'])  # Record the observed data for the the future horizons.
            forecasts[i, j] = float(data.iloc[train_size + i - window:train_size + i, :]['new_deaths'].mean())  # Take the mean of the last number of time steps based on a given window and record record it for each forecasting horizon.
    plot_train_test(data, main_output, train_size, train_data_MO, test_data_MO, forecasts, horizon, model)
    return actual, forecasts