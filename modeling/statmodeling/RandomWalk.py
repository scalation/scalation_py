"""
__author__ = "Mohammed Aldosari"
__date__ = 2/22/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import time
import numpy as np
import pandas as pd
from typing import Any, Union
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from tqdm import tqdm

from util.data_loading import load_data
from util.data_visualization import plot_train_test
from util.data_transforms import data_transform_std
from util.data_transforms import inverse_transformation
from util.data_splitting import train_test_split

def RandomWalk(file_name: str, training_ratio: float, horizon: int, main_output: str, normalization: bool, model: str,
        vis_h: int, LTSF: bool, inverse_transform: bool) -> (np.array, np.array):
    """
    A function used for producing forecasts based on the Random Walk model that simply projects a current value into the future (yhat[t] = y[t-h]).

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

    train_data, val_data, test_data = train_test_split(data, train_ratio=training_ratio)  # No validation data for the Random Walk model.
    train_data_MO: pd.DataFrame = train_data[[main_output]]  # Train set for main output column.
    test_data_MO: pd.DataFrame = test_data[[main_output]]  # Test set for main output column.
    actual: ndarray[Any, float_] = np.zeros(shape=(len(test_data_MO) - horizon, horizon + 1))  # Make an initital array for storing the actual values.
    forecasts: ndarray[Any, float_] = np.zeros(shape=(len(test_data_MO) - horizon, horizon + 1))   # Make an initital array for storing the forecasts values.
    start_time = time.time()
    for i in tqdm(range(len(test_data_MO) - horizon)):
        for j in range(horizon + 1):
            actual[i, j] = float(data.iloc[train_size + i + j, :][main_output])    # Record the observed data for the the future horizons.
            forecasts[i, j] = float(data.iloc[train_size + i - 1,:][main_output])  # Start with the last value from the training set and store it for each forecasting horizon.
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Time:{total_time} seconds.")
    if inverse_transform:
        train_data_MO, test_data_MO, actual, forecasts = inverse_transformation( train_data_MO, test_data_MO, actual, forecasts, scaled_mean_std['scaled'+main_output], 'standard_scaler')
    plot_train_test(data, main_output, train_size, train_data_MO, test_data_MO, forecasts, horizon, model, vis_h)
    return actual, forecasts