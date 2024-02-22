import numpy as np
import pandas as pd
from typing import Any, Union
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit

from util.data_loading import load_data, plot_train_test
from util.data_transforms import data_transform_std
from util.data_splitting import train_test_split

def MeanModel(file_name: str, training_ratio: float, horizon: int, main_output: str, normalization: bool) -> (np.array, np.array):
    """
    A function used for producing forecasts by taking the mean of the training data and project it into the future for all the future horizons.

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

    train_data, val_data, test_data = train_test_split(data, train_ratio=training_ratio)  # No validation data for the Mean Model.
    train_data_MO: pd.DataFrame = train_data[[main_output]]  # Train set for main output column.
    train_data_MO_mean = train_data_MO.mean()
    test_data_MO: pd.DataFrame = test_data[[main_output]]  # Test set for main output column.
    actual: ndarray[Any, dtype[Union[floating[_64Bit], float_]]] = np.zeros(shape=(len(test_data_MO) - horizon, horizon + 1))  # Make an initital array for storing the actual values.
    forecasts: ndarray[Any, dtype[Union[floating[_64Bit], float_]]] = np.zeros(shape=(len(test_data_MO) - horizon, horizon + 1))   # Make an initital array for storing the forecasts values.
    for i in range(len(test_data_MO) - horizon):
        for j in range(horizon + 1):
            actual[i, j] = float(data.iloc[train_size + i + j, :][main_output]) # Record the observed data for the the future horizons.
            forecasts[i, j] = float(train_data_MO_mean.iloc[0])  # Record the mean of the entire training data for each forecasting horizon.
    plot_train_test(data, main_output, train_size, train_data_MO, test_data_MO, forecasts, horizon)
    return actual, forecasts