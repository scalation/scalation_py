"""
__author__ = "Mohammed Aldosari"
__date__ = 2/24/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import time
import numpy as np
import pandas as pd
from numpy import ndarray
from tqdm import tqdm
from typing import Optional
from util.data_loading import load_data
from util.data_visualization import plot_train_test
from util.data_transforms import data_transform_std, inverse_transformation
from util.data_splitting import train_test_split
from statsmodels.tsa.ar_model import AutoReg

def AR(file_name: str, training_ratio: float, horizon: int, main_output: str, normalization: bool, model: str, p: int,
       vis_h: Optional[int], startH: int, inverse_transform: bool, trend: str) -> (np.array, np.array):
    """
    The AR model is the most straightforward forecasting model. It regresses the future values on a given number of past lags.
    The AR model is determined with a p order, which specifies the number of lags used in the model. With some conditions,
    The AR model with order p = 1 can be considered a simple random walk model.
    The partial autocorrelation function (PACF) can be used to inform the selection of the AR order where one wants to select an order with high significant PACF scores.
    The parameters of the AR model are estimated using Ordinary Least Squares (OLS).
    Note: The forecasts of the AR model are generated using the Iterative Forecasting approach.
    For more information, please see https://en.wikipedia.org/wiki/Autoregressive_model
                                     https://cobweb.cs.uga.edu/~jam/scalation_guide/comp_data_science.pdf
                                     https://www.statsmodels.org/dev/generated/statsmodels.tsa.ar_model.AutoReg.html

    Parameters
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
    p: int
        the number of lags to use for the AR model. partial autocorrelation (PACF) is often used for informed selection
        of this parameter where one wants to select the lag with a high significant correlation.
    vis_h: Optional[int]
        this parameter specifies whether to visualize all the forecasting horizons (vis_h = None) or a specific horizon (vis_h = 4).
    startH: int
        startH controls the horizon to which we are interested in forecasting and visualizing. it may be highly useful for long forecasting horizons.
        for example, if (horizon = 28 and startH = 28), we will only get forecasts for the 28th horizon while omitting the previous one.
    inverse_transform: bool
        this parameter does the inverse transformation where we are interested in viewing and visualizing results on the original scale.
    trend: str
        this parameter allows ones to incorporate a trend component into the AR model if the data exhibits a trend.
        These are the options for trends provided by Statsmodels:
        ‘n’ - No trend.
        ‘c’ - Constant only.
        ‘t’ - Time trend only.
        ‘ct’ - Constant and time trend.

    Returns
    -------
    actual: ndarray[float]
    forecasts: ndarray[float]
    """
    horizon = horizon - 1
    startH = startH -1
    data = load_data(file_name, main_output = main_output)
    train_size = int(training_ratio * len(data))
    if normalization:
        scaled_mean_std, data = data_transform_std(data, train_size)
    train_data, val_data, test_data = train_test_split(data, train_ratio = training_ratio)  # No validation data for the Random Walk model.
    train_data_MO: pd.DataFrame = train_data[[main_output]]  # Train set for main output column.
    test_data_MO: pd.DataFrame = test_data[[main_output]]  # Test set for main output column.
    actual: ndarray[float] = np.zeros(shape = (len(test_data_MO) - horizon, horizon + 1))  # Make an initital array for storing the actual values.
    forecasts: ndarray[float] = np.zeros(shape = (len(test_data_MO) - horizon, horizon + 1))   # Make an initital array for storing the forecasts values.
    start_time = time.time()
    for i in tqdm(range(len(test_data_MO) - horizon)):
        data_temp = data.iloc[0:train_size + i][main_output]
        model_ar = AutoReg(data_temp, lags = p, trend = trend, seasonal = False).fit()
        predictions = model_ar.predict(start = len(train_data_MO) + i, end = len(train_data_MO) + i + horizon)
        for j in range(startH, horizon + 1):
            actual[i, j] = data.iloc[train_size + i + j, :][main_output]
            forecasts[i, j] = predictions.values[j]
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Time:{total_time} seconds.")
    if inverse_transform:
        train_data_MO, test_data_MO, actual, forecasts = inverse_transformation(train_data_MO, test_data_MO, actual, forecasts, scaled_mean_std['scaled'+main_output], 'standard_scaler')
    plot_train_test(data, main_output, train_size, train_data_MO, test_data_MO, forecasts, horizon, model, vis_h, startH)
    return actual, forecasts, model_ar