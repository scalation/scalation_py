"""
__author__ = "Mohammed Aldosari"
__date__ = 2/24/24
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
from util.data_transforms import data_transform_std, inverse_transformation
from util.data_splitting import train_test_split
from statsmodels.tsa.ar_model import AutoReg

def AR(file_name: str, training_ratio: float, horizon: int, main_output: str, normalization: bool, model: str, p: int
        , vis_h: int, last_only: bool, inverse_transform: bool, trend: str) -> (np.array, np.array):
    horizon = horizon - 1
    data = load_data(file_name, main_output=main_output)
    train_size = int(training_ratio * len(data))
    if normalization:
        scaled_mean_std, data = data_transform_std(data, train_size)
    train_data, val_data, test_data = train_test_split(data, train_ratio=training_ratio)  # No validation data for the Random Walk model.
    train_data_MO: pd.DataFrame = train_data[[main_output]]  # Train set for main output column.
    test_data_MO: pd.DataFrame = test_data[[main_output]]  # Test set for main output column.
    actual: ndarray[Any, float] = np.zeros(shape=(len(test_data_MO) - horizon, horizon + 1))  # Make an initital array for storing the actual values.
    forecasts: ndarray[Any, float] = np.zeros(shape=(len(test_data_MO) - horizon, horizon + 1))   # Make an initital array for storing the forecasts values.
    start_time = time.time()
    if last_only:
        for i in tqdm(range(len(test_data_MO) - horizon)):
            data_temp = data.iloc[0:train_size + i][main_output]
            model = AutoReg(data_temp, lags=p, trend=trend, seasonal = False).fit()
            actual[i, horizon] = data.iloc[train_size + i + horizon, :][main_output]
            forecasts[i, horizon] = model.predict(start=len(train_data_MO) + i, end=len(train_data_MO) + i + horizon).values[-1]
            forecast = forecasts[i, horizon]
    else:
        for i in tqdm(range(len(test_data_MO) - horizon)):
            data_temp = data.iloc[0:train_size + i][main_output]
            model_ar = AutoReg(data_temp, lags=p, trend=trend, seasonal=False).fit()
            predictions = model_ar.predict(start=len(train_data_MO) + i, end=len(train_data_MO) + i + horizon)
            for j in range(horizon + 1):
                actual[i, j] = data.iloc[train_size + i + j, :][main_output]
                forecasts[i, j] = predictions.values[j]
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Time:{total_time} seconds.")
    if inverse_transform:
        train_data_MO, test_data_MO, actual, forecasts = inverse_transformation( train_data_MO, test_data_MO, actual, forecasts, scaled_mean_std['scaled'+main_output], 'standard_scaler')
    plot_train_test(data, main_output, train_size, train_data_MO, test_data_MO, forecasts, horizon, model, vis_h)
    return actual, forecasts, model_ar