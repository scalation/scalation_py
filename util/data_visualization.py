"""
__author__ = "Mohammed Aldosari"
__date__ = 2/25/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.data_loading import load_data
from util.data_transforms import data_transform_std
from util.data_splitting import train_test_split
import statsmodels.api as sm
def plot_data(data: pd.DataFrame):
    """
    A function used for plotting the data

    Arguments
    ----------
    data: DataFrame
        the name of the dataset

    Returned Values
    ----------

    """
    return data.plot(subplots=True, figsize=(10, 12))


def plot_train_test(df_raw_scaled: pd.DataFrame, main_output: str, train_size: float, train: pd.DataFrame,
                    test: pd.DataFrame, forecasts: pd.DataFrame = None, horizon=24,
                    model = 'Random Walk', vis_h = 1) -> None:
    plt.subplots(figsize=(7, 4))
    plt.plot(train, color='red', label='Observed Train')
    plt.plot(test, color='blue', label='Observed Test')
    if forecasts is not None:
        if vis_h is not None:
            idx = np.arange(train_size + vis_h, train_size + vis_h + forecasts.shape[0], 1)
            plt.plot(idx, forecasts[:, vis_h - 1], color=(random.randint(0, 255)/255.0,
                                                  random.randint(0, 255)/255.0,
                                                  random.randint(0, 255)/255.0),
                     label=str('Forecasts ' + 'h ' + str(vis_h)))
        else:
            for i in range(horizon + 1):
                idx = np.arange(train_size + i, train_size + i + forecasts.shape[0], 1)
                plt.plot(idx, forecasts[:, i], color=(random.randint(0, 255)/255.0,
                                                      random.randint(0, 255)/255.0,
                                                      random.randint(0, 255)/255.0),
                         label=str('Forecasts ' + 'h ' + str(i + 1)))
    plt.ticklabel_format(style='plain')
    plt.title(model + ' - ' + main_output)
    plt.legend()
    plt.show()

def plot_time_series(file_name: str, main_output: str):
    if main_output is None:
        data = load_data(file_name, main_output=main_output)
        data.plot(subplots=True, figsize=(10, 7), title=main_output)
    else:
        data = load_data(file_name, main_output=main_output)
        plt.subplots(figsize=(7, 4))
        plt.plot(data[[main_output]], label=main_output)
        plt.title(main_output)
        plt.show()
    return

def plot_acf(file_name: str, main_output: str, lags: int, diff_order: int):
    data = load_data(file_name, main_output=main_output)
    data = data[[main_output]]
    if diff_order is not None:
        for _ in range(diff_order):
            data = data.diff().dropna()
    sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=lags)
    plt.show()
    return

def plot_pacf(file_name: str, main_output: str, lags: int, diff_order: int):
    data = load_data(file_name, main_output=main_output)
    data = data[[main_output]]
    if diff_order is not None:
        for _ in range(diff_order):
            data = data.diff().dropna()
    sm.graphics.tsa.plot_pacf(data.values.squeeze(), lags=lags)
    plt.show()
    return

def plot_seasonal_difference(file_name: str, main_output: str, lags: int, diff_order: int, diff_orders: int):
    data = load_data(file_name, main_output=main_output)
    data = data[[main_output]]
    data_or = data
    data = data.diff(diff_order).dropna()
    data = pd.concat([data_or.head(1), data], axis=0)
    data = data.diff(diff_orders).dropna()
    data = pd.concat([data_or.head(diff_orders), data], axis=0)
    sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=lags)
    plt.show()
    return
