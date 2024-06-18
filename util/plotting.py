"""
__author__ = "Mohammed Aldosari"
__date__ = 2/25/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.data_loading import load_data
from util.transformations import data_transform_std
from util.data_splitting import train_test_split
import statsmodels.api as sm
from matplotlib import cm

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

def plot_time_series(self, args):
    """
    A function for plotting the time series before doing any preprocessing and modeling.

    Parameters
    ----------
    file_name: str
        the file path for csv data file.
    main_output: str
        the main output column/feature, e.g. '% WEIGHTED ILI'

    Returns
    -------

    """
    data = load_data(self, args.file_name, main_output=None)
    plt.subplots(figsize=(7, 4))
    plt.plot(data[[args.target]], color = 'black', marker = 'o', linewidth=0.5, markersize = 1)
    plt.title(args.dataset + ' ' + args.target)
    plt.grid(False)
    file_path = os.path.join(self.folder_path_plots, str(self.args.dataset) +'.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def plot_data_forecasts(self,
                    h: int,
                    forecasts: pd.DataFrame) -> None:

    if self.args.in_sample:
        train_size = self.args.seq_len + h
    else:
        train_size = self.train_size
    plt.subplots(figsize=(7, 4))

    if self.args.forecast_type.lower() == 'point':
        idx = np.arange(train_size, train_size + forecasts.shape[0], 1)
        plt.plot(self.data_.iloc[:-(self.args.max_horizon - 1), self.data_.columns.get_loc(self.args.target)], color='black',
                 linewidth=0.5, marker='o', markersize=1, label='Observed Data')
        plt.plot(idx, forecasts, color = 'red', marker = 'o', markersize = 1, linewidth=0.5,
                 label=str('Forecasts h = ' + str(h + 1)))

    plt.ylabel("Original Scale")
    plt.ticklabel_format(style='plain')
    plt.title(self.args.validation+'\n Model: '+ self.args.model_name + ' Dataset: ' + self.args.dataset + ' Target: ' + self.args.target)
    plt.grid(False)
    plt.legend()
    file_path = os.path.join(self.folder_path_plots, str(self.args.validation) + '_' + str(self.args.model_name) + '_' + str(self.args.max_horizon) + '_' + str(h + 1) +'.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def plot_acf(file_name: str, main_output: str, lags: int, diff_order: int, start_index: int):
    """
    A function for plotting the autocorrelation function (ACF).

    Parameters
    ----------
    file_name: str
        the file path for csv data file.
    main_output: str
        the main output column/feature, e.g. '% WEIGHTED ILI'
    lags: int
        lags in which we are interested in calculating and visualizing autocorrelation
    diff_order: int
        control the order of differencing before calculating and visualizing autocorrelation
    start_index: int
        determines the start index of the data before calculating and plotting ACF.
        this can be help to inspect non-stationary data as the start index may reveal different ACF scores.
    """
    data = load_data(file_name, main_output=main_output)
    data = data[[main_output]]
    data = data.iloc[start_index:]
    if diff_order is not None:
        for _ in range(diff_order):
            data = data.diff().dropna()
    #data = data - data.shift(24)
    #data = data.dropna()
    sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=lags)
    plt.show()

def plot_pacf(file_name: str, main_output: str, lags: int, diff_order: int, start_index: int):
    """
    A function for plotting the partial autocorrelation function (PACF).

    Parameters
    ----------
    file_name: str
        the file path for csv data file.
    main_output: str
        the main output column/feature, e.g. '% WEIGHTED ILI'
    lags: int
        lags in which we are interested in calculating and visualizing partial autocorrelation
    diff_order: int
        control the order of differencing (first order) before calculating and visualizing partial autocorrelation
    start_index: int
        determines the start index of the data before calculating and plotting PACF.
        this can be help to inspect non-stationary data as the start index may reveal different PACF scores.
    """
    data = load_data(file_name, main_output=main_output)
    data = data[[main_output]]
    data = data.iloc[start_index:]
    if diff_order is not None:
        for _ in range(diff_order):
            data = data.diff().dropna()
    sm.graphics.tsa.plot_pacf(data.values.squeeze(), lags=lags)
    plt.show()

def plot_seasonal_difference(file_name: str, main_output: str, lags: int, diff_order: int, diff_orders: int,
                             function: str):
    """
    A function for plotting the ACF or the PACF of data using simple first order or seasonal differencing.

    Parameters
    ----------
    file_name: str
        the file path for csv data file.
    main_output: str
        the main output column/feature, e.g. '% WEIGHTED ILI'
    lags: int
        lags in which we are interested in calculating and visualizing partial autocorrelation
    diff_order: int
        control the order of differencing (first order) before calculating and visualizing autocorrelation
    diff_orders: int
        control the order of differencing (seasonal order) before calculating and visualizing autocorrelation
    """
    data = load_data(file_name, main_output=main_output)
    data = data[[main_output]]
    data_or = data
    data = data.diff(diff_order).dropna()
    data = pd.concat([data_or.head(1), data], axis=0)
    data = data.diff(diff_orders).dropna()
    data = pd.concat([data_or.head(diff_orders), data], axis=0)
    if function == 'ACF':
        sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=lags)
    elif function == 'PACF':
        sm.graphics.tsa.plot_pacf(data.values.squeeze(), lags=lags)
    plt.show()


def plot_cross_correlation(file_name: str, main_output: str, lags: int, start_index: int):
    """

    Parameters
    ----------
    file_name
    main_output
    lags

    Returns
    -------

    """
    data = load_data(file_name, main_output='new_deaths')
    data = data.iloc[start_index:]
    data = data.drop('date', axis=1)
    data[main_output+'_lagged'] = data[main_output]
    num_columns = len(data.columns)
    columns = data.columns
    #lags = lags
    results = np.zeros((num_columns, num_columns, lags))
    endog_variable = main_output
    for i in range(num_columns):
        for lag in range(lags):
            lagged_data = data.loc[:, data.columns != endog_variable].shift(lag + 1, fill_value=0)
            #data.columns.get_loc(main_output) whether to have endog variable in the same position.
            lagged_data.insert(data.shape[1] - 1, endog_variable, data[[endog_variable]])
            results[:, :, lag] = lagged_data.corr().to_numpy()
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    norm = plt.Normalize(results.min(), results.max())
    for i in range(0, results.shape[0]):
        for j in range(0, results.shape[1]):
            for k in range(0, results.shape[2]):
                color = cm.hot_r(norm(results[i, j, k]))
                ax.scatter(i + 1, j + 1, k + 1, marker='o', s=50, color=color)

    ax.set_xticks(range(1, results.shape[0] + 1))
    ax.set_yticks(range(1, results.shape[1] + 1))
    ax.set_zticks(range(1, results.shape[2] + 1))
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.hot_r)
    mappable.set_array([])
    plt.colorbar(mappable, ax=ax, label='Cross-correlation Scores', shrink=0.75, aspect=20)
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    #ax.xaxis.pane.set_facecolor('whitesmoke')
    columns = [col for col in columns if col != main_output] + [main_output]
    columns = [col.replace('_', ' ').replace('%', '').upper() for col in columns]
    xlabels = [col if col != columns[-1] else f'$\\bf{{{col}}}$' for col in columns]
    ylabels = [col if col != columns[-1] else f'$\\bf{{{col}}}$' for col in columns]
    ax.set_xticklabels(xlabels, fontsize = 10, rotation = 'vertical')
    ax.set_yticklabels(ylabels, fontsize = 10, rotation = 90)

    plt.xticks(range(1, num_columns + 1))
    plt.yticks(range(1, num_columns + 1))
    ax.grid(False)
    ax.view_init(elev=20, azim=20)
    ax.set_xlabel('Variables')
    ax.set_ylabel('Variables')
    ax.set_zlabel('Lags', rotation=90)
    ax.set_title('Cross-correlation Scores')
    plt.show()
    return results

def plot_forecasts(self, forecasts):
    for h in self.args.horizons:
        ignore = -(h - 1) if h > 1 else None
        if self.train_size is None:
            start_index = 0
            end_index = ignore
        else:
            start_index = self.max_horizon - (h - 1)
            end_index = self.max_horizon - (h - 1) + len(self.test_data)
        plot_data_forecasts(self,
                            h - 1,
                            forecasts[start_index:end_index, self.target_int, h - 1])