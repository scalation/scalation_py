"""
__author__ = "Mohammed Aldosari"
__date__ = 11/04/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.rcParams["font.family"] = "Times New Roman"
from prettytable import PrettyTable
import statsmodels.api as sm
import numpy as np
from scipy.stats import shapiro, kstest, anderson
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.api import add_constant, OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from util.QoF import smape

def plot_time_series(self) -> None:
    """
    Plot the target time series for initial visualization. 

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    data = self.data_
    fig, ax = plt.subplots(figsize=(4, 2.5))
    plt.plot(data[[self.target.lower()]], color = 'black', marker = 'o', linewidth=0.5, markersize = 1)
    plt.title(self.dataset + ' ' + self.target + ' - Full Dataset')
    plt.grid(False)
    file_path = os.path.join(self.folder_path_plots, str(self.dataset) +'.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.ylabel("Original Scale")
    plt.xlabel("Time")
    plt.show()
    plt.close()


def plot_acf(self) -> None:
    """
    Plot the autocorrelation function (ACF).

    Parameters
    ----------
    None
    
    Returns
    -------
    None

    """
    fig, ax = plt.subplots(figsize=(4, 2.5))
    data = self.data_[self.target.lower()].iloc[self.start_index_acf_pacf:].copy()
    if self.diff_order_acf_pacf is not None:
        for _ in range(self.diff_order_acf_pacf):
            data = data.diff()
    data = data.dropna()
    sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=self.eda_lags, ax=ax, missing="drop")
    ax.set_title('ACF for ' + self.target.upper())
    file_path = os.path.join(self.folder_path_plots, str(self.dataset) + '_ACF.png')
    fig.savefig(file_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
def plot_pacf(self) -> None:
    """
    Plot the partial autocorrelation function (PACF).

    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(4, 2.5))
    data = self.data_[self.target.lower()].iloc[self.start_index_acf_pacf:].copy()
    if self.diff_order_acf_pacf is not None:
        for _ in range(self.diff_order_acf_pacf):
            data = data.diff()
    data = data.dropna()
    sm.graphics.tsa.plot_pacf(data.values.squeeze(), lags=self.eda_lags, ax=ax)
    ax.set_title('PACF for ' + self.target.upper())
    file_path = os.path.join(self.folder_path_plots, str(self.dataset) + '_PACF.png')
    fig.savefig(file_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    


def plot_forecasts(self) -> None:
    """
    Plots the observed data along with forecasts at the specified horizon.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    legend_x_axis = False
    show_title = False
    show_grid = True
    legend_frameon = True

    for h in self.horizons:

        if 'normalized' in self.plot_mode:
            forecasts = self.forecast_tensor
            if 'test' in self.plot_mode:
                actual = self.data.iloc[self.train_size:, :]
            else:
                actual = self.data
            idx = actual.index
        elif 'original' in self.plot_mode:
            forecasts = self.forecast_tensor_original
            if 'test' in self.plot_mode:
                actual = self.data_.iloc[self.train_size:, :]
            else:
                actual = self.data_
            idx = actual.index
        else:
            raise ValueError("Invalid plot_mode. Expected 'all_normalized', 'all_original', 'test_normalized', or 'test_original'.")
        if self.features == 'ms':
            forecasts = forecasts[:, h-1, self.target_feature]
            actual = actual.iloc[:, self.target_feature].values
        elif self.features == 'm':
            forecasts = forecasts[:, h-1, self.target_feature]
            actual = actual.iloc[:, self.target_feature].values
        elif self.features == 's':
            forecasts = forecasts[:, h-1, :]
            actual = actual.iloc[:,self.target_feature].values

        if self.forecast_type == 'point':
            plt.subplots(figsize=(4, 2.5))
            yp = forecasts
            ##ef4470 #26abe3
            plt.plot(idx, actual, color='black', linewidth=0.5, marker='o', markersize=1, label='Observed y')
            plt.plot(idx[-yp.shape[0]:], yp, color = 'red', marker = 'o', markersize = 1, linewidth=0.5,
                     label=str('Predicted y h = ' + str(h)))
                        
            if self.transformation is None or 'original' in self.plot_mode:
                ylabel = f"Original Scale"
            elif self.transformation == 'z-score' and 'normalized' in self.plot_mode:
                ylabel = f"Normalized Scale"
            elif (self.transformation == 'log1p' or self.transformation == 'log_z-score') and 'normalized' in self.plot_mode:
                ylabel = f"Transformed Scale"
    
            plt.ylabel(ylabel)
            plt.xlabel("Time")
            if show_title:
                plt.title(
                    'Model: ' + self.model_name + ' Validation: ' + self.validation + '\nDataset: ' + self.dataset + ' Target: ' + self.target + ' Forecast type: ' + self.forecast_type.capitalize())
            if show_grid:
                plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
            else:
                plt.grid(False)
            if legend_x_axis:
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=legend_frameon)
            else:
                plt.legend(frameon=legend_frameon)
            file_path = os.path.join(self.folder_path_plots,
                                     str(self.validation) + '_' + str(self.model_name) + '_' + str(
                                         self.pred_len) + '_' + str(h + 1) + '_' + self.args['plot_mode'] + '_' + self.args['target'] + '.pdf')
            plt.savefig(file_path, bbox_inches='tight')
            plt.ticklabel_format(style='plain', axis='y')
            plt.show()
            plt.close()

        elif self.forecast_type == 'interval':
            print('to be implemented.')

