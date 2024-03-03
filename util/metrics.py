"""
__author__ = "Mohammed Aldosari"
__date__ = 2/22/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import numpy as np
import pandas as pd

"""
metrics: 
    A module for evaulating the quality/goodness of fit
    Supports both symmetric and asymmetric metrics

Functions
----------
smape: 
    def smape(y: np.ndarray, y_pred: np.ndarray) -> float:

mae:
    def mae(y: np.ndarray, y_pred: np.ndarray) -> float:

sst:
    def sst(y: np.ndarray) - > float:

sse:
    def sse(y: np.ndarray, y_pred: np.ndarray) -> float:

r2:
    def r2(y: np.ndarray, y_pred: np.ndarray) -> float:

mse:
    def mse(y: np.ndarray, y_pred: np.ndarray) -> float:

rmse:
    def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:

rae:
    def rae(y: np.ndarray, y_pred: np.ndarray) -> float:
"""


def smape(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    A function to calculate the symmetric mean absolute percentage error (sMAPE).

    Arguments
    ----------
    y: np.ndarray
        the response data
    y_pred: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    sMAPE : float

    """
    return 200 * np.sum(np.abs(y - y_pred) / (np.abs(y) + np.abs(y_pred))) / len(y)


def mae(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    A function to calculate the mean absolute error (MAE).

    Arguments
    ----------
    y: np.ndarray
        the response data
    y_pred: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    MAE : float

    """
    return np.sum(np.abs(y - y_pred)) / len(y)


def sst(y: np.ndarray) -> float:
    """
    A function to calculate the sum of squares total (SST).

    Arguments
    ----------
    y: np.ndarray
        the response data

    Returned Values
    ----------
    SST : float
    """
    return np.sum((y - np.mean(y)) ** 2)


def sse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    A function to calculate the sum of squared errors (SSE).

    Arguments
    ----------
    y: np.ndarray
        the response data
    y_pred: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    SSE : float

    """
    return np.sum((y - y_pred) ** 2)


def r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    A function to calculate R squared (r2).

    Arguments
    ----------
    y: np.ndarray
        the response data
    y_pred: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    r2 : float

    """
    return 1 - sse(y, y_pred) / sst(y)


def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    A function to calculate mean squared error (MSE).

    Arguments
    ----------
    y: np.ndarray
        the response data
    y_pred: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    MSE : float

    """
    return sse(y, y_pred) / len(y)


def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    A function to calculate root mean squared error (RMSE).

    Arguments
    ----------
    y: np.ndarray
        the response data
    y_pred: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    RMSE : float

    """
    return np.sqrt(mse(y, y_pred))


def rae(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    A function to calculate relative absolute error (RAE).

    Arguments
    ----------
    y: np.ndarray
        the response data
    y_pred: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    RAE : float

    """
    y_mean = np.mean(y)
    squared_error_num = np.sum(np.abs(y - y_pred))
    squared_error_den = np.sum(np.abs(y - y_mean))
    rae_loss = squared_error_num / squared_error_den

    return rae_loss

def get_metrics(actual, forecasts, args):
    print("Out of Sample Test set metrics {}".format(args.MO))
    total_mse, total_mae, total_smape, total_rmse = 0, 0, 0, 0
    if args.last_only:
        y = np.expand_dims(actual[:, i], axis=-1)
        y_pred = np.expand_dims(forecasts[:, i], axis=-1)
        mse_, mae_, smape_ = 0, 0, 0
        mse_ = mse(y, y_pred)
        mae_ = mae(y, y_pred)
        smape_ = smape(y, y_pred)
        print('h:{}, N:{}, MSE: {}, MAE: {}, sMAPE: {}'.format(args.vis_h, len(actual), mse_, mae_, smape_))
    else:
        for i in range(args.horizon):
            y = np.expand_dims(actual[:, i], axis=-1)
            y_pred = np.expand_dims(forecasts[:, i], axis=-1)
            mse_, mae_, smape_ = 0, 0, 0
            mse_ = mse(y, y_pred)
            mae_ = mae(y, y_pred)
            smape_ = smape(y, y_pred)
            rmse_ = rmse(y, y_pred)
            total_mse += mse_
            total_mae += mae_
            total_smape += smape_
            total_rmse += rmse_
            print('h:{}, N:{}, MSE:{}, MAE:{}, sMAPE:{}, RMSE:{}'.format(i, len(actual), mse_, mae_, smape_, rmse_))
        avg_mse = total_mse / args.horizon
        avg_mae = total_mae / args.horizon
        avg_smape = total_smape / args.horizon
        avg_rmse = total_rmse / args.horizon
        print('Average MSE: {}, Average MAE: {}, Average sMAPE: {}, Average RMSE: {}'.format(avg_mse, avg_mae, avg_smape, avg_rmse))