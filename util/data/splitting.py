"""
__author__ = "Mohammed Aldosari"
__date__ = 2/22/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import sklearn.model_selection as sk
import numpy as np
import torch
import pandas as pd
import math

"""
data_splitting: 
    A module for splitting data.

Functions
----------
train_test_split: 
    def train_test_split(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.25) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
"""

def train_test_split(data: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    tr_size = int(math.ceil(data.shape[0]*train_ratio))
    val_size = int(data.shape[0]*val_ratio)

    train_data = data.iloc[0:tr_size]
    val_data = data.iloc[tr_size:tr_size+val_size]
    test_data = data.iloc[tr_size+val_size:]
    return train_data, val_data, test_data
