"""
__author__ = "Mohammed Aldosari"
__date__ = 6/3/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

from typing import Tuple
from numpy import ndarray
import copy
import os
import numpy as np
import pandas as pd
from util.data_loading import load_data
from util.transformations import data_transform_std
from util.data_splitting import train_test_split
from util.plotting import plot_time_series

class Model:
    def __init__(self, args):
        self.args = args # data dictionary
        self.max_horizon = self.args.max_horizon - 1 # number of additional horizons beyond 1
        self.data = self.load_data()
        self.cols = len(self.data.columns) - 1 # assuming first column is a date column. represent the number of variables

        self.df_raw_len = len(self.data) # the length of the entire data
        self.folder_path_plots = './plots/' + str(self.args.validation) + '/' + self.args.model_name + '/' + str(self.args.dataset) +'/'+str(self.args.max_horizon)
        self.folder_path_results = './results/'+ str(self.args.validation) + '/'

        print('\033[1m' + self.args.model_name + '\033[0m')
        print('\033[1mTarget:\033[0m ' + self.args.target + ' \033[1mMax Horizons:\033[0m ' + str(self.args.max_horizon))

        if not os.path.exists(self.folder_path_plots):
            os.makedirs(self.folder_path_plots)
        if not os.path.exists(self.folder_path_results):
            os.makedirs(self.folder_path_results)

        plot_time_series(self, self.args)

    def load_data(self):
        return load_data(self, self.args.file_name, main_output=self.args.main_output)

    def append_zeros(self, data):
        num_zeros = self.max_horizon
        zeros_df = pd.DataFrame(np.zeros((num_zeros, data.shape[1])), columns=data.columns)
        return pd.concat([data, zeros_df], ignore_index=True)

    def get_target_index(self):
        return self.data.columns.get_loc(self.args.target) - 1   # gets the index of the target column e.g. new_deaths assuming date is the first column

    def data_transform_std(self):
        return data_transform_std(self, self.data)

    def train_test_split(self):
        return train_test_split(self.data, train_ratio=self.args.training_ratio)

    def in_sample_validation(self) -> Tuple[np.array, np.array]:
        self.data = self.append_zeros(self.data)
        self.target_int = self.get_target_index()
        self.data_ = copy.deepcopy(self.data)
        self.train_size = self.df_raw_len - self.args.seq_len

        if self.args.normalization:
            self.scalers, self.data = self.data_transform_std()

        self.forecasts: ndarray[float] = np.zeros(
            shape=(self.train_size,
                   self.cols,
                   self.args.max_horizon))


    def train_test(self) -> Tuple[np.array, np.array]:

        self.train_size = int(self.args.training_ratio * len(self.data))
        self.target_int = self.get_target_index()

        if self.args.normalization:
            self.scalers, self.data = self.data_transform_std()
        self.train_data, _, self.test_data = self.train_test_split()
        self.start_index_test = self.test_data.index[0]
        self.data = self.append_zeros(self.data)
        self.data_ = self.append_zeros(self.data_)

        self.forecasts: ndarray[float] = np.zeros(
            shape=(len(self.test_data) + self.max_horizon,
                   self.cols,
                   self.args.max_horizon))

    def rolling_validation(self) -> Tuple[np.array, np.array]:
        self.train_size = int(self.args.training_ratio * len(self.data))
        self.target_int = self.get_target_index()
        self.data_ = copy.deepcopy(self.data)

        if self.args.normalization:
            self.scalers, self.data = self.data_transform_std()
        self.train_data, _, self.test_data = self.train_test_split()
        self.start_index_test = self.test_data.index[0]
        self.data = self.append_zeros(self.data)
        self.data_ = self.append_zeros(self.data_)

        self.forecasts: ndarray[float] = np.zeros(
            shape=(len(self.test_data) + self.max_horizon,
                   self.cols,
                   self.args.max_horizon))