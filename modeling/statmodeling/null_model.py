"""
__author__ = "Mohammed Aldosari"
__date__ = 2/22/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

from typing import Tuple
import time
import numpy as np
from util.QoF import diagnose
from util.transformations import inverse_transformation
from modeling.statmodeling.model import Model
from util.plotting import plot_forecasts

class NullModel(Model):
    def in_sample_validation(self) -> Tuple[np.array, np.array]:
        super().in_sample_validation()

        self.train_data_mean = self.data.iloc[:, 1:].mean()
        start_time = time.time()
        for i in range(self.df_raw_len - self.args.seq_len):
            self.forecasts[i, :, :] = self.train_data_mean

        forecasts_original = self.forecasts

        if self.args.inverse_transform:
            forecasts_original = inverse_transformation(self.forecasts,
                                                        self.scalers,
                                                        'standard_scaler')

        end_time = time.time()
        total_time = end_time - start_time

        plot_forecasts(self, forecasts_original)
        print(f"In-sample QoF for {self.args.target}")
        diagnose(self, self.forecasts, forecasts_original)
        print(f"Total Time:{total_time} seconds. \n")

        return self.forecasts, forecasts_original

    def rolling_validation(self) -> Tuple[np.array, np.array]:
        super().rolling_validation()

        self.train_data_mean = self.data.iloc[0:self.train_size - self.max_horizon, 1:].mean()
        start_time = time.time()
        for i in range(len(self.test_data) + self.max_horizon):
            self.forecasts[i, :, :] = self.train_data_mean

        forecasts_original = self.forecasts

        end_time = time.time()
        total_time = end_time - start_time

        if self.args.inverse_transform:
            forecasts_original = inverse_transformation(self.forecasts,
                                                        self.scalers,
                                                        'standard_scaler')

        plot_forecasts(self, forecasts_original)
        print(f"Out of Sample {len(self.test_data)} QoF for {self.args.target}")
        diagnose(self, self.forecasts, forecasts_original)
        print(f"Total Time:{total_time} seconds. \n")

        return self.forecasts, forecasts_original
