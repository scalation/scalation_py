"""
__author__ = "Mohammed Aldosari"
__date__ = 2/24/24
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
from modeling.statmodeling.reecursive_forecasting import recursive_forecast_arima
from statsmodels.tsa.arima.model import ARIMA as ARIMA_

class ARIMA(Model):
    def in_sample_validation(self) -> Tuple[np.array, np.array]:
        super().in_sample_validation()

        start_time = time.time()
        for i in range(len(self.data.columns) - 1):
            data_temp = self.data.iloc[:, i + 1]
            model_arima = ARIMA_(data_temp, order=(self.args.p,
                                                   self.args.d,
                                                   self.args.q), trend=self.args.trend).fit()
            for j in range(self.df_raw_len - self.args.seq_len):
                predictions = model_arima.predict(start=j + self.args.seq_len,
                                                  end=j + self.args.seq_len + self.max_horizon,
                                                  dynamic = True)
                self.forecasts[j, i, :] = predictions.values.T.reshape(1, self.args.max_horizon)

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

    def train_test(self) -> Tuple[np.array, np.array]:
        super().train_test()

        start_time = time.time()
        for i in range(len(self.train_data.columns) - 1):
            data_temp = self.data.iloc[0:self.train_size - self.max_horizon, i + 1]
            model_arima = ARIMA_(data_temp, order=(self.args.p,
                                                   self.args.d,
                                                   self.args.q),
                                 trend=self.args.trend).fit()
            for j in range(len(self.test_data) + self.max_horizon):
                data_temp = self.data.iloc[0:self.train_size + j - self.max_horizon, i + 1]
                next_observed = self.data.iloc[self.train_size + j - self.max_horizon:self.train_size + j + 1, i + 1]
                fitted_params = model_arima.params
                predictions = recursive_forecast_arima(self,
                                                 data_temp.values,
                                                 next_observed,
                                                 fitted_params,
                                                 model_arima.resid,
                                                 self.args.max_horizon)
                self.forecasts[j, i, :] = predictions.T.reshape(1, self.args.max_horizon)
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

    def rolling_validation(self) -> Tuple[np.array, np.array]:
        super().rolling_validation()

        start_time = time.time()
        for i in range(len(self.train_data.columns) - 1):
            for j in range(len(self.test_data) + self.max_horizon):
                if self.args.rolling:
                    data_temp = self.data.iloc[j:self.train_size + j - self.max_horizon, i + 1]
                else:
                    data_temp = self.data.iloc[0:self.train_size + j - self.max_horizon, i + 1]
                data_temp = data_temp.reset_index(drop=True)
                model_arima = ARIMA_(data_temp, order=(self.args.p,
                                                       self.args.d,
                                                       self.args.q),
                                     trend=self.args.trend).fit()
                predictions = model_arima.predict(start=data_temp.shape[0],
                                                  end=data_temp.shape[0] + self.max_horizon,
                                                  dynamic = True)
                self.forecasts[j, i, :] = predictions.values.T.reshape(1, self.args.max_horizon)
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