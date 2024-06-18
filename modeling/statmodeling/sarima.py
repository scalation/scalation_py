"""
__author__ = "Mohammed Aldosari"
__date__ = 2/25/24
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
from modeling.statmodeling.reecursive_forecasting import recursive_forecast_sarima
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_
import warnings
warnings.filterwarnings("ignore")

class SARIMA(Model):
    def in_sample_validation(self) -> Tuple[np.array, np.array]:
        super().in_sample_validation()

        start_time = time.time()
        for i in range(len(self.data.columns) - 1):
            data_temp = self.data.iloc[:, i + 1]
            model_sarima = SARIMAX_(data_temp,
                                        order=(self.args.p, self.args.d, self.args.q),
                                        seasonal_order=(self.args.P,
                                                        self.args.D,
                                                        self.args.Q,
                                                        self.args.s),
                                        trend=self.args.trend,
                                        initialization='approximate_diffuse').fit(disp=False)
            for j in range(self.df_raw_len - self.args.seq_len):
                predictions = model_sarima.predict(start=j + self.args.seq_len,
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
            model_sarima = SARIMAX_(data_temp,
                                    order=(self.args.p, self.args.d, self.args.q),
                                    seasonal_order=(self.args.P,
                                                    self.args.D,
                                                    self.args.Q,
                                                    self.args.s),
                                    trend=self.args.trend,
                                    initialization='approximate_diffuse').fit(disp=False)
            for j in range(len(self.test_data) + self.max_horizon):
                data_temp = self.data.iloc[0:self.train_size + j - self.max_horizon, i + 1]
                next_observed = self.data.iloc[self.train_size + j - self.max_horizon:self.train_size + j + 1, i + 1]
                fitted_params = model_sarima.params
                predictions = recursive_forecast_sarima(self,
                                                 data_temp.values,
                                                 next_observed,
                                                 fitted_params,
                                                 model_sarima.resid,
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
                model_sarima = SARIMAX_(data_temp,
                                        order=(self.args.p, self.args.d, self.args.q),
                                        seasonal_order=(self.args.P, self.args.D, self.args.Q, self.args.s),
                                        trend=self.args.trend,
                                        initialization='approximate_diffuse').fit(disp=False)
                predictions = model_sarima.predict(start=self.train_size + j - self.max_horizon,
                                                   end=self.train_size + j, dynamic = True)
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