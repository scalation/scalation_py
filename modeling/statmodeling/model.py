"""
__author__ = "Mohammed Aldosari"
__date__ = 11/03/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import copy
import os
import time
import numpy as np
from util.data.loading import load_data
from util.data.preprocessing import transform_
from util.data.splitting import train_test_split
from util.plotting import plot_time_series, plot_acf, plot_pacf
from util.data.preprocessing import inverse_transform
import datetime
from util.plotting import plot_forecasts
from util.QoF import diagnose
from util.tools import display_save_results
np.set_printoptions(suppress=True)

class Model:
    def __init__(self, args):
        self.args = args
        self.modeling_type = 'statistical'
        self.modeling_mode = 'joint'
        self.dataset_path = args.dataset_path
        self.skip_insample = self.args.skip_insample
        self.target = self.args.target
        self.dataset = self.args.dataset_name
        self.start_index_acf_pacf = 0
        self.diff_order_acf_pacf = 0
        self.horizons = self.args.horizons
        self.transformation = self.args.transformation
        self.rc = self.args.rc
        self.eda_lags = self.args.eda_lags
        self.qof_equal_samples = self.args.qof_equal_samples
        self.plot_eda = self.args.plot_eda
        self.training_ratio = self.args.training_ratio
        self.debugging = self.args.debugging
        self.forecast_type = self.args.forecast_type.lower() if self.args.forecast_type is not None else None
        self.plot_mode = self.args.plot_mode.lower() if self.args.plot_mode is not None else self.args.plot_mode
        self.features = self.args.features.lower() if self.args.features is not None else self.args.features
        self.qof_mode = self.args.qof_mode.lower() if self.args.qof_mode is not None else self.args.qof_mode
        self.transformation = self.args.transformation.lower() if self.args.transformation is not None else self.args.transformation
        self.modeling_mode = self.args.modeling_mode.lower() if self.args.modeling_mode is not None else self.args.modeling_mode

        if self.training_ratio <= 0 or self.training_ratio >= 100:
            raise ValueError(
                f"Invalid value for 'training_ratio'. Expected a fraction or whole number between 0 and 100. For 80% training ratio, both 0.8 and 80 should work.\n"
                f"Received: {self.training_ratio}."
            )

        self.training_ratio = self.args.training_ratio/100 if isinstance(self.args.training_ratio, int) else self.args.training_ratio


        if self.forecast_type not in ['point', 'interval']:
            raise ValueError(
                f"Invalid value for 'forecast_type'. Expected one of the following 'point' or 'interval'\n"
                f"Received: {self.forecast_type}."
            )

        if self.plot_mode not in ['all_original', 'all_transformed', 'test_original', 'test_transformed', None]:
            raise ValueError(
                f"Invalid value for 'plot_mode'. Expected one of the following 'all_original', 'all_transformed', 'test_original', 'test_transformed', or None\n"
                f"Received: {self.plot_mode}."
            )

        if self.features not in ['ms', 'm', 's']:
            raise ValueError(
                f"Invalid value for 'features'. Expected one of the following 'ms', 'm', 's'\n"
                f"Received: {self.features}."
            )

        if self.qof_mode not in ['single', 'cumulative']:
            raise ValueError(
                f"Invalid value for 'qof_mode'. Expected one of the following 'single' or 'cumulative'\n"
                f"Received: {self.qof_mode}."
            )

        if type(self.qof_equal_samples) is not bool:
            raise ValueError(
                f"Invalid value for 'qof_equal_samples'. Expected a boolean (True or False)\n"
                f"Received: {self.qof_equal_samples}."
            )

        if type(self.plot_eda) is not bool:
            raise ValueError(
                f"Invalid value for 'plot_eda'. Expected a boolean (True or False)\n"
                f"Received: {self.plot_eda}."
            )

        if type(self.horizons) is not list:
            raise ValueError(
                f"Invalid value for 'horizons'. Expected a list of target horizons.\n"
                f"Received: {self.horizons}."
            )

        if self.transformation not in ['log1p', 'z-score', 'log_z-score', None]:
            raise ValueError(
                f"Invalid value for 'transformation'. Expected one of the following 'log1p', 'z-score', or None.\n"
                f"Received: {self.transformation}."
            )

        if self.modeling_mode not in ['joint', 'individual']:
            raise ValueError(
                f"Invalid value for 'modeling_mode'. Expected 'joint' or 'individual'.\n"
                f"Received: {self.modeling_mode}."
            )

        if self.horizons != sorted(self.horizons):
            self.horizons.sort()

        if self.skip_insample is None:
            self.validation = 'Out-of-Sample'
        else:
            self.validation = 'In-Sample'

        self.pred_len = max(self.horizons)

        load_data(self)
        self.columns = self.data.columns
        self.data.columns = self.data.columns.str.lower()
        if self.target.lower() not in self.data.columns.to_list():
            raise ValueError(
                f"Invalid value for 'target'. Expected one of the following: {self.data.columns.to_list()}.\n"
                f"Received: {self.target}."
            )

        if self.features == 's':
            self.data = self.data[[self.target.lower()]]

        self.target_feature = self.data.columns.get_loc(self.target.lower())
        self.n_features = len(self.data.columns)

        self.qof = None
        self.today = str(datetime.datetime.today().strftime('%Y-%m-%d'))

    def trainNtest(self) -> np.array:

        self.df_raw_len = len(self.data)
        self.folder_path_plots = './plots/' + str(self.validation) + '/' + self.model_name + '/' + str(
            self.dataset) + '/' + str(self.pred_len)
        self.folder_path_results = './results/' + str(self.validation) + '/'

        if not os.path.exists(self.folder_path_plots):
            os.makedirs(self.folder_path_plots)
        if not os.path.exists(self.folder_path_results):
            os.makedirs(self.folder_path_results)

        self.data_ = copy.deepcopy(self.data)

        if self.transformation is not None:
            self.data = transform_(self)

        if self.skip_insample is None:
            self.train_data, _, self.test_data = train_test_split(self.data, train_ratio=self.training_ratio)
            self.train_size = len(self.train_data)
            self.test_size = len(self.test_data)
        else:
            self.train_size = self.skip_insample
            self.test_size = len(self.data) - self.skip_insample


        if self.skip_insample is None:
            self.sample_mean = self.data_.iloc[:self.train_size,:].mean().to_frame().T
            self.sample_mean_normalized = self.data.iloc[:self.train_size,:].mean().to_frame().T
        else:
            self.sample_mean = self.data_.iloc[self.skip_insample:,:].mean().to_frame().T
            self.sample_mean_normalized = self.data.iloc[self.skip_insample:,:].mean().to_frame().T

        if self.features == 'm':
            self.sample_mean = self.sample_mean
            self.sample_mean_normalized = self.sample_mean_normalized
        elif self.features == 'ms':
            self.sample_mean = self.sample_mean.iloc[:, self.target_feature]
            self.sample_mean_normalized = self.sample_mean_normalized.iloc[:, self.target_feature]
        elif self.features == 's':
            self.sample_mean = self.sample_mean
            self.sample_mean_normalized = self.sample_mean_normalized

        if self.rc is not None:
            if self.skip_insample is not None and self.rc < self.test_size and self.rc is not None:
                raise ValueError(
                    f"Invalid value for 'rc'. Expected a number greater that the test set size ({self.test_size}).\n"
                    f"For in-sample validation, the model will be fitted once on the entire dataset. Retraining cycle is not required.\n"
                    f"Received: {self.rc}."
                )

        if self.plot_eda:
            plot_time_series(self)
            plot_acf(self)
            plot_pacf(self)

        start_time = time.time()

        if self.modeling_mode == 'joint':
            self.train_test()
            self.forecast_tensor_original = inverse_transform(self)
            if self.plot_mode is not None:
                plot_forecasts(self)
            if self.args.internal_diagnose:
                self.args.internal_diagnose = True
                mae_normalized_list, mae_original_list = diagnose(self)
            else:
                diagnose(self)

        elif self.modeling_mode == 'individual':
            for h in self.args.horizons:

                self.horizons = [h]
                self.pred_len = h
                self.train_test()
                self.forecast_tensor_original = inverse_transform(self)
                if self.plot_mode is not None:
                    plot_forecasts(self)
                if self.args.internal_diagnose:
                    self.args.internal_diagnose = True
                    mae_normalized_list, mae_original_list = diagnose(self)
                else:
                    diagnose(self)
                self.args.mase_calc = None
        if self.args.internal_diagnose is None:
            if self.debugging:
                print(self.args)
            display_save_results(self)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time:{total_time} seconds. \n")
        if self.args.internal_diagnose:
            return mae_normalized_list, mae_original_list
        else:
            return self.forecast_tensor_original