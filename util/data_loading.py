"""
__author__ = "Mohammed Aldosari"
__date__ = 2/22/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from util.timefeatures import time_features
import warnings
from torch.utils.data import DataLoader
from statsmodels.tsa.ar_model import AutoReg
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


def load_data(self, data_file: str, columns=None, skip: int = 0, sort: bool = False, date: str = 'date',
              main_output: str = 'main_output') -> pd.DataFrame:
    """
    A function used for loading the data file and selecting features for training

    Arguments
    ----------
    data_file: str
        the name of the dataset
    columns: list[str]
        columns used for training in the multivariate setting
    skip: int
        ignore the first skip rows
    date: str
        the name of the date/time column
    main_output: str
        the name of the main output column for evaluation e.g. new_deaths for the COVID dataset

    Returned Values
    ----------
    data : pd.DataFrame

    """
    data = pd.read_csv(data_file, on_bad_lines='skip')
    data[date] = pd.to_datetime(data[date])  # convert string to datetime
    data[date] = [d.date() for d in data[date]]  # convert datetime to date
    data = data.iloc[skip:]  # keep index location skip to end
    data.reset_index(inplace=True, drop=True)
    if sort:
        data = data.sort_values(by=date)  # sort by date just to make sure
    if columns is None:
        columns = ['date', self.args.target]
    else:
        columns = columns
    data = data[columns]  # keep the column you want
    return data

class Dataset(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 visualize=False, startIndex = 0,
                 validation = 'In Sample', use_original_data = True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.visualize = visualize
        self.args = args
        self.startIndex = startIndex
        self.seq_len = args['seq_len']
        self.pred_len = args['pred_len']
        self.validation = validation
        self.use_original_data = use_original_data
        self.__read_data__()

    def __read_data__(self):
        self.scaler_train = StandardScaler()
        self.scaler_val = StandardScaler()
        self.scaler_test = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        self.column  = df_raw.columns
        self.columns = df_raw.columns
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        self.start_date = df_raw['date'].dt.date.iloc[0]
        self.end_date = df_raw['date'].dt.date.iloc[-1]
        if self.args['in_sample']:
            num_train = int(len(df_raw) * 1.0)
        else:
            num_train = int(len(df_raw) * 0.8)
        num_test = len(df_raw) - num_train
        self.num_train = num_train
        self.num_test = num_test
        num_vali = len(df_raw) - num_train - num_test

        if self.pred_len > 1:
            back_steps = self.seq_len + self.pred_len - 1
            forecast_steps = self.pred_len - 1
        else:
            back_steps = 0
            forecast_steps = None
            
        if self.pred_len > 1:
            if 'Expanding'.lower() in self.args['validation'].lower():
                border1s = [0,
                        num_train - self.seq_len + (self.startIndex),
                        len(df_raw) - num_test - back_steps + (self.startIndex)]
            else:
                border1s = [0 + (self.startIndex),
                        num_train - self.seq_len + (self.startIndex),
                        len(df_raw) - num_test - back_steps + (self.startIndex)]
        else:
            border1s = [0 + (self.startIndex),
            num_train - self.seq_len + (self.startIndex),
            len(df_raw) - num_test + (self.startIndex) - self.seq_len]

        border2s = [num_train + (self.startIndex),
                    num_train + num_vali + (self.startIndex),
                    len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.test_start = border2s[1]
        self.test_end = border2s[2]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        def augment_forecasting(data, lags, forecast_steps):
            model = AutoReg(data, lags=lags)
            model_fit = model.fit()
            values = model_fit.predict(start=len(data), end=len(data) + forecast_steps - 1, dynamic = True)
            return values

        train_data_original = df_data[border1s[0]:border2s[0]].reset_index(drop=True)
        if self.pred_len > 1:
            lags = 3
            forecasted_df = pd.DataFrame()
            for column in train_data_original.columns:
                forecasted_values = augment_forecasting(train_data_original[column], lags=lags, forecast_steps=forecast_steps)
                forecasted_df[column] = forecasted_values
            train_data_original = pd.concat([train_data_original, forecasted_df])
            train_data_original = train_data_original.reset_index(drop=True)
        if self.args['model_name'] == 'Last' or self.args['model_name'] == 'Mean':
            tr_data_stats = df_data[border1s[0]:border2s[1]-1]
            self.scaler_train.fit(tr_data_stats.values)
        else:
            if self.pred_len > 1:
                self.scaler_train.fit(train_data_original.iloc[back_steps:-forecast_steps].values)
            else:
                self.scaler_train.fit(train_data_original.values)
        if self.use_original_data:
            train_data = (train_data_original.values)
        else:
            train_data = self.scaler_train.transform(train_data_original.values)
        train_data_original = train_data_original.values        
        
        val_data_original = df_data[border1s[1]:border2s[1]]
        if self.use_original_data:
            val_data = (val_data_original.values)
        else:
            val_data = self.scaler_train.transform(val_data_original.values)
        val_data_original = val_data_original.values
        
        test_data_original = df_data[border1s[2]:border2s[2]]
        if self.pred_len > 1:
            num_zeros_to_add_test = forecast_steps
            zeros_df = pd.DataFrame(0, index=range(num_zeros_to_add_test), columns=test_data_original.columns)
            test_data_original = pd.concat([test_data_original, zeros_df], ignore_index=True)
            test_data_original = test_data_original.reset_index(drop=True)
        if self.use_original_data:
            test_data = (test_data_original.values)
        else:
            test_data = self.scaler_train.transform(test_data_original.values)
        test_data_original = test_data_original.values

        if self.visualize:
            plt.subplots(figsize = (7, 4))    
            plt.plot(df_data, color='black', label='Observed Data', marker = 'o', markersize = 1, linewidth = 0.5)
            plt.xlabel("Time")
            plt.ylabel("Original Scale")
            plt.title(self.args['validation'] + '\n''Model: '+self.args['model_name'] + ' Dataset: ' + self.args['dataset'] + ' Target: ' + self.args['target'])
            
        df_stamp = df_raw[['date']][border1:border2]
        if self.pred_len > 1:
            frequency = pd.infer_freq(df_stamp['date'])
            end_date = df_stamp['date'].iloc[-1]
            forecast_dates = pd.date_range(start=end_date + pd.Timedelta(weeks=1), periods=forecast_steps, freq=frequency)
            forecast_df = pd.DataFrame({'date': forecast_dates})
            df_stamp = pd.concat([df_stamp, forecast_df])
            df_stamp = df_stamp.reset_index(drop=True)
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.set_type == 0:
            self.data_x = train_data
            self.data_x_original = train_data_original
            self.data_y = train_data
            self.data_y_original = train_data_original
            self.data_stamp = data_stamp
        elif self.set_type == 1:
            self.data_x = val_data
            self.data_x_original = val_data_original
            self.data_y = val_data
            self.data_y_original = val_data_original
            self.data_stamp = data_stamp
        elif self.set_type == 2: 
            self.data_x = test_data
            self.data_x_original = test_data_original
            self.data_y = test_data
            self.data_y_original = test_data_original
            self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_x_original = self.data_x_original[r_begin:r_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_y_original = self.data_y_original[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_x_original, seq_y, seq_y_original, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x_original) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def data_provider(args, flag, visualize = False, startIndex = 0, validation = 'Train Test', use_original_data = False):
    timeenc = 0 if args['embed'] != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args['batch_size']
        freq = args['freq']
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args['batch_size']
        freq = args['freq']

    data_set = Dataset(
        args,
        root_path = args['root_path'],
        data_path = args['data_path'],
        flag = flag,
        size = [args['seq_len'], args['label_len'], args['pred_len']],
        features = args['features'],
        target = args['target'],
        timeenc = timeenc,
        freq = freq,
        visualize = visualize,
        startIndex = startIndex,
        validation = validation,
        use_original_data = use_original_data
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args['num_workers'],
        drop_last=drop_last
    )
    return data_set, data_loader
