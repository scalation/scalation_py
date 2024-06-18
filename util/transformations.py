"""
__author__ = "Mohammed Aldosari"
__date__ = 2/22/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""
data_transforms: 
    A module for transforming data

Functions
----------
data_transform_std: 
    def data_transform_std(df: pd.DataFrame, test_ratio: float = 0.7):
    
data_transform_minmax
    def data_transform_minmax(df: pd.DataFrame, test_ratio: float = 0.7, min_: float = 0, max_: float = 1):
"""
def data_transform_std(self, df: pd.DataFrame):
    
    """
    A function used for data transformation to make sure it's 
    in the sensitive active region of the activation function
    by substracting the mean and dividing by the standard deviation

    Arguments
    ----------
    data : pd.DataFrame
        the input data
    test_ratio : int
        the fraction of data for testing purposes
        
    Returned Values
    ----------
    scaled_mean_std : dict

    """ 
    scalers = {}
    for i in range(0, len(df.columns)):
        if (i == 0):
            continue
        scaled = StandardScaler()
        scaled.fit(df.iloc[0:self.train_size, i].values.reshape(-1, 1))
        transformed = scaled.transform(df.iloc[:, i].values.reshape(-1, 1))
        scalers['scaled' + df.columns[i]] = scaled
        df.iloc[:, i] = pd.DataFrame(transformed)
    for idx, (scaler_name, scaler) in enumerate(scalers.items()):
        mean = scaler.mean_
        std = scaler.scale_
        print("Train Mean: ", "{:.3f}".format(mean[0]), "Train Standard Deviation: ", "{:.3f}".format(std[0]))
    return scalers, df

def data_transform_minmax(df: pd.DataFrame, train_size: float, min_: float = 0, max_: float = 1):
    
    """
    A function used for data transformation to make sure it's 
    in the sensitive active region of the activation function
    by rescaling the data to be between min_ and max_

    Arguments
    ----------
    data : pd.DataFrame
        the input data
    test_ratio : int
        the fraction of data for testing purposes
        
    Returned Values
    ----------
    scaled_mean_std : dict

    """ 
    scalers={}
    for i in range(0, len(df.columns)):
        if (i == 0):
            continue
        scaler = MinMaxScaler(feature_range=(min_, max_))
        scaler.fit(df.iloc[0:train_size, i].values.reshape(-1, 1))
        rescaled = scaler.transform(df.iloc[:, i].values.reshape(-1, 1))
        scalers['scaler_' + df.columns[i]] = scaler
        df.iloc[:, i] = pd.DataFrame(rescaled)
    return scalers, df 

def inverse_transformation(forecasts, scalers, normalization_type):
    if normalization_type == 'min_max':
        mins = np.zeros(len(scalers))
        maxs = np.zeros(len(scalers))
        for idx, (scaler_name, scaler) in enumerate(scalers.items()):
            mean = scaler.data_min_
            std = scaler.data_max_
            mins[idx] = mean
            maxs[idx] = std
        means = mins.reshape(1, -1, 1)
        stds = max.reshape(1, -1, 1)
        forecasts = forecasts * (maxs - mins) + mins
    elif normalization_type == 'standard_scaler':
        means = np.zeros(len(scalers))
        stds = np.zeros(len(scalers))
        for idx, (scaler_name, scaler) in enumerate(scalers.items()):
            mean = scaler.mean_
            std = scaler.scale_
            means[idx] = mean
            stds[idx] = std
        means = means.reshape(1, -1, 1)
        stds = stds.reshape(1, -1, 1)
        forecasts = forecasts * stds + means
    return forecasts


"""
for future experiments.
#sqrt transformation
print('type self data', type(self.data))
plt.plot(self.data[['new_deaths']], color='black', marker='o', linewidth=0.5, markersize=1, label='before')
plt.show()

self.data[['new_deaths']] = np.sqrt(self.data[['new_deaths']].values)
plt.plot(self.data[['new_deaths']], color='black', marker='o', linewidth=0.5, markersize=1, label='after')
plt.show()
forecasts ** 2

#log transformation
print('type self data', type(self.data))
plt.plot(self.data[['new_deaths']], color='black', marker='o', linewidth=0.5, markersize=1, label='before')
plt.show()

self.data[['new_deaths']] = np.sqrt(self.data[['new_deaths']].values)
plt.plot(self.data[['new_deaths']], color='black', marker='o', linewidth=0.5, markersize=1, label='after')
plt.show()
forecasts ** 2
"""