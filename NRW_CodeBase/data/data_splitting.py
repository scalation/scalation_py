import sklearn.model_selection as sk
import numpy as np
import torch
import pandas as pd

"""
data_splitting: 
    A module for splitting data.

Functions
----------
train_test_split: 
    def train_test_split(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.25) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_test_split(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.25) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    
    """ 
    A function used for splitting the data into fixed train and test sets.
    Usually, for time series, it is better to use a rolling window.
    For the train/test split, we use the default split ratio by scikit-learn 
    set to 0.25.

    Arguments
    ----------
    x : pd.DataFrame
        past lagged data
        
    y : pd.DataFrame
        future horizons
        
    test_ratio : int
        the fraction of data for testing purposes
        
    Returned Values
    ----------
    x_train : numpy.ndarray
    x_test : numpy.ndarray
    y_train : numpy.ndarray
    y_test : numpy.ndarray
    
    """ 
    x_train, x_test, y_train, y_test = sk.train_test_split(x, y, test_size = test_ratio, 
                                                           random_state = 42, shuffle = False)
    return x_train, x_test, y_train, y_test


def make_input_output_sequences(series:np.ndarray, n_past:int, n_future:int, include_dates:bool):
    """ 
    A function used for making input and output sequence

    Arguments
    ----------
    series: np.ndarray
            input data values

    n_past: int
            number of past data 
           
    n_future: int
              number of future data (target)
    
    include_dates: bool
                   whether to include date column or not (default true)
        
    Returned Values
    ----------
    input sequence x and output sequence y

    """ 
    X, y = list(), list()
    forecast, target = list(), list()
    for window_start in range(n_past,len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end-n_past > len(series):
            break
        past = series[(window_start-n_past):(past_end-n_past), :]
        future = series[(past_end-n_past):(future_end-n_past), :]
        X.append(np.array([past]).T.tolist())
        y.append(np.array([future]).T.tolist())
        x = np.array(X)
    x = np.squeeze(np.swapaxes(np.array(X), 1, 2), axis=-1)
    y = np.squeeze(np.swapaxes(np.array(y), 1, 2), axis=-1)
    if include_dates is True:
        return np.array(x[:,:,:]), np.array(y[:,:,:])
    else:
        return torch.from_numpy(np.array(x[:,:,1:],dtype=np.float32)).float().to(device), torch.from_numpy(np.array(y[:,:,1:],dtype=np.float32)).float().to(device)

def shift_sequence(x_train:np.ndarray , y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray, window:int, include_dates:bool):
    """ 
    A function to shift input and output sequence by window size. 

    Arguments
    ----------
    x_train : np.ndarray
              train data - input values

    y_train : np.ndarray
              train data - target values
              
    x_test : np.ndarray
              test data - input values

    y_test : np.ndarray
              test data - target values
        
    Returned Values
    ----------
    x_train : numpy.ndarray or tensor
    x_test : numpy.ndarray or tensor
    y_train : numpy.ndarray or tensor
    y_test : numpy.ndarray or tensor

    """
    x_train = x_train[window:,:,:]
    x_train_list = x_train.tolist()
    for i in range (0,window):
        x_train_list.append(x_test[i,:,:].tolist())
    x_train = np.array(x_train_list)
    y_train = y_train[window:,:,:]
    y_train_list = y_train.tolist()
    for i in range(0,window):
        y_train_list.append(y_test[i,:,:].tolist())
    y_train = np.array(y_train_list)
    x_test = x_test[window:,:,:]
    y_test = y_test[window:,:,:]

    if include_dates is True:
        return x_train, x_test, y_train, y_test
    else:
        return torch.from_numpy(x_train).to(device).float(), x_test.to(device).float(), torch.from_numpy(y_train).to(device).float(), y_test.to(device).float()

# def train_test_split(data: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
#     tr_size = int(data.shape[0]*train_ratio)
#     val_size = int(data.shape[0]*val_ratio)
    
#     train_data = data.iloc[0:tr_size]
#     val_data = data.iloc[tr_size:tr_size+val_size]
#     test_data = data.iloc[tr_size+val_size:]
#     return train_data, val_data, test_data