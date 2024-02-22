import numpy as np
from util.data_loading import load_data, plot_train_test
from util.data_transforms import data_transform_std
from util.data_splitting import train_test_split

def MeanModel(file_name: str, training_ratio: float, horizon: int, main_output: str, normalization: bool) -> (
int, float, float, float):
    """
    A function used for producing forecasts by taking the mean of the training.

    Arguments
    ----------
    file_name: str
        the file path for csv data file.
    training_ratio: float
        the training ratio used for splitting the dataset into train and test
    horizon: int
        how many time steps ahead to make the forecasts
    main_output: str
        the main output column/feature, e.g. '% WEIGHTED ILI'
    normalization: bool
        specifies whether the data is normalized or original

    Returned Values
    ----------
    mse: float
    mae: float
    smape: float

    """
    horizon = horizon - 1
    data = load_data(file_name, main_output=main_output)
    train_size = int(training_ratio * len(data))
    if normalization:
        scaled_mean_std, data = data_transform_std(data, train_size)

    train_data, val_data, test_data = train_test_split(data,
                                                       train_ratio=training_ratio)  # No validation data for Random Walk.
    train_data_MO = train_data[[main_output]]  # Train set for main output column.
    train_data_MO_mean = train_data_MO.mean()
    test_data_MO = test_data[[main_output]]  # Test set for main output column.
    actual = np.zeros(shape=(len(test_data_MO) - horizon, horizon + 1))  # Actual complete dataset for main output.
    forecasts = np.zeros(shape=(len(test_data_MO) - horizon, horizon + 1))
    for i in range(len(test_data_MO) - horizon):
        for j in range(horizon + 1):
            actual[i, j] = float(data.iloc[train_size + i + j, :][main_output])
            forecasts[i, j] = float(train_data_MO_mean.iloc[0])
    plot_train_test(data, main_output, train_size, train_data_MO, test_data_MO, forecasts, horizon)
    return actual, forecasts