"""
__author__ = "Mohammed Aldosari"
__date__ = 2/22/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import pandas as pd
import numpy as np


def load_data(data_file: str, columns=None, skip: int = 0, sort: bool = False, date: str = 'date',
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
        columns = data.columns
    data = data[columns]  # keep the column you want
    return data



