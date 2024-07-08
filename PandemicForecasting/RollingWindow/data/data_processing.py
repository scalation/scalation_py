import pandas as pd


def data_processing(data_file: str, columns, skip: int = 0, date: str = 'date') -> pd.DataFrame:
    """
    A function used for data processing and visualization

    Arguments
    ----------
    data_file : str
        the file name
    columns : list[str]
        the name of the variable
    skip : int
        ignore the first skip rows
    date : str
        The date to be added as column

    Returned Values
    ----------
    data : pd.DataFrame

    """
    data = pd.read_csv(data_file)
    data[date] = pd.to_datetime(data[date])  # convert string to datetime
    data[date] = [d.date() for d in data[date]]  # convert datetime to date
    data = data.iloc[skip:]  # keep index location skip to end
    data.reset_index(inplace=True, drop=True)
    data = data.sort_values(by=date)  # sort by date just to make sure
    data = data[columns]  # keep the column you want
    observed = data[['date', 'new_deaths']]
    return data, observed
