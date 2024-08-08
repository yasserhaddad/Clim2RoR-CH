import re
from datetime import datetime, timedelta

import pandas as pd


def detect_date_in_filename(filename):
    """
    Detect a date pattern in a filename using regular expressions.

    Parameters:
        filename (str): The filename to be checked.

    Returns:
        str or None: The detected date string if found, or None if not found.
    """
    date_pattern = r"\d{4}\d{2}\d{2}"  # Pattern for YYYY-MM-DD format
    match = re.search(date_pattern, filename)

    if match:
        return match.group()
    else:
        return None


def generate_day_of_year_timeseries(year):
    """
    Generate a time series with days of the year (without the year) in datetime format.

    Parameters:
        year (int): The year for which to generate the time series.

    Returns:
        pandas.DatetimeIndex: A pandas DatetimeIndex representing the time series.
    """
    start_date = datetime(year, 1, 1)
    end_date = datetime(year + 1, 1, 1)  # End on the next year to include the last day

    days = [
        (start_date + timedelta(days=i)).date()
        for i in range((end_date - start_date).days)
    ]
    timeseries = pd.DatetimeIndex(days).to_period("D").strftime("%m-%d")

    return timeseries
