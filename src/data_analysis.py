import json
import os
import sys
from pathlib import Path

import dask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.append("../..")

sns.set_style("whitegrid", {"grid.color": ".93"})
os.environ["USE_PYGEOS"] = "0"

def compute_yearly_sum_from_dataset(ds, value_name="gen_yearly"):
    """Compute yearly totals from a hydropower xarray.Dataset `ds` with data var `gen`.
    Returns (yearly_da, df, series, values) where:
      - yearly_da: xarray.DataArray of yearly totals (time dimension)
      - df: pandas.DataFrame with columns ['time', value_name, 'year']
      - series: pandas.Series indexed by year with the yearly totals
      - values: numpy array of series values
    """
    yearly_gen = ds.gen.resample(time="YE").sum(dim="time").sum(dim="hydropower")
    yearly_da = yearly_gen
    df = yearly_da.to_dataframe(name=value_name).reset_index()
    df["year"] = df["time"].dt.year
    series = df.set_index("year")[value_name]
    values = series.to_numpy()
    return yearly_da, df, series, values

def compute_monthly_generation_from_ds(ds_gen, gdf_hydropower_locations, start_year=None, end_year=None, resample_rule="ME"):
    """
    For each year in ds_gen, sum monthly generation over hydropower plants that were
    already in operation that year according to gdf_hydropower_locations.
    Returns an xarray.Dataset concatenated over time with generation aggregated.
    """
    list_ds = []
    hp_all = ds_gen.hydropower.to_numpy()
    years = np.unique(ds_gen.time.dt.year.values)
    if start_year is not None:
        years = years[years >= int(start_year)]
    if end_year is not None:
        years = years[years <= int(end_year)]

    for y in years:
        wasta = gdf_hydropower_locations[
            (gdf_hydropower_locations["BeginningOfOperation"] <= int(y))
            & (gdf_hydropower_locations["WASTANumber"].isin(hp_all))
        ]["WASTANumber"].tolist()
        if not wasta:
            # skip years with no plants in operation (keeps function simple)
            continue
        ds_year = ds_gen.sel(hydropower=wasta, time=str(int(y)))
        ds_monthly = ds_year.resample(time=resample_rule).sum(dim=["hydropower", "time"])
        list_ds.append(ds_monthly)

    if not list_ds:
        raise ValueError("No monthly generation data produced (no matching plants/years).")

    return xr.concat(list_ds, dim="time").compute()