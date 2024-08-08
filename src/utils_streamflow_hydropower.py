import pathlib
from ast import literal_eval
from datetime import datetime, timedelta
from itertools import repeat
from multiprocessing import Pool
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression

from src.var_attributes import HYDROPOWER_NETCDF_ENCODINGS

GRAVITY = 9.81
WATER_DENSITY = 1000

def generate_day_of_year_timeseries(year: int):
    """Generate a time series with days of the year (without the year) in datetime format.

    Parameters
    ----------
    year : int
        The year for which to generate the time series.

    Returns
    -------
    pandas.DatetimeIndex
        A pandas DatetimeIndex representing the time series.
    """
    start_date = datetime(year, 1, 1)
    end_date = datetime(year + 1, 1, 1)  # End on the next year to include the last day

    days = [(start_date + timedelta(days=i)).date() for i in range((end_date - start_date).days)]
    timeseries = pd.DatetimeIndex(days).to_period('D').strftime('%m-%d')

    return timeseries


def convert_mm_d_to_cubic_m_s(value: float, area: float) -> float:
    """Convert flow rate in mm/d to m3/s

    Parameters
    ----------
    value : float
        Flow rate to convert
    area : float
        Size of the area where the flow is going through

    Returns
    -------
    float
        The converted flow rate in m3/s
    """
    return value * area / 1000 / (3600 * 24)


def get_polygon_streamflow_dataset(ds_streamflow: xr.Dataset, gdf_polygons: gpd.GeoDataFrame,
                                   df_pts_in_polygons: pd.DataFrame, polygon_ezgnr: int) -> xr.Dataset:
    """Compute streamflow in a catchment by getting all the grid points from the hydrological model
    that lie in the catchment and combine their streamflow data into one dataset.

    Parameters
    ----------
    ds_streamflow : xr.Dataset
        xarray Dataset containing the gridded streamflow data
        from the hydrological model
    gdf_polygons : gpd.GeoDataFrame
        GeoDataframe containing the catchments' locations
    df_pts_in_polygons : pd.DataFrame
        DataFrame containing a matching between each point in
        the gridded data and the catchments
    polygon_ezgnr : int
        Catchment for which to find the relevant streamflow
        gridded data

    Returns
    -------
    xr.Dataset
        xarray Dataset containing the gridded streamflow data for the catchment
    """
    df_points = df_pts_in_polygons.loc[df_pts_in_polygons["EZGNR"] == polygon_ezgnr]
    if len(df_points) > 0:
        list_ds_points = []
        for _, row in df_points.iterrows():
            list_ds_points.append(ds_streamflow.sel(y=row["y"], x=row["x"], method="nearest"))
        ds_polygon_agg = xr.concat(list_ds_points, "point")
    else:
        polygon_centroid = gdf_polygons.loc[gdf_polygons["EZGNR"] == polygon_ezgnr].centroid.iloc[0]
        ds_polygon_agg = ds_streamflow.sel(y=polygon_centroid.y, x=polygon_centroid.x, method="nearest") \
                                      .assign_coords(point=("point", [1]))

    if "x" in ds_polygon_agg.coords.keys():
        ds_polygon_agg = ds_polygon_agg.drop_vars(["x"])

    if "y" in ds_polygon_agg.coords.keys():
        ds_polygon_agg = ds_polygon_agg.drop_vars(["y"])

    return ds_polygon_agg

def compute_streamflow_aggregate_polygon(ds_streamflow: xr.Dataset, gdf_polygons: gpd.GeoDataFrame,
                                         df_pts_in_polygons: pd.DataFrame, polygon_ezgnr: int,
                                         method: str = "mean") -> xr.Dataset:
    """Compute aggregated streamflow data at a certain catchment by getting all the grid points
    from a hydrological model that lie in the catchment and aggregate all the points into a
    a single time series of streamflow data for the catchment.

    Parameters
    ----------
    ds_streamflow : xr.Dataset
        xarray Dataset containing the gridded streamflow data
        from the hydrological model
    gdf_polygons : gpd.GeoDataFrame
        GeoDataframe containing the catchments' locations
    df_pts_in_polygons : pd.DataFrame
        DataFrame containing a matching between each point in
        the gridded data and the catchments
    polygon_ezgnr : int
        Catchment for which to compute the streamflow time series
    method : str, optional
        Aggregation method (either "sum" or "mean"), by default "mean"

    Returns
    -------
    xr.Dataset
        xarray Dataset containing the streamflow timeseries for the catchment

    Raises
    ------
    ValueError
        If the method passed as argument is not "mean" or "sum"
    """
    if method not in ["mean", "sum"]:
        raise ValueError("The method should either be 'mean' or 'sum'.")

    ds_polygon_agg = get_polygon_streamflow_dataset(ds_streamflow, gdf_polygons, df_pts_in_polygons, polygon_ezgnr)
    ds_polygon_agg = ds_polygon_agg.assign_coords(polygon=("polygon", [polygon_ezgnr]))

    if method == "sum":
        return ds_polygon_agg.sum("point")
    else:
        return ds_polygon_agg.mean("point")

def compute_streamflow_aggregate_hydropower(ds_streamflow: xr.Dataset, df_hydropower_polygons: pd.DataFrame,
                                            hydropower_wasta: int, gdf_polygons: gpd.GeoDataFrame,
                                            df_pts_in_polygons: pd.DataFrame, method: str = "mean") -> xr.Dataset:
    """Compute aggregated streamflow at a hydropower plant by aggregating all the streamflow timeseries of
    the catchment area assigned to the power plant (catchment containing the water intake point and its
    upstream catchments).

    Parameters
    ----------
    ds_streamflow : xr.Dataset
        xarray Dataset containing the gridded streamflow data
        from the hydrological model
    df_hydropower_polygons : pd.DataFrame
        DataFrame matching hydropower plants to catchments
        (water intake catchments and their upstream catchments)
    hydropower_wasta : int
        The WASTA number of the hydropower plant
    gdf_polygons : gpd.GeoDataFrame
        GeoDataframe containing the catchments' locations
    df_pts_in_polygons : pd.DataFrame
        DataFrame containing a matching between each point in
        the gridded data and the catchments
    method : str, optional
        Aggregation method (either "sum" or "mean"), by default "mean"

    Returns
    -------
    xr.Dataset
        xarray Dataset containing the streamflow timeseries for the hydropower plant

    Raises
    ------
    ValueError
        If the method passed as argument is not "mean" or "sum"
    """
    if method not in ["mean", "sum"]:
        raise ValueError("The method should either be 'mean' or 'sum'.")

    relevant_row = df_hydropower_polygons.loc[df_hydropower_polygons["WASTANumber"] == hydropower_wasta]
    upstream_polygons = relevant_row["upstream_EZGNR"].apply(lambda l: [] if pd.isnull(l) else literal_eval(l)).iloc[0]
    relevant_polygons = (relevant_row["EZGNR"].to_list() + upstream_polygons)

    list_ds_polygons = []
    for polygon_ezgnr in relevant_polygons:
        list_ds_polygons.append(get_polygon_streamflow_dataset(ds_streamflow, gdf_polygons, df_pts_in_polygons, polygon_ezgnr))
    ds_polygons = xr.concat(list_ds_polygons, "point")

    if method == "sum":
        return ds_polygons.sum("point")
    else:
        return ds_polygons.mean("point")


def compute_simplified_efficiency_term(installed_capacity: float, design_discharge: float, hydraulic_head: float) -> float:
    """Compute simplified efficiency from the installed capacity, the design discharge and hydraulic head
    (assumed to be constant) of a hydropower plant.

    Parameters
    ----------
    installed_capacity : float
        The installed capacity of a power plant, at the
        generator (in W)
    design_discharge : float
        The design discharge of a power plant
        (in m^3/s)
    hydraulic_head : float
        The hydraulic head of a power plant (in m),
        assuming a constant hydraulic head

    Returns
    -------
    float
        The simplified efficiency of a hydropower plant
    """
    return (installed_capacity)/(design_discharge * hydraulic_head)


def compute_hydropower_generation_from_streamflow(streamflow: float, hydraulic_height: float,
                                                  efficiency: float = 0.8,
                                                  simplified_efficiency: float = None,
                                                  design_discharge: float = None,
                                                  installed_capacity: float = None) -> float:
    """Compute the hydropower generation (in TWh) of a hydropower plant given a streamflow
    value running through the hydropower plant's turbine and its technical specifications.

    Parameters
    ----------
    streamflow : float
        The streamflow value to convert (in m^3/s)
    hydraulic_height : float
        The constant hydraulic head of the hydropower plant (in m)
    efficiency : float, optional
        The efficiency of the hydropower plant, by default 0.8
    simplified_efficiency : float, optional
        The simplified efficiency term of a power plant,
        computed from the installed capaciy, the design discharge
        and the hydraulic head, by default None
    design_discharge : float, optional
        The design discharge of the power plant (in m^3/s),
        by default None
    installed_capacity : float, optional
        The installed capacity of the power plant, at the generator
        (in W), by default None

    Returns
    -------
    float
        Hydropower generation estimate (in TWh) of a hydropower plant
        with the given technical specifications and streamflow value
    """
    if design_discharge:
        streamflow = np.clip(streamflow, 0.0, design_discharge)

    estimated_production = streamflow * hydraulic_height
    if simplified_efficiency:
        estimated_production *= simplified_efficiency
    else:
        estimated_production *= efficiency * GRAVITY * WATER_DENSITY

    estimated_production *= 1e-12

    if installed_capacity:
        estimated_production = np.clip(estimated_production, 0, installed_capacity)

    return estimated_production


def get_beta_coeff(ds_hp_production: xr.Dataset, expected_generation: float) -> float:
    """Given a time series of estimated hydropower production (temmporal resolution
    below yearly) of a hydropower plant and its yearly expected generation, compute a linear
    regression between the yearly hydropower production and the expected generation
    and obtain the regression coefficient (beta coefficient). Both the hydropower timeseries
    and the expected generation have to be expressed in the same units (default in TWh).

    Parameters
    ----------
    ds_hp_production : xr.Dataset
        xarray Dataset of estimated hydropower production
        of the hydropower plant
    expected_generation : float
        The yearly expected generation of the hydropower
        plant

    Returns
    -------
    float
        The regression coefficient Beta obtained from the linear regression
        between the yearly hydropower production and the expected generation
    """
    estimated_yearly_hp_production = ds_hp_production.resample(time="Y").sum().rgs.to_numpy()
    expected_yearly_hp_prod = np.array([expected_generation] * len(estimated_yearly_hp_production))
    reg = LinearRegression(fit_intercept=False).fit(estimated_yearly_hp_production.reshape(-1, 1),
                                                    expected_yearly_hp_prod)
    return reg.coef_.item()


def compute_ds_hydropower_generation_from_streamflow(ds_cumulative_streamflow_polygon: xr.Dataset, hydropower_wasta_number: int,
                                                     relevant_polygons: List[int], hydraulic_head: float, efficiency: float = 0.8,
                                                     simplified_efficiency: float = None, design_discharge: float = None,
                                                     installed_capacity: float = None) -> xr.Dataset:
    """Given an xarray Dataset of catchment streamflow, a hydropower plant's technical specifications and its catchment area,
    compute a timeseries of hydropower production estimation.

    Parameters
    ----------
    ds_cumulative_streamflow_polygon : xr.Dataset
        xarray Dataset of cumulative streamflow at
        a hydropower plant (in m^3/s)
    hydropower_wasta_number : int
        The WASTA number of the hydropower plant
    relevant_polygons : List[int]
        A list of the water intake catchments of the
        hydropower plant and their upstream catchments
    hydraulic_height : float
        The constant hydraulic head of the hydropower plant (in m)
    efficiency : float, optional
        The efficiency of the hydropower plant, by default 0.8
    simplified_efficiency : float, optional
        The simplified efficiency term of a power plant,
        computed from the installed capaciy, the design discharge
        and the hydraulic head, by default None
    design_discharge : float, optional
        The design discharge of the power plant (in m^3/s),
        by default None
    installed_capacity : float, optional
        The installed capacity of the power plant, at the generator
        (in W), by default None

    Returns
    -------
    xr.Dataset
        xarray Dataset of the hydropower generation estimates (in TWh) of a hydropower plant
        with the given technical specifications and streamflow values in its catchment area
    """
    ds = ds_cumulative_streamflow_polygon.sel(polygon=relevant_polygons)\
                                         .sum("polygon")\
                                         .map(lambda v: compute_hydropower_generation_from_streamflow(v, hydraulic_head,
                                                                                                      efficiency=efficiency,
                                                                                                      simplified_efficiency=simplified_efficiency,
                                                                                                      design_discharge=design_discharge,
                                                                                                      installed_capacity=installed_capacity))\
                                         .assign_coords(hydropower=("hydropower", [hydropower_wasta_number]))

    return ds


def convert_da_time_series_to_df_per_year(da: xr.DataArray, time_dim="time") -> pd.DataFrame:
    """Convert an xarray DataArray containing an hourly time series into a pandas DataFrame
    with years as columns and hourly values as rows (8760 rows).

    Parameters
    ----------
    da : xr.DataArray
        xarray DataArray containing the hourly time series
    time_dim : str, optional
        Name of the time dimension in the xarray DataArray,
        by default "time"

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the hourly values as rows and years as columns
    """
    years = np.unique(da[time_dim].dt.year.values)
    list_ds = [da.sel(time=str(year)).values for year in years]
    df = pd.DataFrame(np.stack(list_ds).T, index=range(8760), columns=years)

    return df


def concat_list_ds_and_save(list_ds: List[xr.Dataset], output_filepath: pathlib.Path) -> None:
    """Concatenate a list of xarray Datasets containing hydropower timeseries
    and save them with a certain encoding to the given path.

    Parameters
    ----------
    list_ds : List[xr.Dataset]
        List of xarray Datasets containing hydropower timeseries
    output_filepath : pathlib.Path
        Output file path to save the concatenated xarray Dataset
    """
    ds = xr.concat(list_ds, "hydropower").rename({"rgs": "gen"})
    encoding = {var: HYDROPOWER_NETCDF_ENCODINGS.copy() for var in list(ds.data_vars.keys())}
    encoding['time'] = {'units': f"seconds since {np.datetime_as_string(ds.time[0].values)}"}

    if output_filepath.is_file():
        output_filepath.unlink()

    ds.to_netcdf(output_filepath, mode='w', encoding=encoding)

def compute_streamflow_aggregate_polygons_parallel(
        ds_streamflow: pathlib.Path,
        gdf_polygons: gpd.GeoDataFrame,
        df_pts_in_polygons: pd.DataFrame,
        polygons: List[int]
    ) -> xr.Dataset:
    """Compute accumulated streamflow at each polygon in parallel for all polygons
    in the the given list.

    Parameters
    ----------
    ds_streamflow : xr.Dataset
        xarray Dataset containing the gridded streamflow data
        from the hydrological model
    gdf_polygons : gpd.GeoDataFrame
        GeoDataframe containing the catchments' locations
    df_pts_in_polygons : pd.DataFrame
        DataFrame containing a matching between each point in
        the gridded data and the catchments
    polygons : List[int]
        List of polygons to compute accumulated streamflow at

    Returns
    -------
    xr.Dataset
        xarray Dataset containing the streamflow timeseries for all
        the given catchments
    """
    num_workers = 30
    with Pool(num_workers) as p:
        list_ds = [ds for ds in p.starmap(compute_streamflow_aggregate_polygon,
                                            zip(repeat(ds_streamflow),
                                                repeat(gdf_polygons),
                                                repeat(df_pts_in_polygons),
                                                polygons,
                                                repeat("sum")
                                                ))]

    return xr.concat(list_ds, "polygon").sortby("polygon")

