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

def mode_of(arr):
    arr = np.asarray(arr)
    vals, counts = np.unique(arr, return_counts=True)
    maxc = counts.max()
    modes = vals[counts == maxc]
    # if maxc == 1:
    #     print("No repeated values (all unique). Returning the smallest value(s) as 'mode':")
    # print(f"mode count = {maxc}; mode value(s) = {modes}")
    return modes

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
    upstream_polygons = relevant_row["upstream_EZGNR"].apply(lambda lst: [] if pd.isnull(lst) else literal_eval(lst)).iloc[0]
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

def aggregate_streamflow_with_mask(
    ds_streamflow: xr.Dataset,
    mask_da: xr.DataArray,
    method: str = "sum",
    fill_value: int = -1,
    polygons: list[int] | None = None,
    compute_counts: bool = False,
    rechunk: dict | None = None,
) -> xr.Dataset | tuple[xr.Dataset, xr.DataArray]:
    """
    Aggregate gridded streamflow to polygons using a precomputed (y,x) -> EZGNR mask.

    Parameters
    ----------
    ds_streamflow : xr.Dataset
        Dataset with dims including ('time','y','x'). Variables are aggregated.
    mask_da : xr.DataArray
        2D mask (y,x) with polygon ids (EZGNR) and a fill_value for non-polygon cells.
    method : {'sum','mean'}
        Aggregation method.
    fill_value : int
        Value in mask indicating "no polygon".
    polygons : list[int] | None
        Optional subset of polygon ids to keep (after aggregation).
    compute_counts : bool
        If True, also return DataArray of cell counts per polygon.
    rechunk : dict | None
        Optional rechunking (e.g. {'time': 8760, 'y': 400, 'x': 400}) before aggregation.

    Returns
    -------
    xr.Dataset  (or (xr.Dataset, xr.DataArray) if compute_counts)
    """
    if method not in ("sum", "mean"):
        raise ValueError("method must be 'sum' or 'mean'.")

    # Optional rechunk to balance tasks; ensure y/x chunks not too tiny
    if rechunk:
        ds_streamflow = ds_streamflow.chunk({k: v for k, v in rechunk.items() if k in ds_streamflow.dims})

    # Ensure mask aligns (broadcast if needed)
    if not {"y", "x"}.issubset(mask_da.dims):
        raise ValueError("mask_da must have dims ('y','x').")
    # Align coordinates (no data load)
    mask_da = mask_da.sel(
        y=ds_streamflow.y, x=ds_streamflow.x
    )

    # Mask out non-polygon cells
    valid_mask = mask_da != fill_value
    masked_streamflow = ds_streamflow.where(valid_mask)

    # groupby on the mask (xarray allows DataArray of same shape)
    grouped = masked_streamflow.groupby(mask_da.where(valid_mask))

    # Sum over spatial dims (y,x) only; keep time (and other non-spatial dims)
    summed = grouped.sum(dim=("y", "x"))

    # mask_da values become the dimension name of groupby result; rename cleanly
    poly_dim_name = mask_da.name or "EZGNR"
    if poly_dim_name in summed.dims:
        summed = summed.rename({poly_dim_name: "polygon"})

    # Optional subset
    if polygons is not None:
        summed = summed.sel(polygon=[p for p in polygons if p in summed.polygon.values])

    if method == "mean":
        # Compute counts lazily (cells per polygon)
        ones = xr.ones_like(mask_da.where(valid_mask), dtype="int32")
        counts = ones.groupby(mask_da.where(valid_mask)).sum(dim=("y", "x"))
        if counts.name is None:
            counts.name = "cell_count"
        counts = counts.rename({counts.dims[0]: "polygon"})
        if polygons is not None:
            counts = counts.sel(polygon=[p for p in polygons if p in counts.polygon.values])
        # Divide each variable by counts (auto aligns on 'polygon')
        for v in summed.data_vars:
            summed[v] = summed[v] / counts
    else:
        if compute_counts:
            ones = xr.ones_like(mask_da.where(valid_mask), dtype="int32")
            counts = ones.groupby(mask_da.where(valid_mask)).sum(dim=("y", "x"))
            counts = counts.rename({counts.dims[0]: "polygon"})
            if polygons is not None:
                counts = counts.sel(polygon=[p for p in polygons if p in counts.polygon.values])
            return summed.sortby("polygon"), counts.sortby("polygon")

    if compute_counts and method == "mean":
        return summed.sortby("polygon"), counts.sortby("polygon")
    return summed.sortby("polygon")

def build_hydropower_parameter_table(
    df_stats_hydropower: pd.DataFrame,
    df_hydropower_polygons: pd.DataFrame,
    gross_head_cols: list[str] | None = None,
    allowed_types: list[str] | None = None,
    default_efficiency: float = 0.8,
    gravity: float = GRAVITY,
    water_density: float = WATER_DENSITY,
    capacity_col: str = "Max. Leistung ab Generator",
    design_discharge_col: str = "QTurbine [m3/sec]",
    turbined_flag_col: str = "Funktion: Turbinieren",
    yearly_generation_col: str = "Prod. ohne Umwälzbetrieb - J.",
    summer_generation_col: str = "Prod. ohne Umwälzbetrieb - S.",
    winter_generation_col: str = "Prod. ohne Umwälzbetrieb - W.",
    percentage_share_col: str = "Proz. Anteil CH",
    wasta_col_stats: str = "ZE-Nr",
    wasta_col_polygons: str = "WASTANumber",
) -> pd.DataFrame:
    """
    Create per-plant parameter table for vectorized hydropower generation.

    Parameters
    ----------
    df_stats_hydropower : pandas.DataFrame
        DataFrame with hydropower statistics (capacity, design discharge, heads, generation, etc.).
    df_hydropower_polygons : pandas.DataFrame
        DataFrame mapping hydropower plants to polygon EZGNRs and upstream polygons.
    gross_head_cols : list[str] or None, optional
        List of candidate column names in df_stats_hydropower that contain gross head values.
        If None, a default set of column names is used.
    allowed_types : list[str] or None, optional
        If provided, only plants with 'Type' in this list are retained.
    default_efficiency : float, optional
        Efficiency used when inferring head from capacity and design discharge (default 0.8).
    gravity : float, optional
        Gravity constant used when inferring head (default GRAVITY).
    water_density : float, optional
        Water density used when inferring head (default WATER_DENSITY).
    capacity_col : str, optional
        Column name in df_stats_hydropower for installed capacity (default "Max. Leistung ab Generator").
    design_discharge_col : str, optional
        Column name for design discharge (default "QTurbine [m3/sec]").
    turbined_flag_col : str, optional
        Column name indicating whether the plant turbines (default "Funktion: Turbinieren").
    yearly_generation_col : str, optional
        Column name for yearly generation (default "Prod. ohne Umwälzbetrieb - J.").
    summer_generation_col : str, optional
        Column name for summer generation (default "Prod. ohne Umwälzbetrieb - S.").
    winter_generation_col : str, optional
        Column name for winter generation (default "Prod. ohne Umwälzbetrieb - W.").
    percentage_share_col : str, optional
        Column name for percentage share in CH (default "Proz. Anteil CH").
    wasta_col_stats : str, optional
        Column name in df_stats_hydropower containing the WASTA identifier (default "ZE-Nr").
    wasta_col_polygons : str, optional
        Column name in df_hydropower_polygons containing the WASTA identifier (default "WASTANumber").

    Returns
    -------
    pandas.DataFrame
        Tidy DataFrame with one row per hydropower plant containing at minimum the following columns:
        - WASTANumber: plant identifier
        - polygons_full: list of associated polygon EZGNRs (base + upstream)
        - hydraulic_head: inferred or reported hydraulic head (m)
        - simplified_efficiency_F: simplified efficiency term F
        - installed_capacity_MW, design_discharge: original columns renamed for consistency
        - expected_* generation metrics and percentage_share_CH where available
        May also include Type, ZE-Name, BeginningOfOperation, EndOfOperation if present in inputs.

    Notes
    -----
    - Hydraulic head is inferred from available gross head columns or, if missing, from capacity and design
      discharge using the provided default_efficiency, gravity and water_density.
    - Plants with missing or non-positive capacity, design discharge or head are filtered out.
    """
    if gross_head_cols is None:
        gross_head_cols = [
            "Maxim. Bruttofallhöhe [m]",
            "Minim. Bruttofallhöhe [m]",
            "Maxim. Nettofallhöhe [m]",
        ]

    stats = df_stats_hydropower.copy()
    polys = df_hydropower_polygons.copy()

    # Merge polygon + stats
    merge_cols = [c for c in [wasta_col_stats, capacity_col, design_discharge_col, turbined_flag_col,
                              yearly_generation_col, summer_generation_col, winter_generation_col,
                              percentage_share_col] if c in stats.columns]
    head_cols_present = [c for c in gross_head_cols if c in stats.columns]
    opt_cols = [c for c in ("ZE-Name", "ZE-Erste Inbetriebnahme", "ZE-Letzte Inbetriebnahme", "Type") if c in stats.columns]

    merged = polys.merge(
        stats[merge_cols + head_cols_present + opt_cols],
        left_on=wasta_col_polygons,
        right_on=wasta_col_stats,
        how="left",
    )

    if allowed_types is not None and "Type" in merged.columns:
        merged = merged[merged["Type"].isin(allowed_types)]

    def _combine_polygons(row):
        base = row.get("EZGNR", [])
        up = row.get("upstream_EZGNR", [])
        if not isinstance(base, list):
            base = [base]
        if not isinstance(up, list):
            up = [up] if up else []
        # preserve order, remove duplicates
        seen = {}
        for p in base + up:
            seen[p] = True
        return list(seen.keys())

    merged["polygons_full"] = merged.apply(_combine_polygons, axis=1)

    def _infer_head(row):
        for col in gross_head_cols:
            if col in row and pd.notnull(row[col]) and row[col] > 0:
                return float(row[col])
        cap = row.get(capacity_col, np.nan)
        qd = row.get(design_discharge_col, np.nan)
        if pd.notnull(cap) and pd.notnull(qd) and qd > 0:
            try:
                return (cap * 1e6) / (qd * gravity * water_density * default_efficiency)
            except Exception:  # noqa: BLE001
                return np.nan
        return np.nan

    merged["hydraulic_head"] = merged.apply(_infer_head, axis=1)

    def _compute_F(row):
        cap = row.get(capacity_col, np.nan)
        qd = row.get(design_discharge_col, np.nan)
        hh = row.get("hydraulic_head", np.nan)
        if pd.notnull(cap) and pd.notnull(qd) and pd.notnull(hh) and qd > 0 and hh > 0:
            return (cap * 1e6) / (qd * hh)
        return np.nan

    merged["simplified_efficiency_F"] = merged.apply(_compute_F, axis=1)

    valid = (
        merged[design_discharge_col].notnull() & (merged[design_discharge_col] > 0) &
        merged[capacity_col].notnull() & (merged[capacity_col] > 0) &
        merged["hydraulic_head"].notnull() & (merged["hydraulic_head"] > 0)
    )
    merged = merged[valid].copy()

    rename_map = {
        capacity_col: "installed_capacity_MW",
        design_discharge_col: "design_discharge",
        yearly_generation_col: "expected_yearly_generation",
        summer_generation_col: "expected_summer_generation",
        winter_generation_col: "expected_winter_generation",
        percentage_share_col: "percentage_share_CH",
    }

    selected_cols = [wasta_col_polygons, "polygons_full", "hydraulic_head", "simplified_efficiency_F"] + list(rename_map.keys())
    for c in ("Type", "ZE-Name", "ZE-Erste Inbetriebnahme", "ZE-Letzte Inbetriebnahme"):
        if c in merged.columns:
            selected_cols.append(c)

    param_df = merged[selected_cols].rename(columns=rename_map)
    # Rename date columns if present
    date_map = {"ZE-Erste Inbetriebnahme": "BeginningOfOperation", "ZE-Letzte Inbetriebnahme": "EndOfOperation"}
    for k, v in date_map.items():
        if k in param_df.columns:
            param_df = param_df.rename(columns={k: v})

    param_df = param_df.rename(columns={wasta_col_polygons: "WASTANumber"}).sort_values("WASTANumber").reset_index(drop=True)
    return param_df


def build_polygon_plant_weights(
    polygons_all: np.ndarray,
    hp_param_df: pd.DataFrame,
    polygon_list_col: str = "polygons_full",
    plant_id_col: str = "WASTANumber",
    dtype: str | np.dtype = "uint8",
) -> xr.DataArray:
    """Build a dense (polygon x hydropower) weight matrix (0/1) as DataArray.

    Parameters
    ----------
    polygons_all : np.ndarray
        Sorted array of polygon IDs matching ds.polygon coord order.
    hp_param_df : pd.DataFrame
        Parameter table with a list of polygons per plant.
    polygon_list_col : str
        Column containing list[int] of all polygons associated with the plant.
    plant_id_col : str
        Column containing unique hydropower plant IDs.
    dtype : str | np.dtype
        Numpy dtype for weight matrix (uint8 or bool recommended).

    Returns
    -------
    xr.DataArray
        Dimensions: ('polygon','hydropower'). Values 0/1 membership.
    """
    poly_index = {pid: i for i, pid in enumerate(polygons_all)}
    plant_ids = hp_param_df[plant_id_col].to_numpy()
    P = len(polygons_all)
    H = len(plant_ids)
    weights = np.zeros((P, H), dtype=dtype)
    for j, polys in enumerate(hp_param_df[polygon_list_col]):
        if not isinstance(polys, (list, tuple, np.ndarray)):
            continue
        for pid in polys:
            i = poly_index.get(pid)
            if i is not None:
                weights[i, j] = 1
    return xr.DataArray(
        weights,
        dims=("polygon", "hydropower"),
        coords={"polygon": polygons_all, "hydropower": plant_ids},
        name="weights",
    )

def aggregate_polygon_to_plant_flow(
    ds_polygon_flow: xr.Dataset,
    weights_da: xr.DataArray,
    variable: str = "rgs",
) -> xr.DataArray:
    """Aggregate polygon flow variable to hydropower plants using dense weights.

    Parameters
    ----------
    ds_polygon_flow : xr.Dataset
        Dataset with dims including ('time','polygon') and the flow variable.
    weights_da : xr.DataArray
        Weight matrix (polygon, hydropower) with 0/1 membership.
    variable : str
        Name of flow variable in ds_polygon_flow to aggregate.

    Returns
    -------
    xr.DataArray
        Flow per hydropower plant (time, hydropower) in same units as input variable.
    """
    if variable not in ds_polygon_flow.data_vars:
        raise ValueError(f"Variable '{variable}' not in dataset.")
    # Ensure polygon alignment
    weights_da = weights_da.sel(polygon=ds_polygon_flow.polygon)
    flow = xr.dot(ds_polygon_flow[variable], weights_da, dims="polygon")
    flow.name = "flow"
    return flow

def compute_generation_vectorized(
    plant_flow: xr.DataArray,
    hp_params_df: pd.DataFrame,
    timestep_hours: float,
    use_simplified_efficiency: bool = True,
    efficiency: float = 0.8,
    gravity: float = GRAVITY,
    water_density: float = WATER_DENSITY,
) -> xr.DataArray:
    """Compute hydropower generation for all plants vectorized over (time, hydropower).

    Parameters
    ----------
    plant_flow : xr.DataArray
        (time, hydropower) aggregated flow (m^3/s).
    hp_params_df : pd.DataFrame
        Parameter table from build_hydropower_parameter_table.
    timestep_hours : float
        Length of a model timestep in hours (1 for hourly, 24 for daily).
    use_simplified_efficiency : bool
        If True use simplified efficiency term F (flow * head * F). If False use physics formula.
    efficiency : float
        Default turbine efficiency used only when use_simplified_efficiency=False.
    gravity : float
        Gravity constant.
    water_density : float
        Water density.

    Returns
    -------
    xr.DataArray
        Energy generation (TWh) per timestep per plant.
    """
    wasta = hp_params_df["WASTANumber"].to_numpy()
    # Broadcast parameter arrays
    head = xr.DataArray(hp_params_df["hydraulic_head"].to_numpy(), dims=("hydropower",), coords={"hydropower": wasta})
    design_q = xr.DataArray(hp_params_df["design_discharge"].to_numpy(), dims=("hydropower",), coords={"hydropower": wasta})
    capacity_MW = xr.DataArray(hp_params_df["installed_capacity_MW"].to_numpy(), dims=("hydropower",), coords={"hydropower": wasta})
    F = xr.DataArray(hp_params_df["simplified_efficiency_F"].to_numpy(), dims=("hydropower",), coords={"hydropower": wasta})

    # Align hydropower order
    plant_flow = plant_flow.sel(hydropower=wasta)

    # Clip by design discharge
    flow_eff = xr.apply_ufunc(
        np.minimum,
        plant_flow,
        design_q,
        dask="parallelized",
        output_dtypes=[plant_flow.dtype],
    )

    dt_seconds = timestep_hours * 3600.0
    if use_simplified_efficiency:
        power_W = flow_eff * head * F  # assumed F in W/(m3/s * m)
    else:
        power_W = flow_eff * head * gravity * water_density * efficiency

    energy_Wh = power_W * (dt_seconds / 3600.0)
    energy_TWh = energy_Wh * 1e-12

    capacity_TWh_step = (capacity_MW * timestep_hours) * 1e-6  # MW*h -> TWh
    gen = xr.apply_ufunc(
        np.minimum,
        energy_TWh,
        capacity_TWh_step,
        dask="parallelized",
        output_dtypes=[energy_TWh.dtype],
    )
    gen.name = "gen"
    return gen


def compute_hydropower_production_vectorized(
    ds_polygon_flow: xr.Dataset,
    hp_params_df: pd.DataFrame,
    variable: str = "rgs",
    timestep_hours: int = 1,
    use_simplified_efficiency: int = True,
    output_path: pathlib.Path | None = None,
    encoding: dict | None = None,
) -> xr.Dataset:
    """High-level orchestration for vectorized hydropower generation.

    Steps:
      1. Build weights (dense or sparse).
      2. Aggregate polygon flow to plant flow.
      3. Compute generation vectorized.
      4. Attach static parameters.
      5. Optionally write to Zarr/NetCDF.
    """
    polygons_all = ds_polygon_flow.polygon.to_numpy()
    weights_da = build_polygon_plant_weights(polygons_all, hp_params_df)
    plant_flow = aggregate_polygon_to_plant_flow(ds_polygon_flow, weights_da, variable=variable)

    gen = compute_generation_vectorized(
        plant_flow, hp_params_df, timestep_hours=timestep_hours, use_simplified_efficiency=use_simplified_efficiency
    )

    # Build output Dataset
    ds_out = xr.Dataset({"gen": gen})
    # Add parameter coordinates for traceability
    for col in [
        "hydraulic_head",
        "design_discharge",
        "installed_capacity_MW",
        "simplified_efficiency_F",
    ]:
        if col in hp_params_df.columns:
            ds_out[col] = ("hydropower", hp_params_df[col].to_numpy())

    if output_path is not None:
        if encoding is None:
            encoding = {"gen": {"chunks": (int(24 * 365 / timestep_hours), ds_out.sizes["hydropower"])}}
        ds_out.to_zarr(output_path, mode="w", encoding=encoding)
    return ds_out