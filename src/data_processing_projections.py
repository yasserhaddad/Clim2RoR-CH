import json
import os

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from itertools import repeat
from multiprocessing import Pool

from src.extract_runoff_prevah import (
    batch_extraction_prevah,
)
from src.utils_polygons import (
    find_upstream_polygons_recursive,
    flatten_list,
    get_points_in_polygons,
)
from src.utils_streamflow_hydropower import (
    GRAVITY,
    WATER_DENSITY,
    compute_ds_hydropower_generation_from_streamflow,
    compute_simplified_efficiency_term,
    aggregate_streamflow_with_mask,
    concat_list_ds_and_save,
    convert_mm_d_to_cubic_m_s,
    build_hydropower_parameter_table,
    compute_hydropower_production_vectorized,
)
from src.var_attributes import ACCUM_HYDRO_ZARR_ENCODING

# Set environment variables after imports so import block remains contiguous
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["USE_PYGEOS"] = "0"

import dask
from dask.diagnostics import ProgressBar
import geopandas as gpd

DEFAULT_EFFICIENCY = 0.8


class DataProcessingDask:
    def __init__(self, paths_file: str, climate_model_chain: str, climate_scenario: str):
        print("Loading data")
        paths = json.load(open(paths_file))
        self.path_data_ror = Path(paths["path_data_ror"])
        self.path_data_projections = Path(paths["path_data_projections"])
        self.climate_model_chain = climate_model_chain
        self.climate_scenario = climate_scenario

        self.path_data_prevah = (
            self.path_data_projections
            / "cordex_processed"
            / f"{self.climate_model_chain}_{self.climate_scenario}"
            / "prevah"
        )
        self.path_data_hydro = self.path_data_ror / "hydropower"
        self.path_data_polygons = self.path_data_ror / "polygons"

        self.gdf_polygons = gpd.read_file(
            self.path_data_polygons / "EZG_Gewaesser.gpkg"
        )
        self.df_prevah_pts_in_polygons = None

        self.df_stats_hydropower_ch = pd.read_excel(
            self.path_data_hydro / "stats_hydropower_ch" / "wasta_2023_updated.xlsx"
        )
        df_hydropower_locations = self.df_stats_hydropower_ch[
            [
                "ZE-Nr",
                "ZE-Name",
                "ZE-Standort",
                "ZE-Kanton",
                "WKA-Typ",
                "Max. Leistung ab Generator",
                "ZE-Erste Inbetriebnahme",
                "ZE-Letzte Inbetriebnahme",
                "ZE-Koordinaten unscharf (Ost)",
                "ZE-Koordinaten unscharf (Nord)",
            ]
        ]
        df_hydropower_locations = df_hydropower_locations.rename(
            {
                "ZE-Nr": "WASTANumber",
                "ZE-Name": "Name",
                "ZE-Standort": "Location",
                "ZE-Kanton": "Canton",
                "WKA-Typ": "Type",
                "Max. Leistung ab Generator": "Capacity",
                "ZE-Erste Inbetriebnahme": "BeginningOfOperation",
                "ZE-Letzte Inbetriebnahme": "EndOfOperation",
                "ZE-Koordinaten unscharf (Ost)": "_x",
                "ZE-Koordinaten unscharf (Nord)": "_y",
            },
            axis=1,
        ).fillna({"EndOfOperation": 9999})
        self.gdf_hydropower_locations = gpd.GeoDataFrame(
            df_hydropower_locations,
            geometry=gpd.points_from_xy(
                df_hydropower_locations["_x"], df_hydropower_locations["_y"]
            ),
            crs="EPSG:2056",
        )

        self.df_new_hydropower_polygons = pd.read_csv(
            self.path_data_hydro / "hydropower_polygons" / "hp_new_polygons.csv"
        )[["Checked", "To change", "New EZGNR", "Name", "WASTANumber"]]

        df_residual_flow = pd.read_csv(
            self.path_data_hydro / "residual_flow_ch.txt", sep="\t"
        )
        df_residual_flow_hydropower = df_residual_flow[
            df_residual_flow["But du prélèvement"] == "Centrale hydroélectrique"
        ]
        self.gdf_water_intake = gpd.GeoDataFrame(
            df_residual_flow_hydropower,
            geometry=gpd.points_from_xy(
                df_residual_flow_hydropower["Prélèvement - Coordonnées WE"],
                df_residual_flow_hydropower["Prélèvement - Coordonnées SN"],
            ),
            crs="EPSG:21781",
        ).to_crs("EPSG:2056")

        self.df_hydropower_polygons = None
        self.ds_accumulated_streamflow_polygon = None

    def convert_bin_to_netcdf_runoff_prevah(self) -> None:
        """Extracts runoff values from gz binary PREVAH data and stores them in
        netcdf files. Each file contains one year of data.
        """
        path_hydro_tar = self.path_data_prevah / "compressed"
        netcdf_output_dir = self.path_data_prevah / "netcdf"
        if netcdf_output_dir.exists():
            shutil.rmtree(netcdf_output_dir)
        netcdf_output_dir.mkdir(exist_ok=True)

        product = "RGS"
        batch_extraction_prevah(
            path_hydro_tar,
            netcdf_output_dir,
            product,
            prefix_filename_tgz="wsl2zero_",
            prefix_filename_gz="Mob500",
            convert_coords=True,
            num_workers=8,
        )

    def extract_points_in_polygons(
        self,
        output_filename: str = "df_prevah_500_pts_in_polygons.csv"
    ) -> None:
        """Extracts points from the PREVAH grid that are located in the polygons of Swiss waterbodies.

        Parameters
        ----------
        output_filename : str, optional
            Name of output file containing the DataFrame of the points present in each polygon,
            by default "df_prevah_500_pts_in_polygons.csv"
        """
        print("Extracting dataset grid points in polygons")
        time_start = time.time()
        # Load sample runoff data
        ds_sample = xr.open_zarr(
            self.path_data_prevah / "rgs.zarr", chunks="auto"
        )

        # Transform a sample runoff grid into a GeoDataFrame to use operations included in GeoPandas
        df_runoff = ds_sample.isel(time=0).to_dataframe().reset_index()
        gdf_runoff = gpd.GeoDataFrame(
            df_runoff,
            geometry=gpd.points_from_xy(df_runoff.x, df_runoff.y),
            crs="EPSG:2056",
        )[["y", "x", "geometry"]]
        del ds_sample

        # Get points in polygons and speed it up by launching multiple processes in parallel
        step = 500
        split_gdfs = [
            self.gdf_polygons.iloc[i : i + 500]
            for i in range(0, len(self.gdf_polygons), step)
        ]

        num_workers = 20
        with Pool(num_workers) as p:
            list_gdfs = [
                ds
                for ds in p.starmap(
                    get_points_in_polygons, zip(split_gdfs, repeat(gdf_runoff))
                )
            ]

        df_concat = pd.DataFrame(
            pd.concat(list_gdfs, ignore_index=True).drop(columns="geometry")
        ).sort_values(by="EZGNR")

        df_concat[["index_left", "EZGNR", "TEILEZGNR", "y", "x"]].to_csv(
            self.path_data_polygons / output_filename, index=False
        )
        self.df_prevah_pts_in_polygons = pd.read_csv(
            self.path_data_polygons / output_filename
        )

        print(f"\tExtracted successfully all points in polygons! Time elapsed: {(time.time() - time_start)/60:.2f} minutes.")

    def create_prevah_polygon_grid(
        self,
        df_prevah_pts_in_polygons_filename: str = "df_prevah_500_pts_in_polygons.csv",
        fill_value: int = -1,
    ) -> xr.DataArray:
        """Create an xarray DataArray mapping each PREVAH grid cell (y,x) to the EZGNR polygon id.

        The function:
        - loads the points-in-polygons table (if not already loaded),
        - pivots the table into a 2D array indexed by y (rows) and x (cols),
        - aligns that 2D array to the full grid coordinates,
        - returns an xarray.DataArray with dims ('y','x') where each cell contains the EZGNR
          (or fill_value if the grid cell is not in any polygon).

        Notes:
        - If multiple polygon ids exist for the same grid cell, the first is kept.
        """
        # load points-in-polygons table if required
        if self.df_prevah_pts_in_polygons is None:
            self.df_prevah_pts_in_polygons = pd.read_csv(
                self.path_data_polygons / df_prevah_pts_in_polygons_filename
            )

        df_pts = self.df_prevah_pts_in_polygons.copy()

        # ensure columns exist and correct dtypes
        if "x" not in df_pts.columns or "y" not in df_pts.columns or "EZGNR" not in df_pts.columns:
            raise ValueError("df_prevah_pts_in_polygons must contain columns 'x', 'y' and 'EZGNR'")

        # Derive grid coordinates directly from the points-in-polygons dataframe
        x_coords = np.sort(df_pts["x"].unique())
        y_coords = np.sort(df_pts["y"].unique())

        # Pivot the dataframe to 2D: index=y, columns=x, values=EZGNR (keep first if duplicates)
        df_pivot = (
            df_pts[["y", "x", "EZGNR"]]
            .dropna(subset=["x", "y", "EZGNR"])
            # .astype({"x": float, "y": float})
            .pivot_table(index="y", columns="x", values="EZGNR", aggfunc="first")
        )

        # Reindex pivot to match the full grid coordinates (this ensures correct ordering and fills missing)
        df_pivot = df_pivot.reindex(index=y_coords, columns=x_coords)

        # Convert to numpy and fill missing with fill_value
        arr = df_pivot.to_numpy()  # flip y to match increasing y coord
        arr = np.where(np.isnan(arr), fill_value, arr).astype(int)

        # Create DataArray with same coordinate labels as the sample dataset
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
            name="EZGNR",
        )

        # save to zarr for later quick loading
        self.grid_pt_in_polygon = da

        return da

    def compute_accumulated_streamflow_per_polygon(
        self,
        df_prevah_pts_in_polygons_filename: str = "df_prevah_500_pts_in_polygons.csv",
        grid_fill_value: int = -1,
        streamflow_grid_resolution: float = 500,
        output_filename: str = "ds_prevah_500_streamflow_accum_per_polygon",
    ) -> None:
        """Compute accumulated streamflow at each polygon.

        Parameters
        ----------
        df_prevah_pts_in_polygons_filename : str, optional
            Name of file containing the DataFrame of PREVAH grid points present in
            each polygon, by default "df_prevah_500_pts_in_polygons.csv"
        prefix: str, optional
            Prefix of the netCDF files containing streamflow data that indicate the
            simulation names (used to detect year), by default "Mob500_RGS_"
        streamflow_grid_resolution: float, optional
            Grid resolution (in m) of the input streamflow data to aggregate, by default
            500
        output_filename : str, optional
            Name of output file (without extension) containing the xarray Dataset of accumulated
            streamflow at each polygon, by default "ds_prevah_500_streamflow_accum_per_polygon"
        """
        time_start = time.time()
        relevant_polygons = self.gdf_polygons.EZGNR.to_numpy()
        if self.df_prevah_pts_in_polygons is None:
            self.df_prevah_pts_in_polygons = pd.read_csv(
                self.path_data_polygons / df_prevah_pts_in_polygons_filename
            )
        if self.grid_pt_in_polygon is None:
            self.create_prevah_polygon_grid(
                df_prevah_pts_in_polygons_filename=df_prevah_pts_in_polygons_filename,
                fill_value=grid_fill_value
            )

        # Converting streamflow to accumulated streamflow per polygon
        print("Computing accumulated streamflow")
        ds_rgs = xr.open_zarr(self.path_data_prevah / "rgs.zarr", chunks="auto").apply(
            lambda v: xr.apply_ufunc(
                convert_mm_d_to_cubic_m_s,
                v,
                streamflow_grid_resolution**2,
                vectorize=True,
                dask="parallelized",
                output_dtypes=[v.dtype],
            )
        )
        with ProgressBar(dt=10):
            ds_accum = (
                aggregate_streamflow_with_mask(
                    ds_rgs,
                    self.grid_pt_in_polygon,
                    method="sum",
                    fill_value=grid_fill_value,
                    polygons=relevant_polygons,
                )
            )

            ds_accum = ds_accum.sel(time=~(ds_accum.time.dt.month == 2) & ~(ds_accum.time.dt.day == 29))
            encoding = {
                var: {
                    "compressor": ACCUM_HYDRO_ZARR_ENCODING["compressor"],
                    "chunks": {"time": 365, "polygon": 1000}
                }
                for var in list(ds_accum.data_vars.keys())
            }
            encoding["time"] = {
                "units": f"seconds since {np.datetime_as_string(ds_accum.time[0].values)}"
            }

            output_filepath = self.path_data_prevah / f"{output_filename}.zarr"
            ds_accum.to_zarr(output_filepath, mode="w", encoding=encoding)

            new_end_time = ds_accum.time[-1] + np.timedelta64(23, "h")
            ds_accum_hourly = ds_accum.reindex(
                time=pd.date_range(
                    start=ds_accum.time[0].values,
                    end=new_end_time.values,
                    freq="1H",
                    inclusive="both",
                ),
                method="ffill",
            )
            ds_accum_hourly = ds_accum_hourly.sel(time=~(ds_accum_hourly.time.dt.month == 2) & ~(ds_accum_hourly.time.dt.day == 29))
            encoding = {
                var: {
                    "compressor": ACCUM_HYDRO_ZARR_ENCODING["compressor"],
                    "chunks": {"time": 8760, "polygon": 1000}
                }
                for var in list(ds_accum_hourly.data_vars.keys())
            }
            output_filepath = self.path_data_prevah / f"{output_filename}_hourly.zarr"
            ds_accum_hourly.to_zarr(output_filepath, mode="w", encoding=encoding)
            self.ds_accumulated_streamflow_polygon = ds_accum_hourly

        print(
            f"\tTotal time for accumulated streamflow: {(time.time() - time_start)/60:.2f} minutes."
        )

    def get_df_upstream_polygons(self) -> pd.DataFrame:
        """Get all upstream polygons of each polygon in a pandas DataFrame containing the connectivity
        between polygons.

        Returns
        -------
        pd.DataFrame
            pandas DataFrame containing polygons and their corresponding
            upstream polygons, identified by their EZGNR
        """
        df_polygon_connectivity = pd.read_csv(
            self.path_data_polygons / "st2km2_ConnectivityEZGNR_polyg.csv"
        )
        df_upstream_polygons = (
            df_polygon_connectivity.groupby("tEZGNR")["fEZGNR"]
            .apply(list)
            .reset_index()
        )
        df_upstream_polygons.columns = ["EZGNR", "upstream_EZGNR"]

        df_upstream_polygons["upstream_EZGNR"] = df_upstream_polygons.apply(
            lambda row: find_upstream_polygons_recursive(
                df_upstream_polygons, row["EZGNR"]
            ),
            axis=1,
        )

        return df_upstream_polygons

    def get_hydropower_polygons(
        self, df_upstream_polygons: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the polygon containing each hydropower plant present in the WASTA database
        and their corresponding upstream polygons.

        Parameters
        ----------
        df_upstream_polygons : pd.DataFrame
            pandas DataFrame containing polygons and their corresponding
            upstream polygons, identified by their EZGNR

        Returns
        -------
        pd.DataFrame
            pandas DataFrame linking each hydropower plant in the WASTA
            database to the polygon that contains it and its upstream area.
        """
        gdf_hydropower_polygons = gpd.sjoin_nearest(
            self.gdf_hydropower_locations, self.gdf_polygons
        )
        gdf_hydropower_polygons = gpd.GeoDataFrame(
            gdf_hydropower_polygons.merge(df_upstream_polygons, on="EZGNR", how="left")
        )

        df_hydropower_polygons = pd.DataFrame(
            gdf_hydropower_polygons.drop(columns="geometry")
        )

        return df_hydropower_polygons

    def get_water_intake_polygons(self) -> pd.DataFrame:
        """Get the polygons containing water intake points of hydropower plants.

        Returns
        -------
        pd.DataFrame
            pandas DataFrame linking each water intake point of hydropower plants
            to the polygon containing it.
        """
        df_water_intake_polygons = gpd.sjoin(
            self.gdf_polygons,
            self.gdf_water_intake,
            predicate="intersects",
            how="right",
        )
        df_water_intake_polygons = (
            df_water_intake_polygons[~pd.isna(df_water_intake_polygons["n° WASTA"])][
                ["EZGNR", "n° WASTA"]
            ]
            .groupby(["n° WASTA"])["EZGNR"]
            .apply(lambda s: list(set(s)))
            .reset_index()
            .rename(columns={"n° WASTA": "WASTANumber", "EZGNR": "Water Intake EZGNR"})
        )
        df_water_intake_polygons["WASTANumber"] = df_water_intake_polygons[
            "WASTANumber"
        ].astype(int)

        return df_water_intake_polygons

    def get_hydropower_plants_to_update(
        self, df_water_intake_polygons: pd.DataFrame, save: bool = True
    ) -> pd.DataFrame:
        """Get the hydropower plants that have been inspected manually and which
        water intake point needs to be updated.

        Parameters
        ----------
        df_water_intake_polygons : pd.DataFrame
            pandas DataFrame linking each water intake point of hydropower
            plants to the polygon containing it
        save : bool, optional
            Whether to save the resulting pandas DataFrame with
            the new polygons corresponding to the hydropower
            plants to update, by default True

        Returns
        -------
        pd.DataFrame
            pandas DataFrame containing the hydropower plants to update
            along with the polygon of their manually assigned water intake point.
        """
        df_to_change = self.df_new_hydropower_polygons[
            (self.df_new_hydropower_polygons["Checked"].astype(bool))
            & (self.df_new_hydropower_polygons["To change"].astype(bool))
        ]
        df_to_change.loc[:, "New EZGNR"] = df_to_change.apply(
            lambda row: df_water_intake_polygons[
                df_water_intake_polygons["WASTANumber"] == row["WASTANumber"]
            ]["Water Intake EZGNR"].item()
            if pd.isna(row["New EZGNR"])
            else [int(elem) for elem in str(row["New EZGNR"]).split(", ")],
            axis=1,
        )

        df_remaining_to_change = pd.merge(
            self.gdf_hydropower_locations[["WASTANumber", "Name"]],
            df_water_intake_polygons,
        )
        df_remaining_to_change = (
            df_remaining_to_change[
                (
                    ~df_remaining_to_change["WASTANumber"].isin(
                        df_to_change["WASTANumber"]
                    )
                )
            ]
            .copy()
            .reset_index()
        )
        df_remaining_to_change["Checked"] = True
        df_remaining_to_change["To change"] = True
        df_remaining_to_change = df_remaining_to_change[
            ["Checked", "To change", "Water Intake EZGNR", "Name", "WASTANumber"]
        ].rename(columns={"Water Intake EZGNR": "New EZGNR"})

        df_hydropower_polygons_to_update = pd.concat(
            [df_to_change, df_remaining_to_change]
        ).reset_index(drop=True)[["Name", "WASTANumber", "New EZGNR"]]
        if save:
            df_hydropower_polygons_to_update.to_json(
                self.path_data_hydro
                / "hydropower_polygons"
                / "hp_polygons_to_change.json",
                orient="records",
            )

        return df_hydropower_polygons_to_update

    def get_catchment_area_per_hydropower(self):
        """Get the catchment area of each hydropower plant in the WASTA database.
        Each power plant is linked to the polygon containing it and their upstream area.
        Some power plants have been inspected manually and new polygons have been assigned
        to them.
        """
        print("Extracting catchment area of each hydropower plant in the WASTA database")
        time_start = time.time()
        # Get catchment area (all upstream polygons) of hydropower plant
        df_upstream_polygons = self.get_df_upstream_polygons()

        # Get catchment containing the hydropower plants and its upstream polygons
        df_hydropower_polygons = self.get_hydropower_polygons(df_upstream_polygons)

        # Get polygons of hydropower plants to update
        df_hydropower_polygons_to_update = self.get_hydropower_plants_to_update(
            self.get_water_intake_polygons()
        )

        # Update hydropower information with manually assigned polygons, turn the EZGNR field into a list
        df_hydropower_polygons.loc[:, "EZGNR"] = df_hydropower_polygons.apply(
            lambda row: [row["EZGNR"]]
            if row["WASTANumber"] not in df_hydropower_polygons_to_update["WASTANumber"]
            else df_hydropower_polygons_to_update.loc[
                df_hydropower_polygons_to_update["WASTANumber"] == row["WASTANumber"],
                "New EZGNR",
            ],
            axis=1,
        )
        # Re-compute upstream EZGNR for updated hydropower plants
        df_hydropower_polygons.loc[
            df_hydropower_polygons["WASTANumber"].isin(
                df_hydropower_polygons_to_update["WASTANumber"]
            ),
            "upstream_EZGNR",
        ] = df_hydropower_polygons[
            df_hydropower_polygons["WASTANumber"].isin(
                df_hydropower_polygons_to_update["WASTANumber"]
            )
        ].apply(
            lambda row: list(
                set(
                    flatten_list(
                        [
                            find_upstream_polygons_recursive(
                                df_upstream_polygons, catchment
                            )
                            for catchment in row["EZGNR"]
                        ]
                    )
                )
            ),
            axis=1,
        )

        # Fill NaN for upstream polygons with empty list
        df_hydropower_polygons.loc[:, "upstream_EZGNR"] = (
            df_hydropower_polygons["upstream_EZGNR"].fillna("").apply(list)
        )

        # Save dataframe matching hydropower plant locations with the BAFU catchments
        output_path = self.path_data_hydro / "hydropower_polygons"
        output_path.mkdir(parents=True, exist_ok=True)
        df_hydropower_polygons = df_hydropower_polygons[
            [
                "WASTANumber",
                "Name",
                "Type",
                "_x",
                "_y",
                "EZGNR",
                "upstream_EZGNR",
            ]
        ]
        df_hydropower_polygons.to_json(
            self.path_data_hydro
            / "hydropower_polygons"
            / "df_hydropower_polygons.json",
            orient="records",
        )
        self.df_hydropower_polygons = df_hydropower_polygons
        print(f"Time elapsed: {(time.time() - time_start)/60:.2f} minutes.")

    def build_polygon_plant_weights(
            self,
            polygons_all: np.ndarray,
            hp_param_df: pd.DataFrame,
            polygon_col='polygons_full',
            dtype='uint8') -> xr.DataArray:
        # polygons_all: sorted array of all polygon IDs in ds_accumulated_streamflow_polygon.polygon
        poly_index = {pid: i for i, pid in enumerate(polygons_all)}
        H = len(hp_param_df)
        P = len(polygons_all)
        weights = np.zeros((P, H), dtype=dtype)
        for j, polys in enumerate(hp_param_df[polygon_col]):
            for pid in polys:
                i = poly_index.get(pid)
                if i is not None:
                    weights[i, j] = 1
        return xr.DataArray(
            weights,
            dims=('polygon', 'hydropower'),
            coords={'polygon': polygons_all, 'hydropower': hp_param_df['WASTANumber'].values},
            name='weights'
        )



    def compute_hydropower_production(
        self,
        accumulated_streamflow_per_polygon_filename: str = "ds_prevah_500_streamflow_accum_per_polygon_hourly.zarr",
        output_filename_prefix="ds_prevah_500_hydropower_production_ror",
    ) -> None:
        """Compute the hydropower production by converting the accumulated streamflow at the polygon assigned
        as water intake point of the hydropower plant into energy quantities.

        Parameters
        ----------
        accumulated_streamflow_per_polygon_filename : str, optional
            Name of netcdf file containing the accumulated streamflow at each polygon
            for every time step, by default "ds_prevah_500_streamflow_accum_per_polygon_hourly.zarr"
        output_filename_prefix : str, optional
            Prefix to add to each filename of the outputs to save, by default
            "ds_prevah_500_hydropower_production_ror"
        """
        # Load hydropower polygons
        if self.df_hydropower_polygons is None:
            self.df_hydropower_polygons = pd.read_json(
                self.path_data_hydro
                / "hydropower_polygons"
                / "df_hydropower_polygons.json",
                orient="records",
            )
        # Load accumulated streamflow per polygon
        if self.ds_accumulated_streamflow_polygon is None:
            self.ds_accumulated_streamflow_polygon = xr.open_zarr(
                self.path_data_prevah / accumulated_streamflow_per_polygon_filename
            )

        # --------------------------------------------------------------------------------------------------
        # Converting accumulated streamflow into hydropower generation
        print("Computing hydropower generation")

        gross_head_cols = [
            "Maxim. Bruttofallhöhe [m]",
            "Minim. Bruttofallhöhe [m]",
            "Maxim. Nettofallhöhe [m]",
        ]
        allowed_types = ["L"]
        time_start = time.time()

        df_hydropower_to_process = self.df_hydropower_polygons[
            (self.df_hydropower_polygons["Type"].isin(allowed_types))
        ]
        nb_hp = len(df_hydropower_to_process)

        list_ds = []
        list_parameters = []

        for idx, (_, hydropower_info) in enumerate(df_hydropower_to_process.iterrows()):
            relevant_polygons = (
                hydropower_info["EZGNR"] + hydropower_info["upstream_EZGNR"]
            )
            relevant_stats_row = self.df_stats_hydropower_ch[
                self.df_stats_hydropower_ch["ZE-Nr"] == hydropower_info["WASTANumber"]
            ]
            installed_capacity = relevant_stats_row["Max. Leistung ab Generator"].item()
            design_discharge = relevant_stats_row["QTurbine [m3/sec]"].item()
            expected_generation = relevant_stats_row[
                "Prod. ohne Umwälzbetrieb - J."
            ].item()
            expected_summer_generation = relevant_stats_row[
                "Prod. ohne Umwälzbetrieb - S."
            ].item()
            expected_winter_generation = relevant_stats_row[
                "Prod. ohne Umwälzbetrieb - W."
            ].item()
            is_turbined = relevant_stats_row["Funktion: Turbinieren"].item()

            if design_discharge == 0 or pd.isna(is_turbined):
                print(
                    f"\n\t{hydropower_info['WASTANumber']}\t{installed_capacity}\t{design_discharge}"
                )
                continue

            hydraulic_head = relevant_stats_row[gross_head_cols].to_numpy()
            hydraulic_head = hydraulic_head[hydraulic_head > 0]
            if len(hydraulic_head) == 0:
                try:
                    hydraulic_head = int(
                        installed_capacity
                        * 1e6
                        / (
                            design_discharge
                            * GRAVITY
                            * WATER_DENSITY
                            * DEFAULT_EFFICIENCY
                        )
                    )
                except ZeroDivisionError:
                    print(
                        f"\n{hydropower_info['WASTANumber']}\t{installed_capacity}\t{design_discharge}"
                    )
            else:
                hydraulic_head = hydraulic_head[0]

            F = round(
                compute_simplified_efficiency_term(
                    installed_capacity * 1e6, design_discharge, hydraulic_head
                ),
                2,
            )

            ds = compute_ds_hydropower_generation_from_streamflow(
                self.ds_accumulated_streamflow_polygon,
                hydropower_info["WASTANumber"],
                relevant_polygons,
                hydraulic_head,
                DEFAULT_EFFICIENCY,
                simplified_efficiency=F,
                design_discharge=design_discharge,
                installed_capacity=installed_capacity * 1e-6,
            )

            list_parameters.append(
                {
                    "WASTANumber": hydropower_info["WASTANumber"],
                    "Name": relevant_stats_row["ZE-Name"].item(),
                    "Capacity": installed_capacity,
                    "Design discharge": design_discharge,
                    "Hydraulic head": hydraulic_head,
                    "Expected yearly generation": expected_generation,
                    "Expected winter generation": expected_winter_generation,
                    "Expected summer generation": expected_summer_generation,
                    "Percentage share CH": relevant_stats_row["Proz. Anteil CH"].item(),
                    "F": F,
                }
            )
            list_ds.append(ds)
            del ds
            print(
                f"\t{idx+1}/{nb_hp}, elapsed_time: {(time.time() - time_start)/60:.2f} minutes.",
                end="\r",
            )

        output_filepath = (
            self.path_data_hydro / "hydropower_generation"
            / f"{output_filename_prefix}.zarr"
        )
        concat_list_ds_and_save(list_ds, output_filepath)

        pd.DataFrame(list_parameters).to_csv(
            self.path_data_hydro / "hydropower_generation" / f"{output_filename_prefix}_parameters.csv",
            index=False,
        )

        print(f"\n\tTime elapsed: {(time.time() - time_start)/60:.2f} minutes.")

    def compute_hydropower_production_vectorized(
        self,
        accumulated_streamflow_per_polygon_filename: str = "ds_prevah_500_streamflow_accum_per_polygon_hourly.zarr",
        output_filename_prefix: str = "ds_prevah_500_hydropower_production_ror_vectorized",
        allowed_types: list[str] | None = None,
        timestep_hours: int = 1,
        use_sparse: bool = False,
    ) -> None:
        """Vectorized hydropower production using polygon->plant weight matrix and xr.dot.

        Parameters
        ----------
        accumulated_streamflow_per_polygon_filename : str
            Zarr dataset with accumulated polygon streamflow (hourly or daily).
        output_filename_prefix : str
            Prefix for output Zarr store.
        allowed_types : list[str] | None
            Restrict to these plant types (e.g. ["L"]). If None, use all.
        timestep_hours : int
            Length of timestep (1 for hourly, 24 for daily input).
        use_sparse : bool
            Use sparse incidence matrix if True (for very large polygon x plant matrix).
        """
        print("Vectorized hydropower production (start)")
        t0 = time.time()

        # Load hydropower polygons metadata
        if self.df_hydropower_polygons is None:
            self.df_hydropower_polygons = pd.read_json(
                self.path_data_hydro / "hydropower_polygons" / "df_hydropower_polygons.json",
                orient="records",
            )

        # Load polygon streamflow dataset (accumulated)
        if self.ds_accumulated_streamflow_polygon is None:
            self.ds_accumulated_streamflow_polygon = xr.open_zarr(
                self.path_data_prevah / accumulated_streamflow_per_polygon_filename
            )

        # Build parameter table (Step 1)
        hp_params_df = build_hydropower_parameter_table(
            self.df_stats_hydropower_ch,
            self.df_hydropower_polygons,
            allowed_types=allowed_types,
        )
        if hp_params_df.empty:
            print("No hydropower plants found after filtering; aborting.")
            return

        # Vectorized compute (Steps 2-5)
        output_dir = self.path_data_hydro / "hydropower_generation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{output_filename_prefix}.zarr"

        ds_vec = compute_hydropower_production_vectorized(
            self.ds_accumulated_streamflow_polygon,
            hp_params_df,
            variable="rgs",
            timestep_hours=timestep_hours,
            weights_sparse=use_sparse,
            output_path=output_path,
        )

        # Save parameter table for reference
        hp_params_df.to_csv(
            output_dir / f"{output_filename_prefix}_parameters.csv",
            index=False,
        )

        # Basic reporting
        print(
            f"Vectorized hydropower production saved: {output_path} | time elapsed: {(time.time() - t0)/60:.2f} min | plants: {ds_vec.dims.get('hydropower', 0)}"
        )

    def compute_monthly_bias_correction_factors(
        self,
        hydropower_generation_filename: str,
        output_filename: str = "ds_monthly_bias_correction_factors.nc",
    ) -> None:
        """Computes the monthly bias correction factors from monthly historical reported
        generation and a previously computed hydropower generation xarray Dataset. The
        correction factors are then replicated for each corresponding timestep in the hydropower
        generation xarray Dataset.

        Parameters
        ----------
        hydropower_generation_filename : str
            Name of file containing hydropower generation xarray Dataset
        output_filename : str, optional
            Name of output file that contains the monthly bias correction factors,
            replicated for each timestep in the hydropower generation xarray Dataset,
            by default "ds_monthly_bias_correction_factors.nc"
        """
        print("Computing monthly bias correction factors")
        time_start = time.time()
        self.df_reported_generation = pd.read_csv(
            self.path_data
            / "energy"
            / "ogd35_schweizerische_elektrizitaetsbilanz_monatswerte.csv"
        )

        ds_hydropower_generation = xr.open_dataset(
            self.path_data_hydro / "hydropower_generation" / hydropower_generation_filename
        ).sel(time=slice("2000", "2022"))
        df_reported_generation_ror = self.df_reported_generation[
            self.df_reported_generation.Jahr < 2023
        ][["Jahr", "Monat", "Erzeugung_laufwerk_GWh"]]
        df_reported_generation_ror["Erzeugung_laufwerk_GWh"] *= 1e-3  # to TWh

        list_ds = []
        hp_in_ds = ds_hydropower_generation.hydropower.to_numpy()
        for i in np.unique(ds_hydropower_generation.time.dt.year):
            wasta = self.gdf_hydropower_locations[
                (self.gdf_hydropower_locations["BeginningOfOperation"] <= i)
                & (self.gdf_hydropower_locations["WASTANumber"].isin(hp_in_ds))
            ]["WASTANumber"].tolist()
            list_ds.append(
                ds_hydropower_generation.sel(hydropower=wasta, time=str(i))
                .resample(time="M")
                .sum(["hydropower", "time"])
            )
        ds_hydropower_generation_per_month = xr.concat(list_ds, dim="time")

        df_reported_generation_ror["Estimated Generation"] = (
            ds_hydropower_generation_per_month.sel(
                time=slice("2000", None)
            ).gen.to_numpy()
        )
        df_reported_generation_ror["Relative Bias"] = (
            df_reported_generation_ror["Estimated Generation"]
            / df_reported_generation_ror["Reported Generation"]
        )
        df_reported_generation_ror_monthly_mean = df_reported_generation_ror.groupby(
            "Monat"
        ).mean()
        monthly_values = (
            1 / df_reported_generation_ror_monthly_mean["Relative Bias"]
        ).to_numpy()
        indices_months = ds_hydropower_generation.groupby("time.month").groups
        monthly_bias_correction_factors = np.empty(len(ds_hydropower_generation.time))
        for i, month in enumerate(indices_months):
            monthly_bias_correction_factors[indices_months[month]] = monthly_values[i]

        ds_monthly_bias_correction_factors = xr.DataArray(
            monthly_bias_correction_factors,
            dims=["time"],
            coords={"time": (["time"], ds_hydropower_generation.time.values)},
        )
        ds_monthly_bias_correction_factors.rename("bias_correction_factor").to_netcdf(
            self.path_data_hydro / "hydropower_generation" / output_filename
        )
        print(f"\tTime elapsed: {(time.time() - time_start)/60:.2f} minutes.")


if __name__ == "__main__":
    # Example execution block (adjust workers as appropriate for cluster/local machine)
    dask.config.set(num_workers=15, threads_per_worker=16)
    dask.config.set({
        "distributed.worker.memory.target": 0.7,  # Spill to disk at 70%
        "distributed.worker.memory.spill": 0.8,   # Start spilling at 80%
        "distributed.worker.memory.pause": 0.9,   # Pause worker at 90%
        "array.chunk-size": "512MiB",
        "logging.distributed": "error",           # Less verbose logs
    })


    data_processing = DataProcessingDask("paths.json")

    # --- Legacy pipeline (per-plant loops). Uncomment if still needed for comparison ---
    data_processing.extract_points_in_polygons()
    data_processing.compute_accumulated_streamflow_per_polygon()
    data_processing.get_catchment_area_per_hydropower()
    # data_processing.compute_hydropower_production(
    #     output_filename_prefix="ds_prevah_500_hydropower_production_ror"
    # )
    # hydropower_generation_dataset_filename = (
    #     "ds_prevah_500_hydropower_production_ror.nc"
    # )
    # monthly_bias_correction_factors_filename = (
    #     "ds_prevah_500_hydropower_production_ror_monthly_bias_correction_factors.nc"
    # )
    # data_processing.compute_monthly_bias_correction_factors(
    #     hydropower_generation_dataset_filename,
    #     monthly_bias_correction_factors_filename,
    # )

    # --- Vectorized hydropower production pipeline (preferred) ---
    # Requires precomputed polygon streamflow aggregated dataset (e.g., produced by mask aggregation)
    # Example placeholder paths; update to actual locations in your environment.
    data_processing.compute_hydropower_production_vectorized(
        streamflow_polygon_zarr=str(
            data_processing.path_data_hydro / "streamflow_polygons.zarr"
        ),
        hydropower_params_output=str(
            data_processing.path_data_hydro / "hydropower_params.parquet"
        ),
        hydropower_generation_output=str(
            data_processing.path_data_hydro / "hydropower_generation_vectorized.zarr"
        ),
        use_sparse=False,
    )
