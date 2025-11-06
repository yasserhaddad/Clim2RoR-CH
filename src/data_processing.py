import argparse
import json
import os
import time
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from src.utils_polygons import (
    build_prevah_grid_cells,
    build_prevah_grid_points,
    fill_missing_polygons_by_nearest,
    get_points_in_polygons,
    map_polygons_to_grid_by_intersection,
)
from src.utils_streamflow_hydropower import (
    aggregate_streamflow_with_mask,
    build_hydropower_parameter_table,
    compute_hydropower_production_vectorized,
    convert_mm_d_to_cubic_m_s,
    mode_of,
)
from src.var_attributes import ACCUM_HYDRO_ZARR_ENCODING
from src.data_analysis import compute_monthly_generation_from_ds

# Set environment variables after imports so import block remains contiguous
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["USE_PYGEOS"] = "1"

DEFAULT_EFFICIENCY = 0.8


def make_encoding(ds: xr.Dataset | xr.DataArray, compressor, chunk_dict):
    encoding = {}
    data_vars = ds.data_vars if isinstance(ds, xr.Dataset) else {ds.name: ds}
    for var in data_vars:
        dims = data_vars[var].dims
        chunks = tuple(chunk_dict.get(d, data_vars[var].sizes[d]) for d in dims)
        encoding[var] = {"compressor": compressor, "chunks": chunks}
    return encoding


class DataProcessingDask:
    def __init__(
        self,
        paths_file: str,
        climate_model_chain: str = None,
        climate_scenario: str = None,
        weighted_sum: bool = True,
    ):
        print("Loading data")
        paths = json.load(open(paths_file))
        self.path_data = Path(paths["path_data"])
        self.path_data_projections = Path(paths["path_data_projections"])
        self.climate_model_chain = climate_model_chain
        self.climate_scenario = climate_scenario

        if climate_model_chain is not None and climate_scenario is not None:
            self.path_data_prevah = (
                self.path_data_projections
                / "cordex_processed"
                / f"{self.climate_model_chain}_{self.climate_scenario}"
                / "prevah"
            )
        else:
            self.path_data_prevah = self.path_data / "prevah_obs"

        self.path_data_hydro = self.path_data / "hydropower"
        self.path_data_polygons = self.path_data / "polygons"

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
        self.weighted = weighted_sum
        self.grid_pt_in_polygon = None

        if climate_model_chain is None and climate_scenario is None:
            self.df_prevah_pts_in_polygons_filename = "df_prevah_obs_pts_in_polygons"
            self.accumulated_streamflow_per_polygon_filename = (
                "ds_prevah_obs_streamflow_accum_per_polygon"
            )
            self.hydropower_production_filename = (
                "ds_prevah_obs_hydropower_production_ror.zarr"
            )
        else:
            self.df_prevah_pts_in_polygons_filename = "df_prevah_proj_pts_in_polygons"
            self.accumulated_streamflow_per_polygon_filename = f"ds_prevah_{climate_model_chain}_{climate_scenario}_streamflow_accum_per_polygon"

            self.hydropower_production_filename = f"ds_prevah_{climate_model_chain}_{climate_scenario}_hydropower_production_ror.zarr"

        self.df_prevah_pts_in_polygons_filename += (
            "_weighted.csv" if weighted_sum else ".csv"
        )
        self.accumulated_streamflow_per_polygon_filename += (
            "_weighted.zarr" if weighted_sum else ".zarr"
        )

    def extract_points_in_polygons(
        self,
        prevah_grid_resolution: float = 500,
        output_filename: str = "df_prevah_500_pts_in_polygons.csv",
    ) -> None:
        """Extracts points from the PREVAH grid that are located in the polygons of Swiss waterbodies.

        Parameters
        ----------
        output_filename : str, optional
            Name of output file containing the DataFrame of the points present in each polygon,
            by default "df_prevah_500_pts_in_polygons.csv"
        """
        if output_filename is None:
            output_filename = self.df_prevah_pts_in_polygons_filename
        print("Extracting dataset grid points in polygons")
        time_start = time.time()
        # Load sample runoff data
        ds_sample = xr.open_zarr(self.path_data_prevah / "rgs.zarr", chunks="auto")
        gdf_runoff_points = build_prevah_grid_points(ds_sample, prevah_grid_resolution)
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
                    get_points_in_polygons, zip(split_gdfs, repeat(gdf_runoff_points))
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

        print(
            f"\tExtracted successfully all points in polygons! Time elapsed: {(time.time() - time_start)/60:.2f} minutes."
        )

    def extract_points_in_polygons_weighted(
        self,
        prevah_grid_resolution: float = 500,
        fraction_overlap: float = 0.01,
        max_distance_nearest: float = None,
        output_filename: str = "df_prevah_pts_in_polygons.csv",
    ) -> None:
        """Extracts points from the PREVAH grid that are located in the polygons of Swiss waterbodies.

        Parameters
        ----------
        output_filename : str, optional
            Name of output file containing the DataFrame of the points present in each polygon,
            by default "df_prevah_pts_in_polygons.csv"
        """
        if output_filename is None:
            output_filename = self.df_prevah_pts_in_polygons_filename

        print("Extracting dataset grid points in polygons")
        time_start = time.time()
        # Load sample runoff data
        ds_sample = xr.open_zarr(self.path_data_prevah / "rgs.zarr", chunks="auto")
        gdf_runoff_points = build_prevah_grid_points(ds_sample, prevah_grid_resolution)
        del ds_sample

        x_coords = np.sort(gdf_runoff_points.x.unique())
        y_coords = np.sort(gdf_runoff_points.y.unique())
        gdf_grid_cells = build_prevah_grid_cells(
            x_coords, y_coords, prevah_grid_resolution, crs="EPSG:2056"
        )
        gdf_grid_points = gpd.GeoDataFrame(
            {
                "x": gdf_grid_cells.x,
                "y": gdf_grid_cells.y,
                "geometry": gdf_grid_cells.geometry.centroid,
            },
            crs=gdf_grid_cells.crs,
        )

        # Get points in polygons and speed it up by launching multiple processes in parallel
        step = 500
        split_gdfs = [
            self.gdf_polygons.iloc[i : i + step]
            for i in range(0, len(self.gdf_polygons), step)
        ]
        print(len(split_gdfs), " polygon splits to process.")
        print("Total polygons:", len(self.gdf_polygons))

        num_workers = 20
        with Pool(num_workers) as p:
            list_gdfs = [
                ds
                for ds in p.starmap(
                    map_polygons_to_grid_by_intersection,
                    zip(split_gdfs, repeat(gdf_grid_cells), repeat(fraction_overlap)),
                )
            ]

        df_intersections = pd.DataFrame(
            pd.concat(list_gdfs, ignore_index=True)
        ).sort_values(by="EZGNR")
        print(df_intersections["EZGNR"].nunique(), "polygons with points.")

        # Fallback: assign nearest grid cell for polygons with no overlap
        polygons_with_weight = (
            set(df_intersections["EZGNR"].unique())
            if not df_intersections.empty
            else set()
        )
        missing_polygons = (
            set(self.gdf_polygons["EZGNR"].unique()) - polygons_with_weight
        )
        if missing_polygons:
            print(
                f"{len(missing_polygons)} polygons with no overlapping grid cells, assigning nearest grid cell as fallback."
            )
            df_nearest = fill_missing_polygons_by_nearest(
                self.gdf_polygons,
                gdf_grid_points,
                list(missing_polygons),
                max_distance=max_distance_nearest,
            )
            if not df_nearest.empty:
                df_nearest["cell_area"] = float(prevah_grid_resolution**2)
                df_nearest["intersect_area"] = df_nearest["cell_area"]
                df_nearest["weight"] = 1.0
                df_intersections = pd.concat(
                    [
                        df_intersections,
                        df_nearest[
                            ["x", "y", "EZGNR", "cell_area", "intersect_area", "weight"]
                        ],
                    ],
                    ignore_index=True,
                )

        # Build final CSV: keep one row per polygon×cell with fractional weight
        if df_intersections.empty:
            raise RuntimeError(
                "No polygon-cell intersections found (check CRS and grid coordinates)."
            )

        # Add an index_left matching previous expected column (unique cell index)
        df_intersections = df_intersections.reset_index(drop=True)
        df_intersections["index_left"] = df_intersections.index.astype(int)
        df_out = df_intersections.sort_values(by=["EZGNR", "y", "x"])
        print(
            df_out["EZGNR"].nunique(),
            "polygons assigned (area-weighted including nearest-fallback).",
        )
        df_out.to_csv(self.path_data_polygons / output_filename, index=False)
        self.df_prevah_pts_in_polygons = pd.read_csv(
            self.path_data_polygons / output_filename
        )

        print(
            f"\tExtracted successfully all polygon-cell weights! Time elapsed: {(time.time() - time_start)/60:.2f} minutes."
        )

    def create_prevah_polygon_grid(
        self,
        df_pts: pd.DataFrame,
        prevah_grid_resolution: float = 500,
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
        # ensure columns exist and correct dtypes
        if (
            "x" not in df_pts.columns
            or "y" not in df_pts.columns
            or "EZGNR" not in df_pts.columns
        ):
            raise ValueError(
                "df_prevah_pts_in_polygons must contain columns 'x', 'y' and 'EZGNR'"
            )

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
        new_x = (
            np.round(da.x.values / prevah_grid_resolution) * prevah_grid_resolution
        ).astype(da.x.dtype)
        new_y = (
            np.round(da.y.values / prevah_grid_resolution) * prevah_grid_resolution
        ).astype(da.y.dtype)
        da = da.assign_coords(x=new_x, y=new_y)

        self.grid_pt_in_polygon = da

        return da

    def create_polygon_weight_array(
        self,
        ds_rgs: xr.Dataset,
        df_pts: pd.DataFrame,
        output_filename: str = "prevah_polygon_weights.zarr",
    ) -> xr.DataArray:
        # align coords and build indices
        x_coords = np.sort(ds_rgs.x.values)
        y_coords = np.sort(ds_rgs.y.values)
        polygons = np.sort(df_pts["EZGNR"].unique())
        poly_index = {pid: i for i, pid in enumerate(polygons)}
        x_index = {float(x): i for i, x in enumerate(x_coords)}
        y_index = {float(y): i for i, y in enumerate(y_coords)}

        # Build dense weight array (y, x, polygon)
        W = np.zeros((len(y_coords), len(x_coords), len(polygons)), dtype="f4")
        for _, row in df_pts.iterrows():
            x = float(row["x"])
            y = float(row["y"])
            ezgnr = int(row["EZGNR"])
            w = float(row.get("weight", 1.0))
            xi = x_index.get(x)
            yi = y_index.get(y)
            pi = poly_index.get(ezgnr)
            if xi is None or yi is None or pi is None:
                continue
            W[yi, xi, pi] += w
        # Build xr.DataArray of weights aligned with ds_rgs
        weights_da = xr.DataArray(
            W,
            dims=("y", "x", "polygon"),
            coords={"y": y_coords, "x": x_coords, "polygon": polygons},
            name="weights",
        )

        if output_filename is not None:
            print("Saving polygon weights array to Zarr")
            chunks = {
                "x": ds_rgs.chunks["x"][0],
                "y": ds_rgs.chunks["y"][0],
                "polygon": 1000,
            }
            encoding = make_encoding(
                weights_da,
                compressor=ACCUM_HYDRO_ZARR_ENCODING["compressor"],
                chunk_dict=chunks,
            )
            with ProgressBar(dt=10):
                weights_da.chunk(chunks).to_zarr(
                    self.path_data_polygons / output_filename,
                    mode="w",
                    encoding=encoding,
                )

        self.grid_pt_in_polygon = weights_da

        return weights_da

    def compute_accumulated_streamflow_per_polygon(
        self,
        df_prevah_pts_in_polygons_filename: str = "df_prevah_proj_pts_in_polygons.csv",
        grid_fill_value: int = -1,
        prevah_grid_resolution: float = 500,
        output_filename: str = "ds_prevah_streamflow_accum_per_polygon",
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
        prevah_grid_resolution: float, optional
            Grid resolution (in m) of the input streamflow data to aggregate, by default
            500
        output_filename : str, optional
            Name of output file (without extension) containing the xarray Dataset of accumulated
            streamflow at each polygon, by default "ds_prevah_500_streamflow_accum_per_polygon"
        """
        time_start = time.time()
        if output_filename is None:
            output_filename = self.accumulated_streamflow_per_polygon_filename
        if df_prevah_pts_in_polygons_filename is None:
            df_prevah_pts_in_polygons_filename = self.df_prevah_pts_in_polygons_filename

        relevant_polygons = self.gdf_polygons.EZGNR.to_numpy()
        if self.df_prevah_pts_in_polygons is None:
            self.df_prevah_pts_in_polygons = pd.read_csv(
                self.path_data_polygons / df_prevah_pts_in_polygons_filename
            )

        # Converting streamflow to accumulated streamflow per polygon
        print("Computing accumulated streamflow per polygon")
        ds_rgs = xr.open_zarr(self.path_data_prevah / "rgs.zarr", chunks="auto")
        new_x = (
            np.round(ds_rgs.x.values / prevah_grid_resolution) * prevah_grid_resolution
        ).astype(ds_rgs.x.dtype)
        new_y = (
            np.round(ds_rgs.y.values / prevah_grid_resolution) * prevah_grid_resolution
        ).astype(ds_rgs.y.dtype)
        ds_rgs = ds_rgs.assign_coords(x=new_x, y=new_y)

        ds_rgs = ds_rgs.apply(
            lambda v: xr.apply_ufunc(
                convert_mm_d_to_cubic_m_s,
                v,
                prevah_grid_resolution**2,
                vectorize=True,
                dask="parallelized",
                output_dtypes=[v.dtype],
            )
        )

        chunks_spatial = {
            "x": mode_of(ds_rgs.chunks["x"])[0],
            "y": mode_of(ds_rgs.chunks["y"])[0],
        }
        if self.df_prevah_pts_in_polygons is not None:
            if self.weighted:
                print("Using fractional weights for polygon-grid cell mapping.")
                if self.grid_pt_in_polygon is None:
                    self.create_polygon_weight_array(
                        ds_rgs,
                        self.df_prevah_pts_in_polygons.copy(),
                        output_filename=None,
                    )
            else:
                print(
                    "No weights found, assuming uniform weights for polygon-grid cell mapping."
                )
                if self.grid_pt_in_polygon is None:
                    self.create_prevah_polygon_grid(
                        df_pts=self.df_prevah_pts_in_polygons.copy(),
                        fill_value=grid_fill_value,
                    )
        if self.weighted:
            self.grid_pt_in_polygon = self.grid_pt_in_polygon.chunk(
                {
                    "x": ds_rgs.chunks["x"][0],
                    "y": ds_rgs.chunks["y"][0],
                    "polygon": 1000,
                }
            )
        else:
            self.grid_pt_in_polygon = self.grid_pt_in_polygon.chunk(chunks_spatial)

        with ProgressBar(dt=10):
            if self.weighted:
                accum_vars = {
                    var: xr.dot(
                        ds_rgs[var].fillna(0), self.grid_pt_in_polygon, dims=("y", "x")
                    )
                    for var in ds_rgs.data_vars
                }
                ds_accum = xr.Dataset(accum_vars)
            else:
                ds_accum = aggregate_streamflow_with_mask(
                    ds_rgs,
                    self.grid_pt_in_polygon,
                    method="sum",
                    fill_value=grid_fill_value,
                    polygons=relevant_polygons,
                )

            ds_accum = ds_accum.sel(
                time=~((ds_accum.time.dt.month == 2) & (ds_accum.time.dt.day == 29))
            )

            chunks_accum = {"time": 365, "polygon": 1000}
            encoding = make_encoding(
                ds_accum,
                compressor=ACCUM_HYDRO_ZARR_ENCODING["compressor"],
                chunk_dict=chunks_accum,
            )
            encoding["time"] = {
                "units": f"seconds since {np.datetime_as_string(ds_accum.time[0].values)}"
            }

            output_filepath = self.path_data_prevah / f"{output_filename}"
            ds_accum.chunk(chunks_accum).to_zarr(
                output_filepath, mode="w", encoding=encoding
            )

            new_end_time = ds_accum.time[-1] + np.timedelta64(23, "h")
            ds_accum_hourly = ds_accum.reindex(
                time=pd.date_range(
                    start=ds_accum.time[0].values,
                    end=new_end_time.values,
                    freq="1h",
                    inclusive="both",
                ),
                method="ffill",
            )
            ds_accum_hourly = ds_accum_hourly.sel(
                time=~(
                    (ds_accum_hourly.time.dt.month == 2)
                    & (ds_accum_hourly.time.dt.day == 29)
                )
            )
            chunks_accum = {"time": 8760, "polygon": 1000}
            encoding = make_encoding(
                ds_accum_hourly,
                compressor=ACCUM_HYDRO_ZARR_ENCODING["compressor"],
                chunk_dict=chunks_accum,
            )
            encoding["time"] = {
                "units": f"seconds since {np.datetime_as_string(ds_accum_hourly.time[0].values)}"
            }
            output_filepath = (
                self.path_data_prevah
                / f"{output_filename.remove_suffix('.zarr')}_hourly.zarr"
            )
            ds_accum_hourly.chunk(chunks_accum).to_zarr(
                output_filepath, mode="w", encoding=encoding
            )
            self.ds_accumulated_streamflow_polygon = ds_accum_hourly

            print(
                f"\tSaved accumulated streamflow per polygon to Zarr: {output_filepath}"
            )

        print(
            f"\tTotal time for accumulated streamflow: {(time.time() - time_start)/60:.2f} minutes."
        )

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

    def build_polygon_plant_weights(
        self,
        polygon_col="polygons_full",
        dtype="uint8",
    ) -> xr.DataArray:
        if self.ds_accumulated_streamflow_polygon is None:
            self.ds_accumulated_streamflow_polygon = xr.open_zarr(
                self.path_data_prevah
                / self.accumulated_streamflow_per_polygon_filename,
                chunks="auto",
            )
        polygons_all = self.ds_accumulated_streamflow_polygon.polygon.values
        hp_param_df = build_hydropower_parameter_table(
            self.df_hydropower_polygons, self.df_hydropower_polygons
        )
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
            dims=("polygon", "hydropower"),
            coords={
                "polygon": polygons_all,
                "hydropower": hp_param_df["WASTANumber"].values,
            },
            name="weights",
        )

    def compute_hydropower_production_vectorized(
        self,
        accumulated_streamflow_per_polygon_filename: str = "ds_prevah_500_streamflow_accum_per_polygon_hourly.zarr",
        output_filename: str = "ds_prevah_500_hydropower_production_ror_vectorized.zarr",
        allowed_types: list[str] | None = None,
        timestep_hours: int = 1,
        use_simplified_efficiency: bool = False,
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
        use_simplified_efficiency : bool
            Whether to use simplified efficiency calculation or default efficiency (0.8)
        """
        print("Vectorized hydropower production (start)")
        t0 = time.time()
        if accumulated_streamflow_per_polygon_filename is None:
            accumulated_streamflow_per_polygon_filename = (
                self.accumulated_streamflow_per_polygon_filename
            )
        if output_filename is None:
            output_filename = self.hydropower_production_filename

        # Load hydropower polygons metadata
        if self.df_hydropower_polygons is None:
            self.df_hydropower_polygons = pd.read_json(
                self.path_data_hydro
                / "hydropower_polygons"
                / "df_hydropower_polygons.json",
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
        output_path = output_dir / output_filename
        var_name = list(self.ds_accumulated_streamflow_polygon.data_vars.keys())[0]
        with ProgressBar(dt=10):
            ds_vec = compute_hydropower_production_vectorized(
                self.ds_accumulated_streamflow_polygon,
                hp_params_df,
                variable=var_name,
                timestep_hours=timestep_hours,
                use_simplified_efficiency=use_simplified_efficiency,
                output_path=output_path,
            )

        # Save parameter table for reference
        hp_params_df.to_csv(
            output_dir / f"{output_filename.split('.')[0]}_parameters.csv",
            index=False,
        )

        # Basic reporting
        print(
            f"Vectorized hydropower production saved: {output_path} | time elapsed: {(time.time() - t0)/60:.2f} min | plants: {ds_vec.sizes.get('hydropower', 0)}"
        )

    def compute_monthly_bias_correction_factors(
        self,
        method: str,
        output_filename: str = "ds_monthly_bias_correction_factors.nc",
    ) -> None:
        """Computes the monthly bias correction factors from monthly historical reported
        generation and a previously computed hydropower generation xarray Dataset. The
        correction factors are then replicated for each corresponding timestep in the hydropower
        generation xarray Dataset.

        Parameters
        ----------
        method : str
            Method to compute the bias correction factors. Can be either
            "per_timestep" or "monthly_mean".
        output_filename : str, optional
            Name of output file that contains the monthly bias correction factors,
            replicated for each timestep in the hydropower generation xarray Dataset,
            by default "ds_monthly_bias_correction_factors.nc"
        """
        if method not in ["per_timestep", "monthly_mean"]:
            raise ValueError("method must be either 'per_timestep' or 'monthly_mean'")
        print("Computing monthly bias correction factors")
        time_start = time.time()
        self.df_reported_generation = pd.read_csv(
            self.path_data
            / "energy"
            / "ogd35_schweizerische_elektrizitaetsbilanz_monatswerte.csv"
        )
        start_year = self.df_reported_generation.Jahr.min()
        end_year = self.df_reported_generation[
            self.df_reported_generation.Monat == 12
        ].Jahr.max()

        ds_hydropower_generation = xr.open_zarr(
            self.path_data_hydro
            / "hydropower_generation"
            / self.hydropower_production_filename,
            chunks="auto",
        )
        df_reported_generation_ror = self.df_reported_generation[
            self.df_reported_generation.Jahr <= end_year
        ][["Jahr", "Monat", "Erzeugung_laufwerk_GWh"]]
        df_reported_generation_ror["Erzeugung_laufwerk_GWh"] *= 1e-3  # to TWh
        df_reported_generation_ror = df_reported_generation_ror.rename(
            columns={"Erzeugung_laufwerk_GWh": "Reported Generation"}
        )
        ds_hydropower_generation_per_month = compute_monthly_generation_from_ds(
            ds_hydropower_generation, self.gdf_hydropower_locations,
            resample_rule="ME",
            start_year=start_year,
            end_year=end_year,
        )

        df_reported_generation_ror["Estimated Generation"] = (
            ds_hydropower_generation_per_month.sel(
                time=slice(str(start_year), None)
            ).gen.to_numpy()
        )

        if method == "per_timestep":
            df_reported_generation_ror["Relative Bias"] = (
                df_reported_generation_ror["Estimated Generation"]
                / df_reported_generation_ror["Reported Generation"]
            )
            df_reported_generation_ror_monthly_mean = (
                df_reported_generation_ror.groupby("Monat").mean()
            )
            monthly_values = (
                1 / df_reported_generation_ror_monthly_mean["Relative Bias"]
            ).to_numpy()
        else:  # method == "monthly_mean"
            df_monthly_means = df_reported_generation_ror.groupby(
                df_reported_generation_ror["Monat"]
            ).mean()
            df_monthly_means["Relative Bias"] = (
                df_monthly_means["Estimated Generation"]
                / df_monthly_means["Reported Generation"]
            )
            monthly_values = (1 / df_monthly_means["Relative Bias"]).to_numpy()

        indices_months = ds_hydropower_generation.groupby("time.month").groups
        monthly_bias_correction_factors = np.empty(len(ds_hydropower_generation.time))
        for i, month in enumerate(indices_months):
            monthly_bias_correction_factors[indices_months[month]] = monthly_values[i]

        ds_monthly_bias_correction_factors = xr.DataArray(
            monthly_bias_correction_factors,
            dims=["time"],
            coords={"time": (["time"], ds_hydropower_generation.time.values)},
        )
        chunks = {"time": ds_hydropower_generation.chunks["time"][0]}
        ds_monthly_bias_correction_factors.rename("bias_correction_factor").chunk(
            chunks
        ).to_zarr(
            self.path_data_hydro / "hydropower_generation" / output_filename,
            encoding={
                "bias_correction_factor": {
                    "compressor": ACCUM_HYDRO_ZARR_ENCODING["compressor"],
                    "chunks": tuple(chunks.values()),
                }
            },
            mode="w",
        )
        print(f"\tTime elapsed: {(time.time() - time_start)/60:.2f} minutes.")


if __name__ == "__main__":
    dask.config.set(num_workers=15, threads_per_worker=16)
    dask.config.set(
        {
            "distributed.worker.memory.target": 0.7,  # Spill to disk at 70%
            "distributed.worker.memory.spill": 0.8,  # Start spilling at 80%
            "distributed.worker.memory.pause": 0.9,  # Pause worker at 90%
            "array.chunk-size": "512MiB",
            "logging.distributed": "error",  # Less verbose logs
        }
    )

    parser = argparse.ArgumentParser(
        description="Run PREVAH -> hydropower processing. If climate_model_chain and climate_scenario are omitted, observational data are used."
    )
    parser.add_argument(
        "--climate_model_chain",
        "-c",
        type=str,
        default=None,
        help="Climate model chain identifier (e.g. CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1). Omit for observations.",
    )
    parser.add_argument(
        "--climate_scenario",
        "-s",
        type=str,
        default=None,
        help="Climate scenario (e.g. rcp85). Omit for observations.",
    )
    parser.add_argument(
        "--weighted_sum",
        "-w",
        action="store_true",
        help="Use weighted sum for polygon-grid cell mapping.",
    )

    args = parser.parse_args()
    climate_model_chain = args.climate_model_chain
    climate_scenario = args.climate_scenario
    weighted_sum = args.weighted_sum
    prevah_grid_resolution = 500

    if climate_model_chain is None and climate_scenario is None:
        df_prevah_pts_in_polygons_filename = "df_prevah_obs_pts_in_polygons.csv"
        accumulated_streamflow_per_polygon_filename = (
            "ds_prevah_obs_streamflow_accum_per_polygon"
        )
        accumulated_streamflow_per_polygon_filename += (
            "_weighted.zarr" if weighted_sum else ".zarr"
        )
        hydropower_production_filename = "ds_prevah_obs_hydropower_production_ror.zarr"
    else:
        df_prevah_pts_in_polygons_filename = "df_prevah_proj_pts_in_polygons.csv"
        accumulated_streamflow_per_polygon_filename = f"ds_prevah_{climate_model_chain}_{climate_scenario}_streamflow_accum_per_polygon"
        accumulated_streamflow_per_polygon_filename += (
            "_weighted.zarr" if weighted_sum else ".zarr"
        )
        hydropower_production_filename = f"ds_prevah_{climate_model_chain}_{climate_scenario}_hydropower_production_ror.zarr"

    data_processing = DataProcessingDask(
        "paths_projections.json",
        climate_model_chain,
        climate_scenario,
        weighted_sum=weighted_sum,
    )
    # if data_processing.weighted:
    #     print("Using weighted sum for polygon-grid cell mapping.")
    #     data_processing.extract_points_in_polygons_weighted(
    #         prevah_grid_resolution=prevah_grid_resolution,
    #         fraction_overlap=0.001,
    #         max_distance_nearest=2000,
    #         output_filename=df_prevah_pts_in_polygons_filename.removesuffix(".csv")
    #         + "_weighted.csv",
    #     )
    # else:
    #     print("Using uniform weights for polygon-grid cell mapping.")
    #     data_processing.extract_points_in_polygons(
    #         output_filename=df_prevah_pts_in_polygons_filename
    #     )

    # data_processing.compute_accumulated_streamflow_per_polygon(
    #     output_filename=accumulated_streamflow_per_polygon_filename,
    # )
    # data_processing.compute_hydropower_production_vectorized(
    #     accumulated_streamflow_per_polygon_filename=accumulated_streamflow_per_polygon_filename.removesuffix(
    #         ".zarr"
    #     )
    #     + "_hourly.zarr",
    #     output_filename=hydropower_production_filename,
    #     allowed_types=["L"],
    #     timestep_hours=1,
    #     use_simplified_efficiency=True,
    # )
    data_processing.compute_monthly_bias_correction_factors(
        method="per_timestep" if climate_model_chain is None else "monthly_mean",
        output_filename=hydropower_production_filename.removesuffix(".zarr")
        + "_monthly_bias_correction_factors.zarr",
    )
