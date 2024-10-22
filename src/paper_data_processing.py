import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

os.environ["USE_PYGEOS"] = "0"
from itertools import repeat
from multiprocessing import Pool

import geopandas as gpd
from extract_runoff_prevah import (
    batch_extraction_prevah,
    transform_coords_old_to_new_swiss,
)
from utils_polygons import (
    find_upstream_polygons_recursive,
    flatten_list,
    get_points_in_polygons,
)
from utils_streamflow_hydropower import (
    GRAVITY,
    WATER_DENSITY,
    compute_ds_hydropower_generation_from_streamflow,
    compute_simplified_efficiency_term,
    compute_streamflow_aggregate_polygons_parallel,
    concat_list_ds_and_save,
    convert_mm_d_to_cubic_m_s,
    get_beta_coeff,
)
from var_attributes import ACCUM_HYDRO_NETCDF_ENCODINGS

LEAP_YEARS = [str(year) for year in [1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]]
DATES_TO_REMOVE = [
    date
    for year in LEAP_YEARS
    for date in pd.date_range(
        start=f"{year}-02-29 00:00",
        end=f"{year}-03-01 00:00",
        freq="1H",
        inclusive="left",
    )
]
DEFAULT_EFFICIENCY = 0.8


class DataProcessing:
    def __init__(self, paths_file: str):
        paths = json.load(open(paths_file))
        self.path_data = Path(paths["path_data"])

        self.path_data_prevah = self.path_data / "prevah"
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
        # Load sample runoff data
        ds_sample = xr.open_dataset(
                list((self.path_data_prevah / "netcdf").glob("*.nc"))[0]
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

        print("Extracted successfully all points in polygons!")

        df_concat = pd.DataFrame(
            pd.concat(list_gdfs, ignore_index=True).drop(columns="geometry")
        ).sort_values(by="EZGNR")

        df_concat[["index_left", "EZGNR", "TEILEZGNR", "y", "x"]].to_csv(
            self.path_data_polygons / output_filename, index=False
        )
        self.df_prevah_pts_in_polygons = pd.read_csv(
            self.path_data_polygons / output_filename
        )

    def compute_accumulated_streamflow_per_polygon(
        self,
        df_prevah_pts_in_polygons_filename: str = "df_prevah_500_pts_in_polygons.csv",
        output_filename: str = "ds_prevah_500_streamflow_accum_per_polygon",
    ) -> None:
        """Compute accumulated streamflow at each polygon.

        Parameters
        ----------
        df_prevah_pts_in_polygons_filename : str, optional
            Name of file containing the DataFrame of PREVAH grid points present in
            each polygon, by default "df_prevah_500_pts_in_polygons.csv"
        output_filename : str, optional
            Name of output file (without extension) containing the xarray Dataset of accumulated
            streamflow at each polygon, by default "ds_prevah_500_streamflow_accum_per_polygon"
        """
        relevant_polygons = self.gdf_polygons.EZGNR.to_numpy()
        if not self.df_prevah_pts_in_polygons:
            self.df_prevah_pts_in_polygons = pd.read_csv(
                self.path_data_polygons / df_prevah_pts_in_polygons_filename
            )

        # Converting streamflow to accumulated streamflow per polygon
        print("Begin computing accumulated streamflow")
        list_ds_accum_streamflow = []

        time_start = time.time()
        for path_rgs in list((self.path_data_prevah / "netcdf").glob("*")):
            year = path_rgs.stem.split("Mob500_RGS_")[1]
            ds_rgs = xr.open_dataset(path_rgs).apply(
                lambda v: convert_mm_d_to_cubic_m_s(v, 500 * 500)
            )
            list_ds_accum_streamflow.append(
                compute_streamflow_aggregate_polygons_parallel(
                    ds_rgs,
                    self.gdf_polygons,
                    self.df_prevah_pts_in_polygons,
                    relevant_polygons,
                )
            )
            print(
                f"Year: {year}, time elapsed: {(time.time() - time_start)/60:.2f} minutes.",
                end="\r",
            )

        ds_accum = xr.concat(list_ds_accum_streamflow, "time")
        encoding = {
            var: ACCUM_HYDRO_NETCDF_ENCODINGS.copy()
            for var in list(ds_accum.data_vars.keys())
        }
        encoding["time"] = {
            "units": f"seconds since {np.datetime_as_string(ds_accum.time[0].values)}"
        }

        output_filepath = self.path_data_prevah / f"{output_filename}.nc"
        if output_filepath.is_file():
            output_filepath.unlink()

        ds_accum.to_netcdf(output_filepath, mode="w", encoding=encoding)

        new_end_time = ds_accum.time[-1] + np.timedelta64(23, "h")
        ds_accum_hourly = ds_accum.reindex(
            time=pd.date_range(
                start=ds_accum.time[0].values,
                end=new_end_time.values,
                freq="1H",
                inclusive="both",
            ),
            method="ffill",
        ).drop_sel(time=DATES_TO_REMOVE)

        output_filepath = self.path_data_prevah / f"{output_filename}_hourly.nc"
        if output_filepath.is_file():
            output_filepath.unlink()

        ds_accum_hourly.to_netcdf(output_filepath, mode="w", encoding=encoding)
        self.ds_accumulated_streamflow_polygon = ds_accum_hourly.load()

        print(
            f"Total time for accumulated streamflow: {(time.time() - time_start)/60:.2f} minutes."
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
            (self.df_new_hydropower_polygons["Checked"] == True)
            & (self.df_new_hydropower_polygons["To change"] == True)
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

    def compute_hydropower_production(
        self,
        accumulated_streamflow_per_polygon_filename: str = "ds_prevah_500_streamflow_accum_per_polygon_hourly.nc",
        output_filename_prefix="ds_prevah_500_hydropower_production_ror",
    ) -> None:
        """Compute the hydropower production by converting the accumulated streamflow at the polygon assigned
        as water intake point of the hydropower plant into energy quantities.

        Parameters
        ----------
        accumulated_streamflow_per_polygon_filename : str, optional
            Name of netcdf file containing the accumulated streamflow at each polygon
            for every time step, by default "ds_prevah_500_streamflow_accum_per_polygon_hourly.nc"
        output_filename_prefix : str, optional
            Prefix to add to each filename of the outputs to save, by default
            "ds_prevah_500_hydropower_production_ror"
        """
        # Load hydropower polygons
        if not self.df_hydropower_polygons:
            self.df_hydropower_polygons = pd.read_json(
                self.path_data_hydro
                / "hydropower_polygons"
                / "df_hydropower_polygons.json",
                orient="records",
            )
        # Load accumulated streamflow per polygon
        if not self.ds_accumulated_streamflow_polygon:
            self.ds_accumulated_streamflow_polygon = xr.open_dataset(
                self.path_data_prevah / accumulated_streamflow_per_polygon_filename
            ).load()

        # --------------------------------------------------------------------------------------------------
        # Converting accumulated streamflow into hydropower generation
        print("Begin computing hydropower generation")

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
        list_ds_with_beta = []
        list_ds_with_beta_less_than_1 = []
        list_parameters = []

        for idx, (_, hydropower_info) in enumerate(df_hydropower_to_process.iterrows()):
            relevant_polygons = (
                hydropower_info["EZGNR"] + hydropower_info["upstream_EZGNR"]
            )
            relevant_stats_row = [
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
                    f"\n{hydropower_info['WASTANumber']}\t{installed_capacity}\t{design_discharge}"
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

            beta_coeff = (
                round(get_beta_coeff(ds, expected_generation * 1e-3), 2)
                if expected_generation != 0
                else 1
            )
            ds_beta = compute_ds_hydropower_generation_from_streamflow(
                self.ds_accumulated_streamflow_polygon,
                hydropower_info["WASTANumber"],
                relevant_polygons,
                hydraulic_head,
                DEFAULT_EFFICIENCY,
                simplified_efficiency=F * beta_coeff,
                design_discharge=design_discharge,
                installed_capacity=installed_capacity * 1e-6,  # to TW
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
                    "beta_coeff": beta_coeff,
                }
            )
            list_ds.append(ds)
            list_ds_with_beta.append(ds_beta)
            list_ds_with_beta_less_than_1.append(ds if beta_coeff > 1 else ds_beta)
            del ds, ds_beta
            print(
                f"{idx+1}/{nb_hp}, elapsed_time: {(time.time() - time_start)/60:.2f} minutes.",
                end="\r",
            )

            output_filepath = (
                self.path_data_hydro / "hydropower_generation"
                / f"{output_filename_prefix}_simplified_efficiency.nc"
            )
            concat_list_ds_and_save(list_ds, output_filepath)

            output_filepath = (
                self.path_data_hydro / "hydropower_generation"
                / f"{output_filename_prefix}_simplified_efficiency_with_beta.nc"
            )
            concat_list_ds_and_save(list_ds_with_beta, output_filepath)

            output_filepath = (
                self.path_data_hydro / "hydropower_generation"
                / f"{output_filename_prefix}_simplified_efficiency_with_beta_less_than_1.nc"
            )
            concat_list_ds_and_save(list_ds_with_beta_less_than_1, output_filepath)

            pd.DataFrame(list_parameters).to_csv(
                self.path_data_hydro / "hydropower_generation" / f"{output_filename_prefix}_parameters.csv",
                index=False,
            )

            print(f"\nTime elapsed: {(time.time() - time_start)/60:.2f} minutes.")

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
        self.df_reported_generation = pd.read_csv(
            self.path_data
            / "energy"
            / "ogd35_schweizerische_elektrizitaetsbilanz_monatswerte.csv"
        )

        ds_hydropower_generation = xr.open_dataset(
            self.path_data_hydro / "hydropower_generation" / hydropower_generation_filename
        )
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


if __name__ == "__main__":
    data_processing = DataProcessing("../paths.json")
    data_processing.convert_bin_to_netcdf_runoff_prevah()
    data_processing.extract_points_in_polygons()
    data_processing.compute_accumulated_streamflow_per_polygon()
    data_processing.get_catchment_area_per_hydropower()
    data_processing.compute_hydropower_production()

    hydropower_generation_dataset_filename = (
        "ds_prevah_500_hydropower_production_ror_simplified_efficiency.nc"
    )
    monthly_bias_correction_factors_filename = (
        "ds_prevah_500_hydropower_production_ror_simplified_efficiency_monthly_bias_correction_factors.nc"
    )
    data_processing.compute_monthly_bias_correction_factors(
        hydropower_generation_dataset_filename, monthly_bias_correction_factors_filename
    )
