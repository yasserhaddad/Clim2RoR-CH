import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

os.environ["USE_PYGEOS"] = "0"
import geopandas as gpd

MIN_CAPACITY = 5  # Minimum capacity for power plant to be stand alone, otherwise aggregated at bus level


class MatchWastaNexuse:
    def __init__(
        self,
        paths_file: str,
        hydropower_generation_filename: str,
        nexuse_db_filename: str,
    ):
        paths = json.load(open(paths_file))
        self.path_data = Path(paths["path_data"])
        self.path_data_nexuse = Path(paths["path_data_nexuse"])
        self.path_data_hydro = self.path_data / "hydropower"

        self.nexuse_db_filename = nexuse_db_filename
        self.ds_hydropower_generation = xr.open_dataset(
            self.path_data_hydro
            / "hydropower_generation"
            / hydropower_generation_filename
        )

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
                '("LGENERATOR[MW]"/100)*"PROZ.ANTEILCH"',
                '("PROD.OHNEUMWÄLZBETRIEB-J."/100)*"PROZ.ANTEILCH"',
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
                '("LGENERATOR[MW]"/100)*"PROZ.ANTEILCH"': "Capacity",
                '("PROD.OHNEUMWÄLZBETRIEB-J."/100)*"PROZ.ANTEILCH"': "Expected Annual Gen (GWh)",
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

        self.df_nexuse_profiles = pd.read_excel(
            self.path_data_nexuse / "database" / nexuse_db_filename,
            sheet_name="profiles",
            header=2,
        )
        df_nexuse_bus = pd.read_excel(
            self.path_data_nexuse / "database" / nexuse_db_filename,
            sheet_name="bus",
            header=2,
        )
        self.gdf_nexuse_bus = gpd.GeoDataFrame(
            df_nexuse_bus,
            geometry=gpd.points_from_xy(
                df_nexuse_bus["coord_x"], df_nexuse_bus["coord_y"]
            ),
            crs="EPSG:21781",
        ).to_crs("EPSG:2056")
        self.df_nexuse_gens = pd.read_excel(
            self.path_data_nexuse / "database" / nexuse_db_filename,
            sheet_name="gens",
            header=2,
        ).dropna(subset=["Gen_ID"])
        self.df_nexuse_gens["GenNum"] = self.df_nexuse_gens["GenNum"].astype(int)

    def create_wasta_nexuse_ror_gens_db(self) -> pd.DataFrame:
        """Creates a pandas DataFrame containing rows of RoR power plants information,
        updated with the WASTA database and compliant with the Nexus-e database format.

        Returns
        -------
        pd.DataFrame
            DataFrame containing updated RoR power plants information and compliant with
            the Nexus-e database format.
        """
        df_nexuse_gens_hydro = self.df_nexuse_gens[
            self.df_nexuse_gens["Gen_ID"].str.startswith("CH_Hydro")
        ]
        df_nexuse_gens_ch = self.df_nexuse_gens[self.df_nexuse_gens["Country"] == "CH"]
        df_nexuse_gens_ch_ror = df_nexuse_gens_ch[df_nexuse_gens_ch["SubType"] == "RoR"]
        wasta_to_exclude = self.df_stats_hydropower_ch[
            (
                self.df_stats_hydropower_ch["ZE-Name"].isin(
                    df_nexuse_gens_ch.loc[
                        df_nexuse_gens_ch["SubType"] == "Dam",
                        [
                            "ZE-Name Wasserstatistik_1",
                            "ZE-Name Wasserstatistik_2",
                            "ZE-Name Wasserstatistik_3",
                            "ZE-Name Wasserstatistik_4",
                        ],
                    ].values.flatten()
                )
            )
            & (self.df_stats_hydropower_ch["WKA-Typ"] == "L")
        ]

        gdf_ror_loc_with_bus = (
            gpd.sjoin_nearest(
                self.gdf_hydropower_locations[
                    (
                        self.gdf_hydropower_locations["WASTANumber"].isin(
                            self.ds_hydropower_generation.hydropower.values
                        )
                    )
                    & (
                        ~self.gdf_hydropower_locations["WASTANumber"].isin(
                            wasta_to_exclude["ZE-Nr"]
                        )
                    )
                ],
                self.gdf_nexuse_bus.loc[
                    self.gdf_nexuse_bus["node_id"].isin(df_nexuse_gens_hydro["NodeId"]),
                    [
                        "TO USE Node Codes",
                        "TO USE Node Names",
                        "node_id",
                        "node_number",
                        "Canton Name",
                        "canton",
                        "geometry",
                    ],
                ],
                distance_col="dist_node",
            )
            .rename(
                columns={
                    "Name": "name",
                    "Capacity": "P_gen_max in 2015 (MW)",
                    "BeginningOfOperation": "start_year",
                    "EndOfOperation": "end_year (50yrNucl)",
                    "TO USE Node Codes": "NodeCode",
                    "TO USE Node Names": "NodeName",
                    "node_id": "NodeId",
                    "node_number": "AfemNodeNum",
                    "canton": "CantonCode",
                    "Canton": "Canton Plant",
                    "Canton Name": "Canton",
                }
            )
            .drop(columns=["index_right"])
        )

        columns_to_update = [
            "AfemNodeNum",
            "NodeCode",
            "NodeName",
            "NodeId",
            "Canton",
            "CantonCode",
            "P_gen_max in 2015 (MW)",
            "start_year",
            "end_year (50yrNucl)",
            "Expected Annual Gen (GWh)",
        ]

        # Stand alone RoR power plants
        df_nexuse_gens_ch_ror_updated = []
        for i, row in (
            gdf_ror_loc_with_bus[(gdf_ror_loc_with_bus["P_gen_max in 2015 (MW)"] >= 5)]
            .reset_index(drop=True)
            .iterrows()
        ):
            new_row_gens = df_nexuse_gens_ch_ror.iloc[0].to_dict()
            new_row_gens["GenNum"] += i
            new_row_gens["idProfile"] += i
            new_row_gens["ZE-Name Wasserstatistik_1"] = row["name"]
            new_row_gens["name"] = row["name"].replace(" ", "")
            new_row_gens["Gen_ID"] = f'CH_Hydro_RoR_{new_row_gens["name"]}'
            new_row_gens["WASTANumber"] = row["WASTANumber"]
            for col in columns_to_update:
                new_row_gens[col] = row[col]
                if col == "start_year":
                    new_row_gens[col] = (
                        new_row_gens[col] if new_row_gens[col] > 2017 else 2012
                    )
                if col == "end_year (50yrNucl)":
                    new_row_gens[col] = 2100

            df_nexuse_gens_ch_ror_updated.append(new_row_gens)

        df_nexuse_gens_ch_ror_updated = pd.DataFrame(df_nexuse_gens_ch_ror_updated)

        # RoR power plants aggregated at node level
        unique_item = lambda list_items: np.unique(list_items).item()
        df_aggregated_nodes = (
            gdf_ror_loc_with_bus[
                (gdf_ror_loc_with_bus["P_gen_max in 2015 (MW)"] < MIN_CAPACITY)
            ]
            .groupby("NodeId")
            .agg(
                {
                    "WASTANumber": list,
                    "name": list,
                    "P_gen_max in 2015 (MW)": sum,
                    "NodeName": unique_item,
                    "NodeCode": unique_item,
                    "AfemNodeNum": unique_item,
                    "Canton": unique_item,
                    "CantonCode": unique_item,
                    "start_year": min,
                    "Expected Annual Gen (GWh)": sum,
                    "dist_node": "mean",
                }
            )
            .reset_index()
        )

        df_nexuse_gens_ch_ror_agg_updated = []
        for i, row in (
            df_aggregated_nodes[(df_aggregated_nodes["P_gen_max in 2015 (MW)"] >= 5)]
            .reset_index(drop=True)
            .iterrows()
        ):
            new_row_gens = df_nexuse_gens_ch_ror_updated.iloc[-1].to_dict()
            new_row_gens["GenNum"] += i + 1
            new_row_gens["idProfile"] += i + 1
            new_row_gens["ZE-Name Wasserstatistik_1"] = "~~AGGREGATED~~"
            new_row_gens["Notes"] = ", ".join(row["name"])
            new_row_gens["name"] = f"Agg_{row['NodeId'].replace('CH_', '')}"
            new_row_gens["Gen_ID"] = (
                f'CH_Hydro_RoR_Agg_{row["NodeId"].replace("CH_", "")}'
            )
            new_row_gens["WASTANumber"] = ", ".join(
                [str(nb) for nb in row["WASTANumber"]]
            )
            for col in columns_to_update:
                if col == "end_year (50yrNucl)":
                    new_row_gens[col] = 2100
                else:
                    new_row_gens[col] = row[col]
                if col == "start_year":
                    new_row_gens[col] = (
                        new_row_gens[col] if new_row_gens[col] > 2017 else 2012
                    )

            df_nexuse_gens_ch_ror_agg_updated.append(new_row_gens)

        df_nexuse_gens_ch_ror_agg_updated = pd.DataFrame(
            df_nexuse_gens_ch_ror_agg_updated
        )

        df_nexuse_gens_ch_ror_updated = pd.concat(
            [df_nexuse_gens_ch_ror_updated, df_nexuse_gens_ch_ror_agg_updated]
        ).reset_index(drop=True)

        return df_nexuse_gens_ch_ror_updated

    def update_profiles_db(
        self, df_nexuse_gens_ch_ror_updated: pd.DataFrame, nexuse_db_filename: str
    ) -> pd.DataFrame:
        """Creates a pandas DataFrame containing the profiles (time series) for each
        power plant in the Nexus-e database. These include the updated RoR power plants
        from the WASTA database, for which we assign empty profiles (all 0s). These profiles
        are then saved to an Excel file that would allow to update a Nexus-e database.

        Parameters
        ----------
        df_nexuse_gens_ch_ror_updated : pd.DataFrame
            pandas DataFrame containing updated RoR power plants information and compliant
            with the Nexus-e database format
        nexuse_db_filename : str
            Name of the Nexus-e database file to be updated

        Returns
        -------
        pd.DataFrame
            pandas DataFrame containing profiles for all power plants in Nexus-e database
            and including RoR power plants from the WASTA database (profiles of which are
            empty, filled with 0s)
        """

        def empty_inflow_row(gen_name, row_profile):
            updated_row = row_profile.copy()
            updated_row["Name"] = f"inflow_CH_Hydro_RoR_{gen_name}"
            updated_row[6:8766] = 0
            updated_row[8767:8772] = 0
            return updated_row

        self.df_nexuse_profiles["Old Profile Number"] = self.df_nexuse_profiles[
            "Profile Number"
        ]

        template_row_ror_profile = (
            self.df_nexuse_profiles[
                self.df_nexuse_profiles["Name"].str.startswith("inflow_CH_Hydro_RoR")
            ]
            .iloc[0]
            .copy()
        )

        start_profile_number_ror = template_row_ror_profile["Profile Number"]
        last_profile_number_ror_current = self.df_nexuse_profiles.loc[
            self.df_nexuse_profiles["Name"].str.startswith("inflow_CH_Hydro_RoR"),
            "Profile Number",
        ].iloc[-1]

        last_profile_number_ror_updated = start_profile_number_ror + len(
            df_nexuse_gens_ch_ror_updated
        )

        df_nexuse_profiles_keep_before = self.df_nexuse_profiles.loc[
            self.df_nexuse_profiles["Profile Number"] < start_profile_number_ror
        ].copy()

        df_nexuse_profiles_keep_after = self.df_nexuse_profiles.loc[
            self.df_nexuse_profiles["Profile Number"] > last_profile_number_ror_current
        ].copy()

        df_nexuse_profiles_ror_updated = df_nexuse_gens_ch_ror_updated.apply(
            lambda row: empty_inflow_row(row["name"], template_row_ror_profile), axis=1
        )

        df_nexuse_profiles_ror_updated["Profile Number"] = list(
            range(
                start_profile_number_ror,
                last_profile_number_ror_updated,
            )
        )

        df_nexuse_profiles_keep_after["Profile Number"] = list(
            range(
                last_profile_number_ror_updated,
                last_profile_number_ror_updated
                + len(df_nexuse_profiles_keep_after),
            )
        )

        df_nexuse_profiles_updated = pd.concat(
            [
                df_nexuse_profiles_keep_before,
                df_nexuse_profiles_ror_updated,
                df_nexuse_profiles_keep_after,
            ]
        ).reset_index(drop=True)

        return df_nexuse_profiles_updated

    def update_gens_profile_number(
        self,
        df_nexuse_gens_ch_ror_updated: pd.DataFrame,
        df_nexuse_profiles_updated: pd.DataFrame,
    ) -> pd.DataFrame:
        """Update the profile number of Nexus-e generators, the unchanged ones
        as well as the updated RoR power plants from the WASTA database, and
        save the final DataFrame to an Excel that would allow to update a
        Nexus-e database.

        Parameters
        ----------
        df_nexuse_gens_ch_ror_updated : pd.DataFrame
            pandas DataFrame containing updated RoR power plants information and compliant
            with the Nexus-e database format
        df_nexuse_profiles_updated : pd.DataFrame
            pandas DataFrame containing profiles for all power plants in Nexus-e database
            and including RoR power plants from the WASTA database (profiles of which are
            empty, filled with 0s)

        Returns
        -------
        pd.DataFrame
            pandas DataFrame of Nexus-e generators information, including updated
            RoR power plants from the WASTA database, with updated profile numbers
        """
        df_nexuse_gens_unchanged = self.df_nexuse_gens[
            ~self.df_nexuse_gens["Gen_ID"].str.startswith("CH_Hydro")
        ]
        df_nexuse_gens_hydro = self.df_nexuse_gens[
            self.df_nexuse_gens["Gen_ID"].str.startswith("CH_Hydro")
        ]
        df_nexuse_dams_pump_ch = df_nexuse_gens_hydro[
            (df_nexuse_gens_hydro["SubType"].isin(["Dam", "Pump-Open"]))
            & (
                ~df_nexuse_gens_hydro["ZE-Name Wasserstatistik_1"].isin(
                    ["~~MISSING~~", "~~NEW~~"]
                )
            )
        ]

        df_nexuse_gens_updated = pd.concat(
            [
                df_nexuse_dams_pump_ch,
                df_nexuse_gens_ch_ror_updated,
                df_nexuse_gens_unchanged,
            ]
        )
        df_nexuse_gens_updated.loc[
            (df_nexuse_gens_updated["Gen_ID"].str.startswith("CH_Hydro_RoR")),
            "idProfile",
        ] = df_nexuse_gens_updated.loc[
            (df_nexuse_gens_updated["Gen_ID"].str.startswith("CH_Hydro_RoR"))
        ].apply(
            lambda row: df_nexuse_profiles_updated.loc[
                (
                    df_nexuse_profiles_updated["Name"]
                    == f"inflow_CH_Hydro_RoR_{row['name']}"
                ),
                "Profile Number",
            ].item(),
            axis=1,
        ).tolist()

        df_nexuse_gens_updated.loc[
            ((~df_nexuse_gens_updated["Gen_ID"].str.startswith("CH_Hydro_RoR"))
            & (~df_nexuse_gens_updated["idProfile"].isna())),
            "idProfile",
        ] = df_nexuse_gens_updated.loc[
            ((~df_nexuse_gens_updated["Gen_ID"].str.startswith("CH_Hydro_RoR"))
            & (~df_nexuse_gens_updated["idProfile"].isna()))
        ].apply(
            lambda row: df_nexuse_profiles_updated.loc[
                (df_nexuse_profiles_updated["Old Profile Number"] == row["idProfile"]),
                "Profile Number",
            ].item(),
            axis=1,
        ).tolist()

        df_nexuse_gens_updated["GenNum"] = list(
            range(1, len(df_nexuse_gens_updated) + 1)
        )

        return df_nexuse_gens_updated

    def save_new_gens_and_profiles(
        self,
        df_nexuse_gens_updated: pd.DataFrame,
        df_nexuse_profiles_updated: pd.DataFrame,
    ) -> None:
        """Saves the new list of generators and their corresponding profiles to an
        two separate Excel spreadsheets.

        Parameters
        ----------
        df_nexuse_gens_updated : pd.DataFrame
            pandas DataFrame of Nexus-e generators information, including updated
            RoR power plants from the WASTA database, with updated profile numbers
        df_nexuse_profiles_updated : pd.DataFrame
            pandas DataFrame containing profiles for all power plants in Nexus-e database
            and including RoR power plants from the WASTA database (profiles of which are
            empty, filled with 0s)
        """
        df_nexuse_gens_updated.to_excel(
            self.path_data_nexuse
            / "database"
            / f"{self.nexuse_db_filename.split('.')[0]}_new_gens.xlsx",
            sheet_name="gens",
        )

        df_nexuse_profiles_updated.drop(columns=["Old Profile Number"]).to_excel(
            self.path_data_nexuse
            / "database"
            / f"{self.nexuse_db_filename.split('.')[0]}_new_profiles.xlsx",
            sheet_name="profiles",
        )

    def get_profile_generator(
        self,
        row_gen: pd.Series,
        ds_gen: xr.Dataset,
        df_nexuse_profiles: pd.DataFrame,
        year: str,
    ) -> pd.Series:
        """Update the profile of a given generator with the estimated generation
        for a given year.

        Parameters
        ----------
        row_gen : pd.Series
            pandas Series containing information about the power plant
        ds_gen : xr.Dataset
            xarray Dataset containing estimated generation for power plants
        df_nexuse_profiles : pd.DataFrame
            pandas DataFrame containing profiles for all power plants in Nexus-e database
            and including RoR power plants from the WASTA database (profiles of which are
            empty, filled with 0s)
        year : str
            Selected year of estimated generation to update the profile with

        Returns
        -------
        pd.Series
            Updated profile with estimated generation of the given year
        """
        wasta_number = [int(wasta) for wasta in str(row_gen["WASTANumber"]).split(", ")]
        ds_hydropower = ds_gen.sel(time=year, hydropower=wasta_number).sum("hydropower")
        ds_hydropower = ds_hydropower * 1e6 / float(row_gen["P_gen_max in 2015 (MW)"])

        row_profile = (
            df_nexuse_profiles.loc[
                df_nexuse_profiles["Name"] == f"inflow_CH_Hydro_RoR_{row_gen['name']}",
                :,
            ]
            .iloc[0]
            .copy()
        )
        timeseries = ds_hydropower.gen.to_numpy()
        timeseries[timeseries > 1] = 1.0
        row_profile.iloc[6:8766] = timeseries
        row_profile.loc["SUM"] = row_profile.iloc[6:8766].sum()
        row_profile.loc["MAX"] = row_profile.iloc[6:8766].max()
        row_profile.loc["MIN"] = row_profile.iloc[6:8766].min()
        row_profile.loc["AVG"] = row_profile.iloc[6:8766].mean()
        row_profile.loc["Std Dev"] = row_profile.iloc[6:8766].std()
        return row_profile

    def create_profiles_estimated_generation(
        self,
        bias_correction_factors_filename: str,
        df_nexuse_gens_updated: pd.DataFrame,
        df_nexuse_profiles_updated: pd.DataFrame,
    ) -> None:
        """Creates profiles of estimated generation for all the RoR power plants
        in the updated generators DataFrame and all years present in the hydropower
        generation dataset. It then saves each year in a separate .csv file.

        Parameters
        ----------
        bias_correction_factors_filename : str
            Name of file containing the bias correction factors
        df_nexuse_gens_updated : pd.DataFrame
            pandas DataFrame of Nexus-e generators information, including updated
            RoR power plants from the WASTA database, with updated profile numbers
        df_nexuse_gens_ch_ror_updated : pd.DataFrame
            pandas DataFrame containing updated RoR power plants information and compliant
            with the Nexus-e database format
        df_nexuse_profiles_updated : pd.DataFrame
            pandas DataFrame containing profiles for all power plants in Nexus-e database
            and including RoR power plants from the WASTA database (profiles of which are
            empty, filled with 0s)
        """
        ds_monthly_bias_correction_factors = xr.open_dataset(
            self.path_data_hydro
            / "hydropower_generation"
            / bias_correction_factors_filename
        ).load()
        ds_hydropower_generation_monthly_bias_corrected = (
            self.ds_hydropower_generation
            * ds_monthly_bias_correction_factors.bias_correction_factor
        )
        years = np.unique(self.ds_hydropower_generation.time.dt.year)
        output_path = self.path_data_nexuse / "database" / "hydropower_profiles"
        output_path.mkdir(parents=True, exist_ok=True)
        for year in years:
            df_hydropower_profiles = df_nexuse_gens_updated.loc[
                df_nexuse_gens_updated["Gen_ID"].str.startswith("CH_Hydro_RoR"), :
            ].apply(
                lambda row: self.get_profile_generator(
                    row,
                    ds_hydropower_generation_monthly_bias_corrected,
                    df_nexuse_profiles_updated,
                    str(year),
                ),
                axis=1,
            )
            df_hydropower_profiles.to_csv(
                self.path_data_nexuse
                / "database"
                / "hydropower_profiles"
                / f"df_hydropower_profiles_{year}.csv",
                index=False,
            )


if __name__ == "__main__":
    nexuse_db_filename = (
        "Nexuse_DB-Input_v44_TYNDP22-GA08_DistIvPV_static_adj_base_AlpPVexist.xlsx"
    )
    match = MatchWastaNexuse(
        "../paths.json",
        "ds_prevah_500_hydropower_production_ror_simplified_efficiency.nc",
        nexuse_db_filename,
    )
    df_nexuse_gens_ch_ror_updated = match.create_wasta_nexuse_ror_gens_db()
    df_nexuse_profiles_updated = match.update_profiles_db(
        df_nexuse_gens_ch_ror_updated, nexuse_db_filename
    )
    df_nexuse_gens_updated = match.update_gens_profile_number(
        df_nexuse_gens_ch_ror_updated, df_nexuse_profiles_updated
    )

    match.save_new_gens_and_profiles(df_nexuse_gens_updated, df_nexuse_profiles_updated)

    monthly_bias_correction_factors_filename = "ds_prevah_500_hydropower_production_ror_simplified_efficiency_monthly_bias_correction_factors.nc"
    match.create_profiles_estimated_generation(
        monthly_bias_correction_factors_filename,
        df_nexuse_gens_updated,
        df_nexuse_profiles_updated,
    )
