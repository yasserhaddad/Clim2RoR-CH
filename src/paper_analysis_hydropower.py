import os
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

os.environ["USE_PYGEOS"] = "0"
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import ListedColormap

sns.set_style("whitegrid", {"grid.color": ".93"})

CBAR_LEVELS_QUANTILES = [
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
]

LEN_YEAR = len(pd.date_range("01-01-2022", "01-01-2023", freq="h", inclusive="left"))
LEN_SUMMER = len(pd.date_range("04-01-2022", "10-01-2022", freq="h", inclusive="left"))
LEN_WINTER = LEN_YEAR - LEN_SUMMER
cm = 1 / 2.54

FONTSIZE_TICKS = 6
FONTSIZE_LABELS = 7
FONTSIZE_TITLE = 8

blue = sns.color_palette("colorblind")[0]
orange = sns.color_palette("colorblind")[1]
green = sns.color_palette("colorblind")[2]
red = sns.color_palette("colorblind")[3]
deep_orange = sns.color_palette("colorblind")[5]
grey = sns.color_palette("colorblind")[7]


class NationalAnalysisHydropower:
    def __init__(
        self,
        gdf_switzerland: gpd.GeoDataFrame,
        gdf_hydropower_polygons: gpd.GeoDataFrame,
        df_stats_hydropower_ch: pd.DataFrame,
        ds_hydropower_generation: xr.Dataset,
        df_hydropower_production_params: pd.DataFrame,
        df_hydropower_generation_historical: pd.DataFrame,
        path_figs: pathlib.Path,
    ):
        self.gdf_switzerland = gdf_switzerland
        self.gdf_hydropower_polygons = gdf_hydropower_polygons
        self.df_stats_hydropower_ch = df_stats_hydropower_ch
        self.ds_hydropower_generation = ds_hydropower_generation
        self.df_hydropower_production_params = df_hydropower_production_params
        self.df_hydropower_generation_historical = df_hydropower_generation_historical
        self.path_figs = path_figs

        df_hydropower_locations = df_stats_hydropower_ch[
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

        self.percentage_hydropower_ch = xr.DataArray(
            df_stats_hydropower_ch.loc[
                (
                    df_stats_hydropower_ch["ZE-Nr"].isin(
                        ds_hydropower_generation.hydropower.values
                    )
                ),
                "Proz. Anteil CH",
            ].values
            / 100,
            dims=["hydropower"],
            coords={
                "hydropower": (
                    ["hydropower"],
                    df_stats_hydropower_ch.loc[
                        (
                            df_stats_hydropower_ch["ZE-Nr"].isin(
                                ds_hydropower_generation.hydropower.values
                            )
                        ),
                        "ZE-Nr",
                    ].values,
                )
            },
        ).broadcast_like(ds_hydropower_generation)

        self.ds_hydropower_generation_yearly = None
        self.ds_hydropower_generation_per_hp_yearly = None
        self.ds_hydropower_generation_yearly_with_operation_start = None
        self.ds_hydropower_generation_yearly_with_first_year_infrastructure = None

        self.ds_hydropower_generation_seasonal = None
        self.ds_hydropower_generation_per_hp_seasonal = None
        self.ds_hydropower_generation_seasonal_with_operation_start = None
        self.ds_hydropower_generation_seasonal_with_first_year_infrastructure = None

        self.ds_hydropower_generation_per_hp_yearly_ref = None
        self.ds_hydropower_generation_per_hp_seasonal_ref = None

    @staticmethod
    def seasonal_months(years: np.ndarray) -> Dict[str, Dict[str, List[str]]]:
        """Create a dictionary of the winter and summer months to take into account
        for the computation of seasonal electricity generation for each year
        present in the dataset.

        Parameters
        ----------
        years : np.ndarray
            List of years present in the dataset

        Returns
        -------
        Dict[str, Dict[str, List[str]]]
            A dictionary containing for each year, and for each of its winter and
            summer seasons, the months to take into account for the computation
            of seasonal electricity generation.
            Format: "year" -> {"winter": ["year-month",...],
                               "summer": ["year-month",...]}
        """
        summer = ["04", "05", "06", "07", "08", "09"]
        years_seasons = {}
        for idx, year in enumerate(years):
            if idx == 0:
                winter_year = [f"{year}-{month}" for month in ["01", "02", "03"]]
                summer_year = [f"{year}-{month}" for month in summer]
            else:
                winter_year = [f"{year - 1}-{month}" for month in ["10", "11", "12"]]
                winter_year.extend([f"{year}-{month}" for month in ["01", "02", "03"]])
                summer_year = [f"{year}-{month}" for month in summer]
            years_seasons[year] = {"winter": winter_year, "summer": summer_year}

        return years_seasons

    def aggregate_yearly_estimated_generation(
        self,
        with_percentage: bool = False,
        with_first_year_infrastructure: bool = False,
    ) -> None:
        """Aggregate estimated generation across all hydropower plants to obtain a yearly
        national estimated hydropower generation.

        Parameters
        ----------
        with_percentage : bool, optional
            Whether to multiply the hydropower generation of
            each power plant by the percentage of power that
            Switzerland is entitled to, by default False
        with_first_year_infrastructure : bool, optional
            Whether to fix the hydropower fleet to its state in the
            first year present in the study period, by default False
        """
        years = np.unique(self.ds_hydropower_generation.time.dt.year)
        hp_in_ds = self.ds_hydropower_generation.hydropower.to_numpy()
        wasta = (
            self.gdf_hydropower_locations[
                (self.gdf_hydropower_locations["BeginningOfOperation"] <= years[0])
                & (self.gdf_hydropower_locations["WASTANumber"].isin(hp_in_ds))
            ]["WASTANumber"].tolist()
            if with_first_year_infrastructure
            else hp_in_ds
        )
        variable_name = "ds_hydropower_generation_yearly"

        if with_first_year_infrastructure:
            variable_name += "_with_first_year_infrastructure"

        if not getattr(self, variable_name):
            setattr(
                self,
                variable_name,
                (
                    self.ds_hydropower_generation * self.percentage_hydropower_ch
                    if with_percentage
                    else self.ds_hydropower_generation
                )
                .sel(hydropower=wasta)
                .resample(time="Y")
                .sum(["hydropower", "time"]),
            )

    def aggregate_yearly_estimated_generation_per_hp(
        self, reference_period: slice = None
    ) -> None:
        """Aggregate estimated generation to obtain a yearly estimated hydropower generation
        per hydropower plant.

        Parameters
        ----------
        reference_period : slice
            Slice indicating the reference period,
            by default slice("1991", "2020")
        """
        dataset_name = (
            "ds_hydropower_generation_per_hp_yearly_ref"
            if reference_period
            else "ds_hydropower_generation_per_hp_yearly"
        )
        if not getattr(self, dataset_name):
            period = reference_period if reference_period else slice(None, None)
            setattr(
                self,
                dataset_name,
                (
                    self.ds_hydropower_generation.sel(time=period)
                    .resample(time="Y")
                    .sum(["time"])
                ),
            )
            getattr(self, dataset_name)["time"] = (
                "time",
                getattr(self, dataset_name).time.dt.year.values,
            )
            setattr(
                self,
                dataset_name,
                getattr(self, dataset_name)[
                    ["time", "hydropower"]
                    + [var_name for var_name in getattr(self, dataset_name).data_vars]
                ],
            )

    def aggregate_yearly_estimated_generation_with_operation_start(
        self, with_percentage: bool = False
    ):
        """Aggregate estimated generation taking only hydropower plants in operation for each year
        in the database to obtain a yearly national estimated hydropower generation.

        Parameters
        ----------
        with_percentage : bool
            Whether to multiply the hydropower generation of
            each power plant by the percentage of power that
            Switzerland is entitled to, by default True
        """
        if not self.ds_hydropower_generation_yearly_with_operation_start:
            list_ds = []
            hp_in_ds = self.ds_hydropower_generation.hydropower.to_numpy()
            for i in np.unique(self.ds_hydropower_generation.time.dt.year):
                wasta = self.gdf_hydropower_locations[
                    (self.gdf_hydropower_locations["BeginningOfOperation"] <= i)
                    & (self.gdf_hydropower_locations["WASTANumber"].isin(hp_in_ds))
                ]["WASTANumber"].tolist()
                list_ds.append(
                    (
                        self.ds_hydropower_generation * self.percentage_hydropower_ch
                        if with_percentage
                        else self.ds_hydropower_generation
                    )
                    .sel(hydropower=wasta, time=str(i))
                    .resample(time="Y")
                    .sum(["hydropower", "time"])
                )
            self.ds_hydropower_generation_yearly_with_operation_start = xr.concat(
                list_ds, dim="time"
            )

    def aggregate_seasonal_generation(
        self,
        year: int,
        wasta: List[int],
        years_seasons: Dict[str, Dict[str, List[str]]],
        season: str,
        per_hydropower: bool = False,
        with_percentage: bool = False,
    ) -> xr.Dataset:
        """Aggregate hydropower generation for a specific year and season to get either a seasonal
        aggregated value per hydropower plant or accross all the selected hydropower plants in the
        dataset.

        Parameters
        ----------
        year : int
            Year to select in dataset
        wasta : List[int]
            List of hydropower identifiers (WASTANumber) of the hydropower plants
            to select in the dataset
        years_seasons : Dict[str, Dict[str, List[str]]]
            A dictionary of the winter and summer months for each year in the dataset
        season : str
            Season for which to aggregate the hydropower generation time series
        per_hydropower : bool, optional
            Whether to obtain individual values for each hydropower plant or one
            aggregated value for all selected hydropower plants, by default False
        with_percentage : bool
            Whether to multiply the hydropower generation of each power plant by the
            percentage of power that Switzerland is entitled to, by default True

        Returns
        -------
        xr.Dataset
            Seasonally aggregated hydropower generation for the given year and season

        Raises
        ------
        ValueError
            In case the season is neither 'winter' nor 'summer'
        """
        if season not in ["winter", "summer"]:
            raise ValueError("The season can only be 'winter' or 'summer'.")

        dims_agg = ["time"] if per_hydropower else ["hydropower", "time"]

        return (
            (
                self.ds_hydropower_generation * self.percentage_hydropower_ch
                if with_percentage
                else self.ds_hydropower_generation
            )
            .sel(
                hydropower=wasta,
                time=slice(
                    years_seasons[year][season][0],
                    years_seasons[year][season][-1],
                ),
            )
            .resample(time="M")
            .sum(dims_agg)
            .sum(["time"])
            .rename(
                {
                    var_name: f"{var_name}_{season}"
                    for var_name in self.ds_hydropower_generation.data_vars
                }
            )
            .expand_dims({"time": [year]})
            .transpose()
        )

    def aggregate_seasonal_estimated_generation(
        self,
        with_operation_start: bool = True,
        per_hydropower: bool = False,
        with_percentage: bool = False,
        with_first_year_infrastructure: bool = False,
    ):
        """Aggregate hydropower generation to get either a seasonal aggregated
        value per hydropower plant or accross all the selected hydropower plants
        in the dataset.

        Parameters
        ----------
        with_operation_start : bool, optional
            Whether to take into account the operation start year of
            the hydropower plants in the aggregation of yearly or
            seasonal hydropower generation timeseries, by default True
        per_hydropower : bool, optional
            Whether to obtain individual values for each hydropower plant or one
            aggregated value for all selected hydropower plants, by default False
        with_percentage : bool
            Whether to multiply the hydropower generation of each power plant by
            the percentage of power that Switzerland is entitled to, by default True
        with_first_year_infrastructure : bool, optional
            Whether to fix the hydropower fleet to its state in the first year
            present in the study period, by default False
        """
        list_ds = []
        years = np.unique(self.ds_hydropower_generation.time.dt.year)
        years_seasons = self.seasonal_months(years)

        variable_name = "ds_hydropower_generation"
        if per_hydropower:
            variable_name += "_per_hp"
        variable_name += "_seasonal"
        if with_operation_start:
            variable_name += "_with_operation_start"
        if with_first_year_infrastructure:
            variable_name += "_with_first_year_infrastructure"

        if not getattr(self, variable_name):
            for i, year in enumerate(years):
                selected_year = years[0] if with_first_year_infrastructure else year
                wasta = (
                    self.gdf_hydropower_locations[
                        (
                            self.gdf_hydropower_locations["BeginningOfOperation"]
                            <= selected_year
                        )
                        & (
                            self.gdf_hydropower_locations["WASTANumber"].isin(
                                self.ds_hydropower_generation.hydropower.to_numpy()
                            )
                        )
                    ]["WASTANumber"].tolist()
                    if with_operation_start or with_first_year_infrastructure
                    else self.ds_hydropower_generation.hydropower.to_numpy()
                )

                ds_hydropower_generation_winter = self.aggregate_seasonal_generation(
                    year,
                    wasta,
                    years_seasons,
                    season="winter",
                    per_hydropower=per_hydropower,
                    with_percentage=with_percentage,
                )
                if i == 0:
                    ds_hydropower_generation_winter = (
                        ds_hydropower_generation_winter.where(
                            ds_hydropower_generation_winter.time == str(year), np.nan
                        )
                    )
                ds_hydropower_generation_summer = self.aggregate_seasonal_generation(
                    year,
                    wasta,
                    years_seasons,
                    season="summer",
                    per_hydropower=per_hydropower,
                    with_percentage=with_percentage,
                )
                list_ds.append(
                    xr.merge(
                        [
                            ds_hydropower_generation_winter,
                            ds_hydropower_generation_summer,
                        ]
                    )
                )
            setattr(self, variable_name, xr.concat(list_ds, dim="time"))

    def aggregate_expected_generation(self, season: str = None) -> float:
        """Returns the national yearly or seasonal expected hydropower generation (in TWh), taking into
        account the hydropower plants for which estimations were produced.

        Parameters
        ----------
        season : str
            Aggregate hydropower generarion on a seasonal scale by providing the
            energy season (winter or summer), by default None

        Returns
        -------
        float
            The national yearly or seasonal expected hydropower generation (in TWh)

        Raises
        ------
        ValueError
            In case the df_hydropower_production_params does not include the appropriate
            expected generation column
        ValueError
            In case the season is neither 'winter' nor 'summer'
        """
        if season and season not in ["summer", "winter"]:
            raise ValueError("The season can only be 'winter' or 'summer'.")

        column_type = "yearly" if not season else season

        if (
            f"Expected {column_type} generation"
            not in self.df_hydropower_production_params
        ):
            raise ValueError(
                f"The df_hydropower_production_params DataFrame does not include a 'Expected {column_type} generation' column."
            )
        return (
            self.df_hydropower_production_params[
                f"Expected {column_type} generation"
            ].sum()
            * 1e-3
        )

    def aggregate_yearly_historical_generation(
        self, columns_to_aggregate: List[str] = ["Erzeugung_laufwerk_GWh"]
    ) -> pd.Series:
        """Returns a timeseries of the national yearly reported hydropower generation (in TWh).

        Parameters
        ----------
        columns_to_aggregate : List[str], optional
            Columns from the historical hydropower generation DataFrame to aggregate,
            by default ["Erzeugung_laufwerk_GWh"]

        Returns
        -------
        pd.Series
            Timeseries of the national yearly reported hydropower generation (in TWh)

        Raises
        ------
        ValueError
            In case not all required columns (columns to aggregate and 'Jahr') are
            present in df_hydropower_generation_historical
        """
        if any(
            col not in self.df_hydropower_generation_historical
            for col in columns_to_aggregate + ["Jahr"]
        ):
            raise ValueError(
                "Not all required columns are present in df_hydropower_generation_historical"
            )

        df_historical_data_year = (
            self.df_hydropower_generation_historical[
                self.df_hydropower_generation_historical["Jahr"] < 2023
            ]
            .groupby("Jahr")
            .agg(sum)
        )
        return df_historical_data_year[columns_to_aggregate].sum(axis=1) * 1e-3

    def aggregate_seasonal_historical_generation(
        self, season: str, columns_to_aggregate: List[str] = ["Erzeugung_laufwerk_GWh"]
    ) -> pd.Series:
        """Returns a timeseries of the national seasonal (either winter or summer)
        reported hydropower generation (in TWh).

        Parameters
        ----------
        season : str
            Season for which to aggregate the hydropower generation time series
        columns_to_aggregate : List[str], optional
            Columns from the historical hydropower generation DataFrame to aggregate,
            by default ["Erzeugung_laufwerk_GWh"]

        Returns
        -------
        pd.Series
            Timeseries of the national seasonal reported hydropower generation (in TWh)

        Raises
        ------
        ValueError
            In case not all required columns (columns to aggregate, 'Jahr' and 'Monat')
            are present in df_hydropower_generation_historical
        """
        if any(
            col not in self.df_hydropower_generation_historical
            for col in columns_to_aggregate + ["Jahr", "Monat"]
        ):
            raise ValueError(
                "Not all required columns are present in df_hydropower_generation_historical"
            )
        if "Jahr_monat" not in self.df_hydropower_generation_historical:
            self.df_hydropower_generation_historical = (
                self.df_hydropower_generation_historical.assign(
                    Jahr_monat=self.df_hydropower_generation_historical.apply(
                        lambda row: f"{row['Jahr']}-{row['Monat']:02d}", axis=1
                    ).to_list()
                )
            )

        historical_data_years = np.unique(
            self.df_hydropower_generation_historical[
                self.df_hydropower_generation_historical["Jahr"] < 2023
            ]["Jahr"]
        )
        years_seasons = self.seasonal_months(historical_data_years)
        seasonal_historical_hydropower_generation = []
        for year in historical_data_years:
            if year in historical_data_years and year != historical_data_years[0]:
                seasonal_historical_hydropower_generation.append(
                    self.df_hydropower_generation_historical.loc[
                        self.df_hydropower_generation_historical["Jahr_monat"].isin(
                            years_seasons[year][season]
                        ),
                        columns_to_aggregate,
                    ]
                    .sum()
                    .item()
                    * 1e-3
                )
            else:
                seasonal_historical_hydropower_generation.append(np.nan)

        return pd.Series(
            seasonal_historical_hydropower_generation,
            index=historical_data_years,
        )

    def aggregate_reference_seasonal_estimated_generation(
        self, reference_period: np.ndarray = np.arange(1991, 2021)
    ):
        """Aggregate hydropower generation to get a seasonal aggregated value
        per hydropower plant for the given reference period.

        Parameters
        ----------
        reference_period : np.ndarray, optional
            Array containing the years in the reference period,
            by default np.arange(1991, 2021)
        """
        list_ds = []
        years_seasons = self.seasonal_months(reference_period)

        if not self.ds_hydropower_generation_per_hp_seasonal_ref:
            wasta = self.ds_hydropower_generation.hydropower.to_numpy()
            for i, year in enumerate(np.unique(reference_period)):
                list_ds_year = []
                if i != 0:
                    list_ds_year.append(
                        self.aggregate_seasonal_generation(
                            year,
                            wasta,
                            years_seasons,
                            season="winter",
                            per_hydropower=True,
                        )
                    )
                list_ds_year.append(
                    self.aggregate_seasonal_generation(
                        year,
                        wasta,
                        years_seasons,
                        season="summer",
                        per_hydropower=True,
                    )
                )
                list_ds.append(xr.merge(list_ds_year))
            self.ds_hydropower_generation_per_hp_seasonal_ref = xr.concat(
                list_ds, dim="time"
            )

    def create_dataframe_yearly_values(
        self,
        with_operation_start: bool = True,
        with_percentage: bool = False,
        with_first_year_infrastructure: bool = False,
        historical_data_columns_to_aggregate: List[str] = ["Erzeugung_laufwerk_GWh"],
    ) -> pd.DataFrame:
        """Constructs a pandas DataFrame containing national yearly estimated, expected and reported hydropower generation.

        Parameters
        ----------
        with_operation_start : bool, optional
            Whether to take into account the operation start year of
            the hydropower plants in the aggregation of yearly or
            seasonal hydropower generation timeseries, by default True
        with_percentage : bool, optional
            Whether to multiply the hydropower generation of each power plant by
            the percentage of power that Switzerland is entitled to, by default True
        with_first_year_infrastructure : bool, optional
            Whether to fix the hydropower fleet to its state in the first year
            present in the study period, by default False
        historical_data_columns_to_aggregate : List[str], optional
            Columns from the historical hydropower generation DataFrame to aggregate,
            by default ["Erzeugung_laufwerk_GWh"]

        Returns
        -------
        pd.DataFrame
            A DataFrame containing national yearly estimated, expected and reported hydropower generation
        """
        estimated_variable_name = "ds_hydropower_generation_yearly"
        if with_operation_start:
            estimated_variable_name += "_with_operation_start"
        if with_first_year_infrastructure:
            estimated_variable_name += "_with_first_year_infrastructure"

        if with_operation_start:
            self.aggregate_yearly_estimated_generation_with_operation_start(
                with_percentage=with_percentage
            )
            estimated = {
                f"Estimated Generation {' '.join(var_name.replace('hp_', '').split('_')).title()}": array.values
                for var_name, array in getattr(
                    self, estimated_variable_name
                ).data_vars.items()
            }
        else:
            self.aggregate_yearly_estimated_generation(
                with_percentage=with_percentage,
                with_first_year_infrastructure=with_first_year_infrastructure,
            )
            estimated = {
                f"Estimated Generation {' '.join(var_name.replace('hp_', '').split('_')).title()}": array.values
                for var_name, array in getattr(
                    self, estimated_variable_name
                ).data_vars.items()
            }
        expected = {
            "Expected Generation": [self.aggregate_expected_generation()]
            * len(self.ds_hydropower_generation_yearly.time)
        }
        df_hydropower_yearly = pd.DataFrame(
            estimated | expected,
            index=self.ds_hydropower_generation_yearly.time.dt.year.to_numpy(),
        )
        df_hydropower_yearly["Reported Generation"] = (
            self.aggregate_yearly_historical_generation(
                columns_to_aggregate=historical_data_columns_to_aggregate
            )
        )
        return df_hydropower_yearly

    def create_dataframe_seasonal_values(
        self,
        with_operation_start: bool = True,
        with_percentage: bool = False,
        with_first_year_infrastructure: bool = False,
        historical_data_columns_to_aggregate: List[str] = ["Erzeugung_laufwerk_GWh"],
    ) -> pd.DataFrame:
        """Constructs a pandas DataFrame containing national seasonal estimated, expected and
        reported hydropower generation.

        Parameters
        ----------
        with_operation_start : bool, optional
            Whether to take into account the operation start year of
            the hydropower plants in the aggregation of yearly or
            seasonal hydropower generation timeseries, by default True
        with_percentage : bool, optional
            Whether to multiply the hydropower generation of each power plant by
            the percentage of power that Switzerland is entitled to, by default False
        with_first_year_infrastructure : bool, optional
            Whether to fix the hydropower fleet to its state in the first year
            present in the study period, by default False
        historical_data_columns_to_aggregate : List[str], optional
            Columns from the historical hydropower generation DataFrame to aggregate,
            by default ["Erzeugung_laufwerk_GWh"]

        Returns
        -------
        pd.DataFrame
            A DataFrame containing national seasonal estimated, expected and reported hydropower generation
        """
        self.aggregate_seasonal_estimated_generation(
            with_operation_start=with_operation_start,
            per_hydropower=False,
            with_percentage=with_percentage,
            with_first_year_infrastructure=with_first_year_infrastructure,
        )
        estimated_variable_name = "ds_hydropower_generation_seasonal"
        if with_operation_start:
            estimated_variable_name += "_with_operation_start"
        if with_first_year_infrastructure:
            estimated_variable_name += "_with_first_year_infrastructure"

        estimated = {
            f"Estimated Generation {' '.join(var_name.replace('hp_', '').split('_')).title()}": array.values
            for var_name, array in getattr(
                self, estimated_variable_name
            ).data_vars.items()
        }

        expected_winter = {
            "Expected Generation Winter": [np.nan]
            + [self.aggregate_expected_generation(season="winter")]
            * (len(getattr(self, estimated_variable_name).time) - 1)
        }
        expected_summer = {
            "Expected Generation Summer": [
                self.aggregate_expected_generation(season="summer")
            ]
            * len(getattr(self, estimated_variable_name).time)
        }

        df_hydropower_seasonal = pd.DataFrame(
            estimated | expected_winter | expected_summer,
            index=getattr(self, estimated_variable_name).time.to_numpy(),
        )

        df_hydropower_seasonal["Reported Generation Winter"] = (
            self.aggregate_seasonal_historical_generation(
                season="winter",
                columns_to_aggregate=historical_data_columns_to_aggregate,
            )
        )

        df_hydropower_seasonal["Reported Generation Summer"] = (
            self.aggregate_seasonal_historical_generation(
                season="summer",
                columns_to_aggregate=historical_data_columns_to_aggregate,
            )
        )

        return df_hydropower_seasonal

    def create_dataframe_seasonal_estimated_values_per_hp(
        self,
        variable_name: str,
        with_percentage: bool = False,
        with_first_year_infrastructure: bool = False,
    ) -> pd.DataFrame:
        """Constructs a pandas DataFrame containing seasonal estimated hydropower generation
        per hydropower plant.

        Parameters
        ----------
        variable_name : str
            Variable name to select in the xarray Dataset of seasonal
            estimated generation
        with_percentage : bool, optional
            Whether to multiply the hydropower generation of each power plant by
            the percentage of power that Switzerland is entitled to, by default False
        with_first_year_infrastructure : bool, optional
            Whether to fix the hydropower fleet to its state in the first year
            present in the study period, by default False

        Returns
        -------
        pd.DataFrame
            pandas DataFrame with seasonal estimated hydropower generation per hydropower plant
            (index) and every year in the study period (columns)
        """
        self.aggregate_seasonal_estimated_generation(
            with_operation_start=False,
            per_hydropower=False,
            with_percentage=with_percentage,
            with_first_year_infrastructure=with_first_year_infrastructure,
        )

        df_hydropower_generation_per_hp_seasonal = (
            self.ds_hydropower_generation_per_hp_seasonal[variable_name]
            .to_dataframe()
            .reset_index(level=1)
        )
        df_hydropower_generation_per_hp_seasonal = (
            df_hydropower_generation_per_hp_seasonal.pivot_table(
                values=variable_name,
                index=df_hydropower_generation_per_hp_seasonal.index,
                columns="time",
            )
        )

        return df_hydropower_generation_per_hp_seasonal

    def create_dataframe_monthly_estimated_generation(
        self, variable_name: str
    ) -> pd.DataFrame:
        """Constructs a pandas DataFrame containing national monthly estimated
        hydropower generation.

        Parameters
        ----------
        variable_name : str
            Variable name to select in the xarray Dataset

        Returns
        -------
        pd.DataFrame
            A DataFrame containing national monthly estimated hydropower generation
        """
        df_hydropower_per_month = []
        for month in range(1, 13):
            df_hydropower_per_month.append(
                self.ds_hydropower_generation.sel(
                    time=(self.ds_hydropower_generation["time.month"] == month)
                )
                .resample(time="Y")
                .sum(["hydropower", "time"])[variable_name]
                .to_numpy()
            )
        df_hydropower_per_month = pd.DataFrame(df_hydropower_per_month)
        df_hydropower_per_month.index = list(range(1, 13))
        df_hydropower_per_month.columns = np.unique(
            self.ds_hydropower_generation.time.dt.year
        )

        return df_hydropower_per_month

    def get_hydropower_generation_and_ref(
        self, yearly: bool, variable_name: str
    ) -> Tuple[pd.DataFrame, xr.Dataset]:
        """Returns respectively a pandas DataFrame and an xarray Dataset of yearly or seasonal
        hydropower generation per hydropower plant for the entire period and for the reference period.

        Parameters
        ----------
        yearly : bool
            Whether to compute the correlation on yearly or seasonal
            timeseries of hydropower generation
        variable_name : str
            Variable name to select in the xarray Dataset

        Returns
        -------
        Tuple[pd.DataFrame, xr.Dataset]
            A pandas DataFrame and an xarray Dataset of yearly or seasonal
            hydropower generation per hydropower plant for the entire period
            and for the reference period
        """
        dataset_name = (
            "ds_hydropower_generation_per_hp_yearly"
            if yearly
            else "ds_hydropower_generation_per_hp_seasonal"
        )
        dataset_ref_name = (
            "ds_hydropower_generation_per_hp_yearly_ref"
            if yearly
            else "ds_hydropower_generation_per_hp_seasonal_ref"
        )
        df_generation = getattr(self, dataset_name).to_dataframe()[[variable_name]]
        ds_generation_ref = getattr(self, dataset_ref_name)

        if "winter" in variable_name:
            start_year = getattr(self, dataset_name).time.values[0]
            df_generation = df_generation[
                df_generation.index.get_level_values("time") > start_year
            ]
            ds_generation_ref = ds_generation_ref.sel(
                time=ds_generation_ref.time.values[1:]
            )

        return df_generation, ds_generation_ref

    def create_dataframe_with_quantiles(
        self, yearly: bool, variable_name: str
    ) -> pd.DataFrame:
        """Returns a pandas DataFrame of yearly or seasonal hydropower generation
        per hydropower plant along with their quantiles with respect to a reference period.

        Parameters
        ----------
        yearly : bool
            Whether to compute the correlation on yearly or seasonal
            timeseries of hydropower generation
        variable_name : str
            Variable name to select in the xarray Dataset

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame of yearly or seasonal hydropower generation
            per hydropower plant along with their quantiles with respect
            to a reference period
        """
        df_generation, ds_generation_ref = self.get_hydropower_generation_and_ref(
            yearly, variable_name
        )

        df_generation["quantile"] = (
            df_generation.reset_index()
            .apply(
                lambda r: round(
                    (
                        ds_generation_ref[variable_name].sel(hydropower=r["hydropower"])
                        < r[variable_name]
                    )
                    .mean()
                    .item(),
                    2,
                ),
                axis=1,
            )
            .to_list()
        )

        df_generation["capacity"] = (
            df_generation.reset_index()
            .apply(
                lambda r: self.df_stats_hydropower_ch.loc[
                    self.df_stats_hydropower_ch["ZE-Nr"] == r["hydropower"],
                    "Max. Leistung ab Generator",
                ].item(),
                axis=1,
            )
            .to_list()
        )

        return df_generation

    def create_dataframe_with_percentage_change(self, yearly: bool, variable_name: str):
        """Returns a pandas DataFrame of yearly or seasonal hydropower generation
        per hydropower plant along with the percentage of change with
        respect to a reference period.

        Parameters
        ----------
        yearly : bool
            Whether to compute the correlation on yearly or seasonal
            timeseries of hydropower generation
        variable_name : str
            Variable name to select in the xarray Dataset

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame of yearly or seasonal hydropower generation
            per hydropower plant along with percentage of change with respect
            to a reference period
        """

        def get_percentage_change(
            row: pd.Series, ds_ref: xr.Dataset, variable_name: str
        ):
            ref = ds_ref.sel(hydropower=row["hydropower"])[variable_name].item()
            return round((row[variable_name] - ref) / ref * 100, 2)

        df_generation, ds_generation_ref = self.get_hydropower_generation_and_ref(
            yearly, variable_name
        )
        ds_generation_ref_mean = ds_generation_ref.mean(["time"])

        df_generation["percentage_change"] = (
            df_generation.reset_index()
            .apply(
                lambda r: get_percentage_change(
                    r, ds_generation_ref_mean, variable_name
                ),
                axis=1,
            )
            .to_list()
        )

        return df_generation

    def create_dataframe_capacity_factors(
        self, yearly: bool, variable_name: str
    ) -> pd.DataFrame:
        """Returns a pandas DataFrame of yearly or seasonal hydropower generation per hydropower plant
        along with their capacity factors.

        Parameters
        ----------
        yearly : bool
            Whether to compute the capacity factors on yearly or seasonal
            scale
        variable_name : str
            Variable name to select in the xarray Dataset

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame of yearly or seasonal hydropower generation
            per hydropower plant along with percentage of change with respect
            to a reference period
        """
        df_generation, _ = self.get_hydropower_generation_and_ref(yearly, variable_name)

        len_period = (
            LEN_YEAR
            if yearly
            else (LEN_WINTER if "winter" in variable_name else LEN_SUMMER)
        )

        df_generation["max"] = (
            df_generation.reset_index()
            .apply(
                lambda r: self.df_stats_hydropower_ch.loc[
                    self.df_stats_hydropower_ch["ZE-Nr"] == r["hydropower"],
                    "Max. Leistung ab Generator",
                ].item()
                * 1e-6
                * len_period,
                axis=1,
            )
            .to_list()
        )

        df_generation["capacity_factor"] = (
            df_generation[variable_name] / df_generation["max"]
        )

        return df_generation

    def compute_pred_obs_correlation(
        self,
        yearly: bool = True,
        with_operation_start: bool = True,
        with_percentage: bool = False,
    ) -> pd.DataFrame:
        """Compute the correlation between the estimated hydropower
        generation and the reported/expected generation. The values
        are stored in pandas DataFrame.

        Parameters
        ----------
        yearly : bool, optional
            Whether to compute the correlation on yearly or seasonal
            timeseries of hydropower generation, by default True
        with_operation_start : bool, optional
            Whether to take into account the operation start year of
            the hydropower plants in the aggregation of yearly or
            seasonal hydropower generation timeseries, by default True
        with_percentage : bool
            Whether to multiply the hydropower generation of each power plant by
            the percentage of power that Switzerland is entitled to, by default True

        Returns
        -------
        pd.DataFrame
            Correlation matrix between the estimated hydropower
            generation and the reported/expected generation.
        """
        df_hydropower = (
            self.create_dataframe_yearly_values(
                with_operation_start=with_operation_start,
                with_percentage=with_percentage,
            ).dropna()
            if yearly
            else self.create_dataframe_seasonal_values(
                with_operation_start=with_operation_start,
                with_percentage=with_percentage,
            ).dropna()
        )
        return df_hydropower.corr()

    def compute_pred_bias(
        self,
        yearly: bool = True,
        with_operation_start: bool = True,
        with_percentage: bool = True,
        relative_bias: bool = False,
    ) -> pd.DataFrame:
        """Compute the bias between the estimated hydropower
        generation and the reported/expected generation. The values
        are stored in pandas DataFrame.

        Parameters
        ----------
        yearly : bool, optional
            Whether to compute the bias on yearly or seasonal
            timeseries of hydropower generation, by default True
        with_operation_start : bool, optional
            Whether to take into account the operation start year of
            the hydropower plants in the aggregation of yearly or
            seasonal hydropower generation timeseries, by default True
        with_percentage : bool
            Whether to multiply the hydropower generation of each power plant by
            the percentage of power that Switzerland is entitled to, by default True
        relative_bias : bool
            Whether to compute the relative bias rather than the additive bias

        Returns
        -------
        pd.DataFrame
            Bias between the estimated hydropower generation and
            the reported/expected generation.
        """
        df_hydropower = (
            self.create_dataframe_yearly_values(
                with_operation_start=with_operation_start,
                with_percentage=with_percentage,
            ).dropna()
            if yearly
            else self.create_dataframe_seasonal_values(
                with_operation_start=with_operation_start,
                with_percentage=with_percentage,
            ).dropna()
        )
        bias = {}
        for col in df_hydropower.columns:
            if col.startswith("Estimated"):
                reported_generation_column = "Reported Generation"
                if not yearly:
                    reported_generation_column += (
                        " Winter" if "Winter" in col else " Summer"
                    )
                bias[col] = (
                    np.mean(
                        df_hydropower.loc[:, col]
                        / df_hydropower.loc[:, reported_generation_column]
                    )
                    if relative_bias
                    else np.mean(
                        df_hydropower.loc[:, col]
                        - df_hydropower.loc[:, reported_generation_column]
                    )
                )

        return pd.DataFrame(bias, index=["Bias"])

    @staticmethod
    def compute_trend_statsmodel(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05,
        round_results: bool = True,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Compute linear trend using statsmodel's OLS class. This method
        returns the coefficients alongside their confidence interval.

        Parameters
        ----------
        X : np.ndarray
            X array to be passed to the Ordinary Least Squares method
        y : np.ndarray
            Array of target values to compute a linear trend on
        alpha : float, optional
            The alpha level for the confidence interval, by default 0.05
        round_results : bool, optional
            Whether to round the coefficients, by default True

        Returns
        -------
        Tuple[float, float, np.ndarray, np.ndarray]
            The coefficient and the intercept of the linear regression, along
            with the predicted values of the linear regression and the
            confidence interval for the coefficients
        """
        lr = sm.OLS(y, sm.add_constant(X)).fit()
        conf_interval = lr.conf_int(alpha)
        pred = lr.predict(sm.add_constant(X))
        coef = lr.params[1]
        intercept = lr.params[0]
        if round_results:
            coef = round(coef, 3)
            intercept = round(intercept, 3)

        return coef, intercept, pred, conf_interval

    def plot_ror_map_capacities_hist(
        self, save: bool = False, output_filename: str = None
    ) -> None:
        """Plot a map of the location of RoR hydropower plants in the WASTA database, along
        with a histogram of their capacities.

        Parameters
        ----------
        save : bool, optional
            Whether to save the plot, by default False
        output_filename : str, optional
            The name of the file containing the plot, by default None
        """
        fig, ax = plt.subplots(1, 2, figsize=(15 * cm, 6 * cm), width_ratios=[2, 1])
        capacity_upper_10 = self.gdf_hydropower_locations[
            (
                self.gdf_hydropower_locations["WASTANumber"].isin(
                    self.ds_hydropower_generation.hydropower.values
                )
            )
        ]["Capacity"].quantile(0.9)

        self.gdf_switzerland.plot(ax=ax[0], color="white", edgecolor="black")
        self.gdf_hydropower_locations[
            (
                self.gdf_hydropower_locations["WASTANumber"].isin(
                    self.ds_hydropower_generation.hydropower.values
                )
            )
            & (self.gdf_hydropower_locations["Capacity"] < capacity_upper_10)
            & (~pd.isna(self.gdf_hydropower_locations["Canton"]))
        ].plot(ax=ax[0], legend=False, color=blue, marker=".", markersize=4)
        self.gdf_hydropower_locations[
            (
                self.gdf_hydropower_locations["WASTANumber"].isin(
                    self.ds_hydropower_generation.hydropower.values
                )
            )
            & (self.gdf_hydropower_locations["Capacity"] >= capacity_upper_10)
            & (~pd.isna(self.gdf_hydropower_locations["Canton"]))
        ].plot(ax=ax[0], legend=False, color=red, marker=".", markersize=20)

        ax[0].axis("off")
        ax[0].set_title(
            "a", fontweight="bold", loc="left", fontsize=FONTSIZE_TITLE, y=1.05
        )

        self.gdf_hydropower_locations[
            (
                self.gdf_hydropower_locations["WASTANumber"].isin(
                    self.ds_hydropower_generation.hydropower.values
                )
            )
            & (~pd.isna(self.gdf_hydropower_locations["Canton"]))
        ]["Capacity"].plot.hist(ax=ax[1], bins=20, log=True)
        ax[1].set_xlabel("Capacity (in MW)", fontsize=FONTSIZE_LABELS)
        ax[1].set_ylabel("Log-Frequency", fontsize=FONTSIZE_LABELS)
        ax[1].set_title(
            "b", fontweight="bold", loc="left", fontsize=FONTSIZE_TITLE, x=-0.24, y=1.05
        )

        for i in range(len(ax)):
            ax[i].tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
            ax[i].tick_params(axis="both", which="minor", labelsize=FONTSIZE_TICKS)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()

        circle = mlines.Line2D(
            [0],
            [0],
            color="white",
            markerfacecolor=red,
            marker=".",
            markeredgewidth=0,
            markeredgecolor="black",
            markersize=10,
            label="Largest 10% of RoR\nhydropower plants",
        )
        ax[0].legend(
            handles=[circle],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            fontsize=FONTSIZE_LABELS,
            frameon=False,
        )

        if save and output_filename:
            output_path = self.path_figs
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                output_path / output_filename,
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def plot_validation(
        self,
        ax: np.ndarray[plt.Axes],
        with_percentage: bool = False,
        yearly_column_to_plot: str = None,
        winter_column_to_plot: str = None,
        summer_column_to_plot: str = None,
        subplots_titles: List[str] = None,
    ) -> None:
        """Plot a comparison of the yearly and seasonal estimated generation with
        the reported generation.

        Parameters
        ----------
        ax : np.ndarray[plt.Axes]
            Array of Axes objects to plot on
        with_percentage : bool, optional
            Whether to multiply the hydropower generation of each power plant by
            the percentage of power that Switzerland is entitled to, by default False
        yearly_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the yearly
            values, by default None
        winter_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the winter
            values, by default None
        summer_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the summer
            values, by default None
        subplots_titles : List[str], optional
            Titles of the three subplots, by default None
        """
        df_hydropower_yearly = self.create_dataframe_yearly_values(
            with_operation_start=True, with_percentage=with_percentage
        ).merge(
            self.create_dataframe_yearly_values(
                with_operation_start=False, with_percentage=with_percentage
            ),
            left_index=True,
            right_index=True,
            suffixes=("", "_fixed_system"),
        )

        df_hydropower_seasonal = self.create_dataframe_seasonal_values(
            with_operation_start=True, with_percentage=with_percentage
        ).merge(
            self.create_dataframe_seasonal_values(
                with_operation_start=False, with_percentage=with_percentage
            ),
            left_index=True,
            right_index=True,
            suffixes=("", "_fixed_system"),
        )

        if not yearly_column_to_plot:
            yearly_column_to_plot = [
                col for col in df_hydropower_yearly.columns if "Estimated" in col
            ][0]

        if not winter_column_to_plot:
            winter_column_to_plot = [
                col
                for col in df_hydropower_seasonal.columns
                if ("Estimated" in col) and ("Winter" in col)
            ][0]

        if not summer_column_to_plot:
            summer_column_to_plot = [
                col
                for col in df_hydropower_seasonal.columns
                if ("Estimated" in col) and ("Summer" in col)
            ][0]

        max_val = 20.5
        min_val = 13

        df_hydropower_yearly.plot(
            y=yearly_column_to_plot,
            ax=ax[0],
            label="Estimated Generation",
            legend=False,
            color=blue,
        )
        df_hydropower_yearly.plot(
            y="Reported Generation", ax=ax[0], legend=False, color=red
        )
        ax[0].set_ylim(min_val - 1.0, max_val + 1.0)
        if subplots_titles:
            ax[0].set_title(subplots_titles[0], fontsize=FONTSIZE_TITLE)
        xmin, xmax = ax[0].get_xlim()
        ax[0].set_ylabel("Generation (in TWh)", fontsize=FONTSIZE_LABELS)

        max_val_winter = 9
        min_val_winter = 4

        df_hydropower_seasonal.plot(
            y=winter_column_to_plot,
            ax=ax[1],
            label="Estimated Generation",
            legend=False,
            color=blue,
        )
        df_hydropower_seasonal.plot(
            y="Reported Generation Winter",
            ax=ax[1],
            label="Reported Generation",
            legend=False,
            color=red,
        )
        ax[1].set_ylim(min_val_winter - 1.0, max_val_winter + 1.0)
        ax[1].set_yticks(np.arange(3, 10, 2))
        if subplots_titles:
            ax[1].set_title(subplots_titles[1], fontsize=FONTSIZE_TITLE)
        ax[1].set_xlim(xmin, xmax)

        max_val_summer = 14.25
        min_val_summer = 8

        df_hydropower_seasonal.plot(
            y=summer_column_to_plot,
            ax=ax[2],
            label="Estimated Generation",
            color=blue,
            legend=False,
        )
        df_hydropower_seasonal.plot(
            y="Reported Generation Summer",
            ax=ax[2],
            label="Reported Generation",
            color=red,
            legend=False,
        )
        ax[2].set_ylim(min_val_summer - 1.0, max_val_summer + 1.0)
        if subplots_titles:
            ax[2].set_title(subplots_titles[2], fontsize=FONTSIZE_TITLE)

        for i in range(len(ax)):
            ax[i].tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
            ax[i].tick_params(axis="both", which="minor", labelsize=FONTSIZE_TICKS)

    def plot_infrastructure_trend(
        self,
        ax: plt.Axes,
        with_percentage: bool = False,
        yearly_column_to_plot: str = None,
        winter_column_to_plot: str = None,
        summer_column_to_plot: str = None,
        subplots_titles: List[str] = None,
    ):
        """Plot a comparison of the yearly and seasonal estimated generation for
        three different aggregation methods: evolving capacities, fixed capacities
        of 1991 and fixed capacities of 2022.

        Parameters
        ----------
        ax : np.ndarray[plt.Axes]
            Array of Axes objects to plot on
        with_percentage : bool, optional
            Whether to multiply the hydropower generation of each power plant by
            the percentage of power that Switzerland is entitled to, by default False
        yearly_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the yearly
            values, by default None
        winter_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the winter
            values, by default None
        summer_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the summer
            values, by default None
        subplots_titles : List[str], optional
            Titles of the three subplots, by default None
        """
        df_hydropower_yearly = (
            self.create_dataframe_yearly_values(
                with_operation_start=True, with_percentage=with_percentage
            )
            .merge(
                self.create_dataframe_yearly_values(
                    with_operation_start=False, with_percentage=with_percentage
                ),
                left_index=True,
                right_index=True,
                suffixes=("", "_fixed_system_2022"),
            )
            .merge(
                self.create_dataframe_yearly_values(
                    with_operation_start=False,
                    with_percentage=with_percentage,
                    with_first_year_infrastructure=True,
                ),
                left_index=True,
                right_index=True,
                suffixes=("", "_fixed_system_1991"),
            )
        )

        df_hydropower_seasonal = (
            self.create_dataframe_seasonal_values(
                with_operation_start=True, with_percentage=with_percentage
            )
            .merge(
                self.create_dataframe_seasonal_values(
                    with_operation_start=False, with_percentage=with_percentage
                ),
                left_index=True,
                right_index=True,
                suffixes=("", "_fixed_system_2022"),
            )
            .merge(
                self.create_dataframe_seasonal_values(
                    with_operation_start=False,
                    with_percentage=with_percentage,
                    with_first_year_infrastructure=True,
                ),
                left_index=True,
                right_index=True,
                suffixes=("", "_fixed_system_1991"),
            )
        )

        if not yearly_column_to_plot:
            yearly_column_to_plot = [
                col for col in df_hydropower_yearly.columns if "Estimated" in col
            ][0]

        if not winter_column_to_plot:
            winter_column_to_plot = [
                col
                for col in df_hydropower_seasonal.columns
                if ("Estimated" in col) and ("Winter" in col)
            ][0]

        if not summer_column_to_plot:
            summer_column_to_plot = [
                col
                for col in df_hydropower_seasonal.columns
                if ("Estimated" in col) and ("Summer" in col)
            ][0]

        # Yearly hydropower generation plot
        max_val = 20.5
        min_val = 13

        df_hydropower_yearly.plot(
            y=yearly_column_to_plot,
            ax=ax[0],
            label="Estimated Generation",
            legend=False,
            color=blue,
        )
        df_hydropower_yearly.plot(
            y=yearly_column_to_plot + "_fixed_system_2022",
            ax=ax[0],
            legend=False,
            color=green,
            alpha=0.8,
        )

        df_hydropower_yearly.plot(
            y=yearly_column_to_plot + "_fixed_system_1991",
            ax=ax[0],
            legend=False,
            color=orange,
            alpha=0.8,
        )
        ax[0].set_ylabel("Generation (in TWh)", fontsize=FONTSIZE_LABELS)
        ax[0].set_ylim(min_val - 1.0, max_val + 1.0)
        ax[0].set_title(
            subplots_titles[0] if subplots_titles else yearly_column_to_plot,
            fontsize=FONTSIZE_TITLE,
        )
        xmin, xmax = ax[0].get_xlim()

        # Winter hydropower generation plot
        max_val_winter = 9
        min_val_winter = 4

        df_hydropower_seasonal.plot(
            y=winter_column_to_plot,
            ax=ax[1],
            label="Estimated Generation",
            legend=False,
            color=blue,
        )

        df_hydropower_seasonal.plot(
            y=winter_column_to_plot + "_fixed_system_2022",
            ax=ax[1],
            legend=False,
            color=green,
            alpha=0.8,
        )

        df_hydropower_seasonal.plot(
            y=winter_column_to_plot + "_fixed_system_1991",
            ax=ax[1],
            legend=False,
            color=orange,
            alpha=0.8,
        )

        subplot_title = (
            f"{subplots_titles[1]}"
            if subplots_titles
            else f"Winter - {winter_column_to_plot}"
        )
        ax[1].set_title(subplot_title, fontsize=FONTSIZE_TITLE)
        ax[1].set_ylim(min_val_winter - 1.0, max_val_winter + 1.0)
        ax[1].set_yticks(np.arange(3, 10, 2))
        ax[1].set_xlim(xmin, xmax)

        # Summer hydropower generation plot
        max_val_summer = 14.25
        min_val_summer = 8

        df_hydropower_seasonal.plot(
            y=summer_column_to_plot,
            ax=ax[2],
            label="Estimated Generation (evolving capacities)",
            color=blue,
            legend=False,
        )

        df_hydropower_seasonal.plot(
            y=summer_column_to_plot + "_fixed_system_2022",
            ax=ax[2],
            label="Estimated Generation (2022 capacities)",
            legend=False,
            color=green,
            alpha=0.8,
        )

        df_hydropower_seasonal.plot(
            y=summer_column_to_plot + "_fixed_system_1991",
            ax=ax[2],
            label="Estimated Generation (1991 capacities)",
            legend=False,
            color=orange,
            alpha=0.8,
        )

        subplot_title = (
            f"{subplots_titles[2]}"
            if subplots_titles
            else f"Summer - {summer_column_to_plot}"
        )
        ax[2].set_title(subplot_title, fontsize=FONTSIZE_TITLE)
        ax[2].set_ylim(min_val_summer - 1.0, max_val_summer + 1.0)
        ax[2].set_xlim(xmin, xmax)

        for i in range(len(ax)):
            ax[i].tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
            ax[i].tick_params(axis="both", which="minor", labelsize=FONTSIZE_TICKS)

    def compute_hydropower_generation_different_capacities_trend_slopes(
        self,
        column_name: str,
        df_hydropower_generation: pd.DataFrame,
    ):
        """Computes the trend slopes of the yearly estimated generation
        for evolving and fixed capacities, along with their confidence intervals.

        Parameters
        ----------
        column_name : str
            Column containing the hydropower generation values
        df_hydropower_generation : pd.DataFrame
            Pandas DataFrame containing the hydropower generation values

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame containing the trend slopes and confidence intervals
            for the yearly estimated generation for evolving and fixed capacities
            (1991 and 2022)
        """
        columns_to_plot = {
            "Evolving capacities": column_name,
            "1991 capacities": f"{column_name}_fixed_system_1991",
            "2022 capacities": f"{column_name}_fixed_system_2022",
        }

        coeffs = []
        X = len(df_hydropower_generation.index)
        if "Winter" in column_name:
            X -= 1
        X = np.arange(X).reshape(-1, 1)

        for column in columns_to_plot:
            y = df_hydropower_generation[columns_to_plot[column]].values
            if "Winter" in column_name:
                y = y[1:]
            coef, _, _, conf_interval = self.compute_trend_statsmodel(X, y)
            coeffs.append(
                {
                    "name": column,
                    "coef": coef,
                    "lower": conf_interval[1][0],
                    "upper": conf_interval[1][1],
                }
            )

        return pd.DataFrame(coeffs)

    def plot_trend_slope(
        self,
        ax: np.ndarray[plt.Axes],
        with_percentage: bool = False,
        yearly_column_to_plot: str = None,
        winter_column_to_plot: str = None,
        summer_column_to_plot: str = None,
    ):
        """Plot a comparison of the trend slopes for yearly and seasonal
        estimated generation for three different aggregation methods: evolving capacities,
        fixed capacities of 1991 and fixed capacities of 2022.

        Parameters
        ----------
        ax : np.ndarray[plt.Axes]
            Array of Axes objects to plot on
        with_percentage : bool, optional
            Whether to multiply the hydropower generation of each power plant by
            the percentage of power that Switzerland is entitled to, by default False
        yearly_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the yearly
            values, by default None
        winter_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the winter
            values, by default None
        summer_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the summer
            values, by default None
        """
        df_hydropower_yearly = (
            self.create_dataframe_yearly_values(
                with_operation_start=True, with_percentage=with_percentage
            )
            .merge(
                self.create_dataframe_yearly_values(
                    with_operation_start=False, with_percentage=with_percentage
                ),
                left_index=True,
                right_index=True,
                suffixes=("", "_fixed_system_2022"),
            )
            .merge(
                self.create_dataframe_yearly_values(
                    with_operation_start=False,
                    with_percentage=with_percentage,
                    with_first_year_infrastructure=True,
                ),
                left_index=True,
                right_index=True,
                suffixes=("", "_fixed_system_1991"),
            )
        )

        df_hydropower_seasonal = (
            self.create_dataframe_seasonal_values(
                with_operation_start=True, with_percentage=with_percentage
            )
            .merge(
                self.create_dataframe_seasonal_values(
                    with_operation_start=False, with_percentage=with_percentage
                ),
                left_index=True,
                right_index=True,
                suffixes=("", "_fixed_system_2022"),
            )
            .merge(
                self.create_dataframe_seasonal_values(
                    with_operation_start=False,
                    with_percentage=with_percentage,
                    with_first_year_infrastructure=True,
                ),
                left_index=True,
                right_index=True,
                suffixes=("", "_fixed_system_1991"),
            )
        )

        if not yearly_column_to_plot:
            yearly_column_to_plot = [
                col for col in df_hydropower_yearly.columns if "Estimated" in col
            ][0]

        if not winter_column_to_plot:
            winter_column_to_plot = [
                col
                for col in df_hydropower_seasonal.columns
                if ("Estimated" in col) and ("Winter" in col)
            ][0]

        if not summer_column_to_plot:
            summer_column_to_plot = [
                col
                for col in df_hydropower_seasonal.columns
                if ("Estimated" in col) and ("Summer" in col)
            ][0]

        yearly_coeffs = (
            self.compute_hydropower_generation_different_capacities_trend_slopes(
                yearly_column_to_plot, df_hydropower_yearly
            )
        )
        winter_coeffs = (
            self.compute_hydropower_generation_different_capacities_trend_slopes(
                winter_column_to_plot, df_hydropower_seasonal
            )
        )
        summer_coeffs = (
            self.compute_hydropower_generation_different_capacities_trend_slopes(
                summer_column_to_plot, df_hydropower_seasonal
            )
        )

        yearly_error = [
            yearly_coeffs["coef"].values - yearly_coeffs["lower"].values,
            yearly_coeffs["upper"].values - yearly_coeffs["coef"].values,
        ]
        winter_error = [
            winter_coeffs["coef"].values - winter_coeffs["lower"].values,
            winter_coeffs["upper"].values - winter_coeffs["coef"].values,
        ]
        summer_error = [
            summer_coeffs["coef"].values - summer_coeffs["lower"].values,
            summer_coeffs["upper"].values - summer_coeffs["coef"].values,
        ]

        colors = [blue, orange, green]
        alphas = [(1,), (0.8,), (0.8,)]
        colors_with_alpha = [c + a for c, a in zip(colors, alphas)]

        min_x = np.min(
            [
                yearly_coeffs["lower"].min(),
                winter_coeffs["lower"].min(),
                summer_coeffs["lower"].min(),
            ]
        )
        max_x = np.max(
            [
                yearly_coeffs["upper"].max(),
                winter_coeffs["upper"].max(),
                summer_coeffs["upper"].max(),
            ]
        )

        ax[0].errorbar(
            yearly_coeffs["coef"],
            yearly_coeffs["name"],
            xerr=yearly_error,
            fmt="-o",
            color="none",
            ecolor=colors_with_alpha,
            linewidth=0,
            elinewidth=1,
        )
        ax[0].scatter(
            yearly_coeffs["coef"], yearly_coeffs["name"], color=colors_with_alpha
        )
        ax[0].axvline(0, color="black", alpha=0.7, linestyle="--")

        ylim = ax[0].get_ylim()
        ax[0].set_xlim([min_x - 0.02, max_x + 0.02])
        ax[0].set_ylim([ylim[0] - 0.3, ylim[1] + 0.3])
        ax[0].set_yticks(ax[0].get_yticks(), ["   ", "   ", "   "])
        ax[0].set_ylabel("    ", fontsize=FONTSIZE_LABELS)

        ax[1].errorbar(
            winter_coeffs["coef"],
            winter_coeffs["name"],
            xerr=winter_error,
            fmt="-o",
            color="none",
            ecolor=colors_with_alpha,
            linewidth=0,
            elinewidth=1,
        )
        ax[1].scatter(
            winter_coeffs["coef"], winter_coeffs["name"], color=colors_with_alpha
        )
        ax[1].axvline(0, color="black", alpha=0.7, linestyle="--")

        ax[1].set_xlim([min_x - 0.02, max_x + 0.02])
        ax[1].set_ylim([ylim[0] - 0.3, ylim[1] + 0.3])
        ax[1].set_yticks(ax[0].get_yticks(), ["  ", "  ", "  "])
        ax[1].set_xlabel("Slope (TWh/year)", fontsize=FONTSIZE_LABELS)

        ax[2].errorbar(
            summer_coeffs["coef"],
            summer_coeffs["name"],
            xerr=summer_error,
            fmt="-o",
            color="none",
            ecolor=colors_with_alpha,
            linewidth=0,
            elinewidth=1,
        )
        ax[2].scatter(
            summer_coeffs["coef"], summer_coeffs["name"], color=colors_with_alpha
        )
        ax[2].axvline(0, color="black", alpha=0.7, linestyle="--")
        ax[2].set_xlim([min_x - 0.02, max_x + 0.02])
        ax[2].set_ylim([ylim[0] - 0.3, ylim[1] + 0.3])
        ax[2].set_yticks(ax[0].get_yticks(), ["  ", "  ", "  "])

        for i in range(len(ax)):
            ax[i].tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
            ax[i].tick_params(axis="both", which="minor", labelsize=FONTSIZE_TICKS)

    def plot_trend_analysis(
        self,
        with_percentage: bool = False,
        yearly_column_to_plot: str = None,
        winter_column_to_plot: str = None,
        summer_column_to_plot: str = None,
        subplots_titles: List[str] = None,
        save: bool = False,
        output_filename: str = None,
    ):
        """Plot a comparison of the yearly and seasonal estimated generation and
        their trend slopes for three different aggregation methods: evolving capacities,
        fixed capacities of 1991 and fixed capacities of 2022.

        Parameters
        ----------
        ax : np.ndarray[plt.Axes]
            Array of Axes objects to plot on
        with_percentage : bool, optional
            Whether to multiply the hydropower generation of each power plant by
            the percentage of power that Switzerland is entitled to, by default False
        yearly_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the yearly
            values, by default None
        winter_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the winter
            values, by default None
        summer_column_to_plot : str, optional
            Name of the column in the DataFrame that contains the summer
            values, by default None
        save : bool, optional
            Whether to save the plot, by default False
        output_filename : str, optional
            The name of the file containing the plot, by default None
        """
        fig, axs = plt.subplots(2, 3, figsize=(15 * cm, 9 * cm), height_ratios=[1.8, 1])
        self.plot_infrastructure_trend(
            ax=axs[0, :],
            with_percentage=with_percentage,
            yearly_column_to_plot=yearly_column_to_plot,
            winter_column_to_plot=winter_column_to_plot,
            summer_column_to_plot=summer_column_to_plot,
            subplots_titles=subplots_titles,
        )
        axs[0, 0].text(
            x=-0.22,
            y=1.05,
            s="a",
            fontweight="bold",
            fontsize=FONTSIZE_TITLE,
            transform=axs[0, 0].transAxes,
        )

        self.plot_trend_slope(
            ax=axs[1, :],
            with_percentage=with_percentage,
            yearly_column_to_plot=yearly_column_to_plot,
            winter_column_to_plot=winter_column_to_plot,
            summer_column_to_plot=summer_column_to_plot,
        )
        axs[1, 0].text(
            x=-0.22,
            y=1.05,
            s="b",
            fontweight="bold",
            fontsize=FONTSIZE_TITLE,
            transform=axs[1, 0].transAxes,
        )
        plt.tight_layout()

        colors = [blue, orange, green]
        alphas = [(1,), (0.8,), (0.8,)]
        colors_with_alpha = [c + a for c, a in zip(colors, alphas)]
        legend_elements = [
            mlines.Line2D(
                [0],
                [0],
                color=colors_with_alpha[0],
                lw=1,
                label="Estimated Generation (evolving capacities)",
            ),
            mlines.Line2D(
                [0],
                [0],
                color=colors_with_alpha[1],
                lw=1,
                label="Estimated Generation (1991 capacities)",
            ),
            mlines.Line2D(
                [0],
                [0],
                color=colors_with_alpha[2],
                lw=1,
                label="Estimated Generation (2022 capacities)",
            ),
        ]

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.52, -0.1),
            fontsize=FONTSIZE_LABELS,
        )

        if save and output_filename:
            output_path = self.path_figs
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                output_path / output_filename,
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def compute_hydropower_generation_trend_slopes(
        self,
        df_hydropower_generation: pd.DataFrame,
        round_results: bool = True
    ) -> pd.DataFrame:
        """Computes the trend slopes for the estimated generation
        along with their confidence intervals. It could be on a monthly
        or seasonal level, or by hydropower plant. The index of the input
        DataFrame should represent the intended spatial or temporal granularity.

        Parameters
        ----------
        df_hydropower_generation : pd.DataFrame
            pandas DataFrame containing the estimated generation
            with indices indicating the spatial and the temporal granularity
            (e.g. the months if monthly trends or hydropower plants if per
            hydropower trends) and the columns the years in the study period.
        round_results : bool, optional
            Whether to round the coefficients, by default True

        Returns
        -------
        pd.DataFrame
            DataFrame containing the trend slopes for the monthly estimated
            generation along with their confidence intervals.
        """
        coeffs = []
        X = np.arange(len(df_hydropower_generation.columns)).reshape(-1, 1)

        for i in df_hydropower_generation.index:
            y = df_hydropower_generation.loc[i].values
            coef, _, _, conf_interval = self.compute_trend_statsmodel(X, y,
                                                                      round_results=round_results)
            coeffs.append(
                {
                    "name": i,
                    "coef": coef,
                    "lower": conf_interval[1][0],
                    "upper": conf_interval[1][1],
                }
            )

        coeffs = pd.DataFrame(coeffs)

        return coeffs

    def plot_trend_analysis_per_month(
        self, variable_name: str, save: bool = False, output_filename: str = None
    ) -> None:
        """Plot a comparison of the monthly trend slopes of the estimated generation
        along with their confidence intervals.

        Parameters
        ----------
        variable_name : str
            Variable name to select in the xarray Dataset
        save : bool, optional
            Whether to save the plot, by default False
        output_filename : str, optional
            The name of the file containing the plot, by default None
        """
        df_hydropower_generation_per_month = (
            self.create_dataframe_monthly_estimated_generation(
                variable_name=variable_name
            )
        )
        monthly_coeffs = self.compute_hydropower_generation_trend_slopes(
            df_hydropower_generation=df_hydropower_generation_per_month
        )
        error = [
            monthly_coeffs["coef"].values - monthly_coeffs["lower"].values,
            monthly_coeffs["upper"].values - monthly_coeffs["coef"].values,
        ]

        fig, ax = plt.subplots(figsize=(10 * cm, 5 * cm))

        ax.errorbar(
            monthly_coeffs["coef"],
            monthly_coeffs["name"],
            xerr=error,
            fmt="-o",
            color="black",
            linewidth=0,
            elinewidth=1,
        )
        ax.axvline(0, color="black", alpha=0.5, linestyle="--")

        ax.set_xlim([-0.035, 0.035])
        ax.set_xlabel("Trend (TWh/year)", fontsize=FONTSIZE_LABELS)
        ax.set_yticks(range(1, 13))
        ax.set_ylabel("Month", fontsize=FONTSIZE_LABELS)
        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
        ax.tick_params(axis="both", which="minor", labelsize=FONTSIZE_TICKS)

        if save and output_filename:
            output_path = self.path_figs
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path / output_filename, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_winter_trend_map_and_distribution(
        self,
        variable_name: str,
        with_percentage: bool = False,
        with_first_year_infrastructure: bool = False,
        save: bool = False,
        output_filename: str = None,
    ) -> None:
        """Plot a map of the quantiles of yearly or seasonal generation for each
        RoR hydropower plants in the WASTA database.

        Parameters
        ----------
        variable_name : str
            Variable name to select in the xarray Dataset of seasonal
            estimated generation
        with_percentage : bool, optional
            Whether to multiply the hydropower generation of each power plant by
            the percentage of power that Switzerland is entitled to, by default False
        with_first_year_infrastructure : bool, optional
            Whether to fix the hydropower fleet to its state in the first year
            present in the study period, by default False
        save : bool, optional
            Whether to save the plot, by default False
        output_filename : str, optional
            The name of the file containing the plot, by default None
        """
        df_hydropower_generation_per_hp_winter = (
            self.create_dataframe_seasonal_estimated_values_per_hp(
                variable_name=variable_name,
                with_percentage=with_percentage,
                with_first_year_infrastructure=with_first_year_infrastructure,
            )
        )
        winter_coeffs_per_hp = self.compute_hydropower_generation_trend_slopes(
            df_hydropower_generation_per_hp_winter,
            round_results=False
        )

        winter_coeffs_per_hp["winter_mean"] = (
            (df_hydropower_generation_per_hp_winter).mean(axis=1).to_numpy()
        )
        winter_coeffs_per_hp["relative_change"] = (
            winter_coeffs_per_hp["coef"] / winter_coeffs_per_hp["winter_mean"] * 100
        )
        winter_coeffs_per_hp["relative_change_decadal"] = (
            winter_coeffs_per_hp["relative_change"] * 10
        )
        winter_coeffs_per_hp["relative_change_decadal_category"] = winter_coeffs_per_hp[
            "relative_change_decadal"
        ].apply(lambda c: -1 if c < -1 else (1 if c > 1 else 0))

        tick_labels = ["< -1%", "-1 to 1%", "> 1%"]
        # ticks = [-1, 0, 1]
        colors = ["red", grey, "blue"]
        cmap = ListedColormap(colors)

        gdf_hydropower_coef_map = self.gdf_hydropower_locations.copy()

        gdf_hydropower_coef_map = gdf_hydropower_coef_map[
            ~pd.isna(gdf_hydropower_coef_map["Canton"])
        ]
        gdf_hydropower_coef_map = gdf_hydropower_coef_map[
            gdf_hydropower_coef_map["WASTANumber"].isin(
                self.ds_hydropower_generation.hydropower.to_numpy()
            )
        ]

        gdf_hydropower_coef_map = pd.merge(
            gdf_hydropower_coef_map,
            winter_coeffs_per_hp,
            left_on="WASTANumber",
            right_on="name",
        )

        gdf_hydropower_coef_map["quantile_category"] = gdf_hydropower_coef_map[
            "Capacity"
        ].apply(
            lambda c: np.floor(
                round((gdf_hydropower_coef_map["Capacity"] < c).mean(), 3) * 10
            )
        )

        capacity_upper_10 = gdf_hydropower_coef_map[
            (
                gdf_hydropower_coef_map["name"].isin(
                    self.ds_hydropower_generation.hydropower.values
                )
            )
        ]["Capacity"].quantile(0.9)

        fig, ax = plt.subplots(1, 2, figsize=(15 * cm, 7 * cm), width_ratios=(2, 1))

        self.gdf_switzerland.plot(
            ax=ax[0],
            color="white",
            edgecolor="black",
            linewidth=0.5,
        )

        gdf_hydropower_coef_map[
            gdf_hydropower_coef_map["Capacity"] < capacity_upper_10
        ].plot(
            "relative_change_decadal_category",
            ax=ax[0],
            legend=False,
            marker=".",
            s=4,
            cmap=cmap,
        )

        gdf_hydropower_coef_map[
            gdf_hydropower_coef_map["Capacity"] >= capacity_upper_10
        ].plot(
            "relative_change_decadal_category",
            ax=ax[0],
            legend=False,
            marker=".",
            s=20,
            edgecolor="black",
            linewidth=1,
            cmap=cmap,
        )
        ax[0].axis("off")
        ax[0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        ax[0].set_title("a", fontweight="bold", loc="left", fontsize=FONTSIZE_TITLE, y=1)

        df_frequency_categories = (
            gdf_hydropower_coef_map.groupby(
                ["quantile_category", "relative_change_decadal_category"]
            )
            .size()
            .unstack(fill_value=0)
        )
        df_frequency_categories.plot.bar(
            ax=ax[1], stacked=True, legend=False, color=colors
        )

        ax[1].set_xticks(
            range(len(df_frequency_categories)), CBAR_LEVELS_QUANTILES, rotation=45
        )
        ax[1].set_xlabel("Quantiles of capacity", fontsize=FONTSIZE_LABELS)
        ax[1].set_ylabel("Frequency", fontsize=7)
        ax[1].set_title("b", fontweight="bold", loc="left", fontsize=FONTSIZE_TITLE, x=-0.24, y=1)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        circle = mlines.Line2D(
            [],
            [],
            color="white",
            marker=".",
            markeredgewidth=1,
            markeredgecolor="black",
            markersize=10,
            label="Largest 10% of RoR\nhydropower plants",
        )
        ax[0].legend(
            handles=[circle],
            markerscale=1,
            loc="lower left",
            bbox_to_anchor=(0, -0.2),
            fontsize=FONTSIZE_LABELS,
            frameon=False,
        )

        handles, labels = ax[1].get_legend_handles_labels()
        fig.legend(
            handles,
            tick_labels,
            loc="lower center",
            ncol=3,
            # bbox_to_anchor=(0.5, -0.6),
            fontsize=7,
            title="Decadal relative change (%)",
            title_fontsize=FONTSIZE_LABELS,
            frameon=False
        )
        ax[1].tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
        ax[1].tick_params(axis="both", which="minor", labelsize=FONTSIZE_TICKS)

        if save and output_filename:
            output_path = self.path_figs
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path / output_filename, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_quantile_maps(
        self,
        yearly: bool,
        variable_name: str,
        with_operation_start: bool = False,
        save: bool = False,
        output_filename: str = None,
    ) -> None:
        """Plot a map of the quantiles of yearly or seasonal generation for each
        RoR hydropower plants in the WASTA database.

        Parameters
        ----------
        yearly : bool
            Whether to plot  on yearly or seasonal
            timeseries of hydropower generation
        variable_name : str
            Variable name to select in the xarray Dataset
        save : bool, optional
            Whether to save the plot, by default False
        output_filename : str, optional
            The name of the file containing the plot, by default None
        """
        num_bins = 10
        colormap = plt.get_cmap("RdBu")
        colors = [colormap(i / num_bins) for i in range(num_bins)]
        cmap = ListedColormap(colors)

        nb_plots_row = 10
        df_generation = self.create_dataframe_with_quantiles(yearly, variable_name)

        years = df_generation.index.get_level_values("time").unique()
        start_decade = int(np.floor(years[0] / 10.0) * 10)

        fig, axs = plt.subplots(
            nb_plots_row,
            int(np.ceil(len(years) / nb_plots_row)),
            figsize=(15 * cm, 20 * cm),
        )

        for i in range(len(axs)):
            for j in range(len(axs[i, :])):
                axs[i, j].axis("off")

        for i, year in enumerate(years):
            row = year % 10
            col = (year - start_decade) // 10

            wasta = (
                self.gdf_hydropower_locations[
                    (
                        self.gdf_hydropower_locations["BeginningOfOperation"]
                        <= year
                    )
                    & (
                        self.gdf_hydropower_locations["WASTANumber"].isin(
                            self.ds_hydropower_generation.hydropower.to_numpy()
                        )
                    )
                ]["WASTANumber"].tolist()
                if with_operation_start
                else self.ds_hydropower_generation.hydropower.to_numpy()
            )

            capacity_upper_10 = self.gdf_hydropower_locations[
                self.gdf_hydropower_locations["WASTANumber"].isin(wasta)
            ]["Capacity"].quantile(0.9)

            gdf_hydropower_quantile_map = self.gdf_hydropower_locations.copy()
            gdf_hydropower_quantile_map = gdf_hydropower_quantile_map[
                ~pd.isna(gdf_hydropower_quantile_map["Canton"])
            ]
            gdf_hydropower_quantile_map = gdf_hydropower_quantile_map[
                gdf_hydropower_quantile_map["WASTANumber"].isin(
                    self.ds_hydropower_generation.hydropower.to_numpy()
                )
            ]

            gdf_hydropower_quantile_map = pd.merge(
                gdf_hydropower_quantile_map,
                df_generation.loc[(year, slice(None)), :].reset_index(),
                left_on="WASTANumber",
                right_on="hydropower",
            )

            gdf_hydropower_quantile_map["quantile_categorical"] = (
                gdf_hydropower_quantile_map[
                    "quantile"
                ].apply(lambda q: np.floor(q * 10))
            )

            self.gdf_switzerland.plot(
                ax=axs[row, col],
                color="white",
                edgecolor="black",
                linewidth=0.5,
            )

            gdf_hydropower_quantile_map[
                gdf_hydropower_quantile_map["capacity"] < capacity_upper_10
            ].plot(
                "quantile_categorical",
                ax=axs[row, col],
                legend=False,
                marker=".",
                s=0.8,
                cmap=cmap,
            )

            gdf_hydropower_quantile_map[
                gdf_hydropower_quantile_map["capacity"] >= capacity_upper_10
            ].plot(
                "quantile_categorical",
                ax=axs[row, col],
                legend=False,
                marker=".",
                s=11,
                edgecolor="black",
                linewidth=0.6,
                cmap=cmap,
            )

            axs[row, col].set_title(year, fontsize=6, loc="left", y=0.7)
            axs[row, col].axis("off")
            axs[row, col].tick_params(
                left=False, labelleft=False, bottom=False, labelbottom=False
            )

        fig.tight_layout()
        plt.subplots_adjust(bottom=None, top=None, wspace=0, hspace=0)
        circle = mlines.Line2D(
            [],
            [],
            color="white",
            marker=".",
            markeredgewidth=1,
            markeredgecolor="black",
            markersize=10,
            label="Largest 10% of RoR\nhydropower plants",
        )
        plt.legend(
            handles=[circle],
            markerscale=1,
            loc="lower right",
            bbox_to_anchor=(1.15, 1.1),
            fontsize=FONTSIZE_LABELS,
            frameon=False,
        )
        # Create the colorbar with specified number of bins
        colorbar = plt.cm.ScalarMappable(cmap=cmap)
        colorbar.set_array([])
        colorbar.set_clim(-0.5, num_bins - 0.5)
        cbar = plt.colorbar(
            colorbar,
            ax=axs,
            ticks=range(num_bins),
            orientation="vertical",
            aspect=50,
            fraction=0.15,
            shrink=0.3,
            pad=0,
        )
        cbar.set_ticklabels(CBAR_LEVELS_QUANTILES)
        cbar.ax.tick_params(labelsize=FONTSIZE_TICKS)
        cbar.ax.set_title("Percentile", fontsize=FONTSIZE_LABELS)

        pos1 = cbar.ax.get_position()
        pos2 = [pos1.x0 - 0.11, pos1.y0 - 0.1, pos1.width, pos1.height]
        cbar.ax.set_position(pos2)

        if save and output_filename:
            output_path = self.path_figs
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path / output_filename, dpi=200, bbox_inches="tight")

        plt.show()

    def plot_hist_prod_quantile_threshold_per_decade(
        self,
        yearly: bool,
        variable_name: str,
        quantile_threshold: float,
        higher_than: bool,
        with_operation_start: bool = False,
        save: bool = False,
        output_filename: str = None,
    ):
        df_quantiles = self.create_dataframe_with_quantiles(yearly, variable_name)
        df_quantiles["quantile_threshold"] = df_quantiles["quantile"].apply(lambda q: q >= quantile_threshold if higher_than else q <= quantile_threshold)
        for year in df_quantiles.index.get_level_values(level=0).unique():
            wasta = (
                self.gdf_hydropower_locations[
                    (
                        self.gdf_hydropower_locations["BeginningOfOperation"]
                        <= year
                    )
                    & (
                        self.gdf_hydropower_locations["WASTANumber"].isin(
                            self.ds_hydropower_generation.hydropower.to_numpy()
                        )
                    )
                ]["WASTANumber"].tolist()
                if with_operation_start
                else self.ds_hydropower_generation.hydropower.to_numpy()
            )
            df_quantiles.loc[(year, wasta), "quantile_threshold"] = False

        df_quantiles_threshold_per_year = df_quantiles.groupby("time").sum("quantile_threshold")[["quantile_threshold"]]
        df_quantiles_threshold_per_year = df_quantiles_threshold_per_year.groupby((df_quantiles_threshold_per_year.index//10)*10).sum()

        _, ax = plt.subplots(figsize=(7.5*cm, 6*cm))
        df_quantiles_threshold_per_year.plot.bar(ax=ax, color=blue, legend=False)
        ax.set_xlabel("")
        ax.set_ylabel("Number of hydropower plants", fontsize=FONTSIZE_LABELS)

        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
        ax.tick_params(axis="both", which="minor", labelsize=FONTSIZE_TICKS)
        plt.xticks(rotation=45)

        if save and output_filename:
            output_path = self.path_figs
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path / output_filename, dpi=300, bbox_inches="tight")

        plt.show()


def plot_pre_post_bias_correction_validation(
    analysis_pre_correction: NationalAnalysisHydropower,
    analysis_post_correction: NationalAnalysisHydropower,
    with_percentage: bool = False,
    yearly_column_to_plot: str = None,
    winter_column_to_plot: str = None,
    summer_column_to_plot: str = None,
    subplots_titles: List[str] = None,
    save: bool = False,
    output_filename: str = None,
):
    """Plot a comparison of the yearly and seasonal estimated generation with
        the reported generation, before and after bias correcting the estimates.

    Parameters
    ----------
    analysis_pre_correction : NationalAnalysisHydropower
        NationalAnalysisHydropower object that analyses hydropower generation
        prior to bias correction
    analysis_post_correction : NationalAnalysisHydropower
        NationalAnalysisHydropower object that analyses hydropower generation
        after bias correction
    with_percentage : bool, optional
        Whether to multiply the hydropower generation of each power plant by
        the percentage of power that Switzerland is entitled to, by default False
    yearly_column_to_plot : str, optional
        Name of the column in the DataFrame that contains the yearly
        values, by default None
    winter_column_to_plot : str, optional
        Name of the column in the DataFrame that contains the winter
        values, by default None
    summer_column_to_plot : str, optional
        Name of the column in the DataFrame that contains the summer
        values, by default None
    save : bool, optional
        Whether to save the plot, by default False
    output_filename : str, optional
        The name of the file containing the plot, by default None
    """
    fig, axs = plt.subplots(2, 3, figsize=(15 * cm, 10 * cm))
    analysis_pre_correction.plot_validation(
        ax=axs[0],
        with_percentage=with_percentage,
        yearly_column_to_plot=yearly_column_to_plot,
        winter_column_to_plot=winter_column_to_plot,
        summer_column_to_plot=summer_column_to_plot,
        subplots_titles=subplots_titles
    )
    axs[0, 0].text(
        x=-0.22,
        y=1.05,
        s="a",
        fontweight="bold",
        fontsize=FONTSIZE_TITLE,
        transform=axs[0, 0].transAxes,
    )

    analysis_post_correction.plot_validation(
        ax=axs[1],
        with_percentage=with_percentage,
        yearly_column_to_plot=yearly_column_to_plot,
        winter_column_to_plot=winter_column_to_plot,
        summer_column_to_plot=summer_column_to_plot,
        subplots_titles=subplots_titles
    )
    axs[1, 0].text(
        x=-0.22,
        y=1.05,
        s="b",
        fontweight="bold",
        fontsize=FONTSIZE_TITLE,
        transform=axs[1, 0].transAxes,
    )

    handles, labels = axs[1, 2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=FONTSIZE_LABELS,
    )

    plt.tight_layout()

    if save and output_filename:
        output_path = analysis_pre_correction.path_figs
        output_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(
            output_path / output_filename,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
