from typing import Any, List

import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points


def get_points_in_polygons(
    gdf_polygons: gpd.GeoDataFrame,
    gdf_points: gpd.GeoDataFrame,
    predicate: str = "intersects",
) -> gpd.GeoDataFrame:
    """Find all points in a GeoDataFrame that are present in polygons in another
    GeoDataFrame.

    Parameters
    ----------
    gdf_polygons : gpd.GeoDataFrame
        GeoDataFrame containing the polygons
    gdf_points : gpd.GeoDataFrame
        GeoDataFrame containing the points
    predicate : str, optional
        Predicate for the spatial joing to be carried between
        the two GeoDataFrames, by default "intersects"

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame matching polygons from gdf_polygons with the points
        from gdf_points that they contain.
    """
    list_gdf_points_in_polygons = []
    for _, row in gdf_polygons.iterrows():
        gdf_points_in_polygons = gpd.sjoin(
            gdf_polygons.loc[gdf_polygons["EZGNR"] == row["EZGNR"]],
            gdf_points,
            predicate=predicate,
            how="right",
        )
        gdf_points_in_polygons = gdf_points_in_polygons[
            ~gdf_points_in_polygons["index_left"].isna()
        ]
        list_gdf_points_in_polygons.append(gdf_points_in_polygons)

    list_gdf_points_in_polygons = [
        gdf for gdf in list_gdf_points_in_polygons if not (gdf.empty)
    ]

    gdf_concat = gpd.GeoDataFrame(
        pd.concat(list_gdf_points_in_polygons, ignore_index=True),
        crs=list_gdf_points_in_polygons[0].crs,
    )

    return gdf_concat

def get_df_upstream_polygons(
    df_polygon_connectivity: pd.DataFrame,
    from_column: str = "fEZGNR",
    to_column: str = "tEZGNR",
) -> pd.DataFrame:
    """Given a Pandas DataFrame containing the connectivity between catchment areas (from -> to connectivity),
    finds for each catchment areas their directly connected and contiguous catchment areas.

    Parameters
    ----------
    df_polygon_connectivity : pd.DataFrame
        Pandas DataFrame describing connectivity between different catchment areas.
        It must contain a column indicating the origin catchment area and one indicating
        the target catchment area.
    from_column : str, optional
        Name of column in the Pandas DataFrame containing the origin catchment area, by default "fEZGNR"
    to_column : str, optional
        Name of column in the Pandas DataFrame containing the target catchment area, by default "tEZGNR"

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame indicating the upstream catchment areas for each catchment area in
        the original Pandas DataFrame
    """
    df_upstream_polygons = (
        df_polygon_connectivity.groupby(to_column)[from_column]
        .apply(list)
        .reset_index()
    )
    df_upstream_polygons.columns = ["EZGNR", "upstream_EZGNR"]

    return df_upstream_polygons


def find_upstream_polygons_recursive(
    df_upstream_polygons: pd.DataFrame,
    origin_polygon: int,
    connected_polygons: set = None,
    origin_column: str = "EZGNR",
    upstream_catchments_columns: str = "upstream_EZGNR",
) -> List[int]:
    """Given a Pandas DataFrame containing catchment areas and their direct upstream catchments,
    the ID of the origin catchment areas and a set of the IDs of its  connected catchments,
    recursively find all the upstream catchments from the origin.

    Parameters
    ----------
    df_upstream_polygons : pd.DataFrame
        Pandas DataFrame containing catchment areas and their direct upstream catchments.
    origin_polygon : int
        ID of the origin catchment
    connected_polygons : set, optional
        Set of the IDs of retrieved catchment areas, by default None
    origin_column : str, optional
        Name of column in the Pandas DataFrame containing the origin
        catchment area, by default "EZGNR"
    upstream_catchments_columns : str, optional
        Name of column in the Pandas DataFrame containing the upstream
        catchment areas, by default "upstream_EZGNR"

    Returns
    -------
    List[int]
        List of all the upstream catchment areas from the origin
    """
    if connected_polygons is None:
        connected_polygons = set()

    upstream_polygons = df_upstream_polygons.loc[
        df_upstream_polygons[origin_column] == origin_polygon,
        upstream_catchments_columns,
    ]
    if not upstream_polygons.empty:
        for upstream_polygon in upstream_polygons.iloc[0]:
            if upstream_polygon not in connected_polygons:
                connected_polygons.add(upstream_polygon)
                find_upstream_polygons_recursive(
                    df_upstream_polygons,
                    upstream_polygon,
                    connected_polygons,
                    origin_column,
                    upstream_catchments_columns,
                )

    return list(connected_polygons)


def get_nearest_values(
    row, other_gdf, point_column="geometry", value_column="geometry"
) -> Any:
    """Find the nearest point and return the corresponding value from specified value column."""

    # Create an union of the other GeoDataFrame's geometries:
    other_points = other_gdf["geometry"].unary_union

    # Find the nearest points
    nearest_geoms = nearest_points(row[point_column], other_points)

    # Get corresponding values from the other df
    nearest_data = other_gdf.loc[other_gdf["geometry"] == nearest_geoms[1]]

    nearest_value = nearest_data[value_column].values[0]

    return nearest_value


def get_total_catchment_area(
    gdf_polygons: gpd.GeoDataFrame, list_ezgnr: List[int]
) -> float:
    """Returns the total area of the selected polygons.

    Parameters
    ----------
    gdf_polygons : gpd.GeoDataFrame
        GeoDataFrame containing polygons to select from
    list_ezgnr : List[int]
        The list of polygons, identified with their EZGNR,
        to select in the GeoDataFrame

    Returns
    -------
    float
        Total area of the selected polygons
    """
    return (
        gdf_polygons[gdf_polygons["EZGNR"].isin(list_ezgnr)]
        .apply(lambda row: row["geometry"].area * 1e-6, axis=1)
        .sum()
    )


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list with 2 levels.

    Parameters
    ----------
    nested_list : List[List[Any]]
        A 2-level nested list to flatten

    Returns
    -------
    List[Any]
        Flattened list
    """
    return [elem for sublist in nested_list for elem in sublist]
