from typing import Any, List

import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points
from shapely.geometry import box
from typing import Optional

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

def build_prevah_grid_points(
    ds_prevah: xr.Dataset,
    resolution: float,
    crs: str = "EPSG:2056",
) -> gpd.GeoDataFrame:
    df_runoff = ds_prevah.isel(time=0)
    new_x = (
        np.round(df_runoff.x.values / resolution) * resolution
    ).astype(df_runoff.x.dtype)
    new_y = (
        np.round(df_runoff.y.values / resolution) * resolution
    ).astype(df_runoff.y.dtype)
    df_runoff = df_runoff.assign_coords(x=new_x, y=new_y)
    df_runoff = df_runoff.to_dataframe().reset_index()

    # Build grid point GeoDataFrame (centers) and grid cell GeoDataFrame (polygons)
    return gpd.GeoDataFrame(
        df_runoff,
        geometry=gpd.points_from_xy(df_runoff.x, df_runoff.y),
        crs=crs,
    )[["y", "x", "geometry"]]

def build_prevah_grid_cells(
    x_coords: List[float],
    y_coords: List[float],
    resolution: float,
    crs: str = "EPSG:2056",
) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame of PREVAH rectangular grid cells from 1D coordinate arrays.

    Parameters
    ----------
    x_coords : List[float]
        1D sequence of x coordinates (cell centers).
    y_coords : List[float]
        1D sequence of y coordinates (cell centers).
    resolution : float
        Side length of each square grid cell (same units as coordinates, e.g. meters for EPSG:2056).
    crs : str, optional
        Coordinate reference system for the returned GeoDataFrame, by default "EPSG:2056".

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with columns ['x', 'y', 'geometry'] where 'geometry' contains
        shapely.box polygons representing square grid cells centered on (x, y).
    """
    half = resolution / 2.0
    rows = [
        {"x": float(x), "y": float(y), "geometry": box(x - half, y - half, x + half, y + half)}
        for y in y_coords
        for x in x_coords
    ]
    gdf = gpd.GeoDataFrame(rows, crs=crs)
    return gdf


def map_polygons_to_grid_by_intersection(
    gdf_polygons: gpd.GeoDataFrame,
    gdf_grid_cells: gpd.GeoDataFrame,
    min_fraction: float = 0.0,
) -> pd.DataFrame:
    """
    Compute intersection area between polygons and grid cells and return weights.

    Parameters
    ----------
    gdf_polygons : geopandas.GeoDataFrame
        GeoDataFrame containing polygons with an 'EZGNR' column and 'geometry'.
    gdf_grid_cells : geopandas.GeoDataFrame
        GeoDataFrame containing grid cell polygons with 'x', 'y' and 'geometry' columns
        (grid cell centers stored in 'x' and 'y').
    min_fraction : float, optional
        Minimum fraction of the grid cell area that must be covered by the polygon
        to keep the mapping (intersect_area / cell_area). Default is 0.0 (keep all).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - x : float
            x coordinate of the grid cell center
        - y : float
            y coordinate of the grid cell center
        - EZGNR : int
            Polygon identifier that intersects the cell
        - cell_area : float
            Area of the grid cell geometry (same units as input CRS)
        - intersect_area : float
            Area of the intersection between the cell and the polygon
        - weight : float
            Fraction of the cell area covered by the polygon (intersect_area / cell_area)
    """
    # quick candidate selection via spatial join
    candidates = gpd.sjoin(
        gdf_grid_cells,
        gdf_polygons[["EZGNR", "geometry"]],
        how="left",
        predicate="intersects",
    )
    candidates = candidates.dropna(subset=["EZGNR"]).reset_index(drop=True)

    if candidates.empty:
        return pd.DataFrame(columns=["x", "y", "EZGNR", "cell_area", "intersect_area", "weight"])

    # build map for faster geometry lookup
    poly_map = {row.EZGNR: row.geometry for row in gdf_polygons.itertuples(index=False)}
    results = []
    for row in candidates.itertuples(index=False):
        cell_geom = row.geometry
        ezg = int(row.EZGNR)
        poly_geom = poly_map.get(ezg)
        if poly_geom is None:
            continue
        inter = cell_geom.intersection(poly_geom)
        if inter.is_empty:
            continue
        cell_area = float(cell_geom.area)
        inter_area = float(inter.area)
        if cell_area <= 0:
            continue
        weight = inter_area / cell_area
        if weight >= min_fraction:
            results.append(
                {
                    "x": float(row.x),
                    "y": float(row.y),
                    "EZGNR": ezg,
                    "cell_area": cell_area,
                    "intersect_area": inter_area,
                    "weight": weight,
                }
            )

    return pd.DataFrame(results)


def fill_missing_polygons_by_nearest(
    gdf_polygons: gpd.GeoDataFrame,
    gdf_grid_points: gpd.GeoDataFrame,
    missing_ezg_list: List[int],
    max_distance: Optional[float] = None,
) -> pd.DataFrame:
    """
    Assign the nearest PREVAH grid point to polygons that had no overlapping grid cell.

    Parameters
    ----------
    gdf_polygons : geopandas.GeoDataFrame
        GeoDataFrame of polygons containing at least 'EZGNR' and 'geometry' columns.
    gdf_grid_points : geopandas.GeoDataFrame
        GeoDataFrame of PREVAH grid points (centers) containing 'x', 'y' and 'geometry'.
    missing_ezg_list : list of int
        List of EZGNR identifiers for polygons that lacked overlapping grid cells.
    max_distance : float or None, optional
        Maximum search distance (in the units of the CRS, e.g. meters for EPSG:2056).
        If None the function computes a default value equal to half the diagonal of a grid cell
        inferred from the two smallest distinct x and y steps. Default is None.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per successfully assigned polygon containing columns:
        - EZGNR (int): polygon identifier
        - x (float): x coordinate of assigned grid point (cell center)
        - y (float): y coordinate of assigned grid point (cell center)
        - method (str): assignment method, e.g. "nearest"
        - distance_m (float): distance from polygon representative point to assigned grid point
    """
    assigned = []
    # spatial index for grid points
    pts_sindex = gdf_grid_points.sindex
    # compute default max_distance if not provided
    if max_distance is None:
        xs = sorted(gdf_grid_points.x.unique())
        ys = sorted(gdf_grid_points.y.unique())
        if len(xs) > 1 and len(ys) > 1:
            dx = abs(xs[1] - xs[0])
            dy = abs(ys[1] - ys[0])
            max_distance = 0.5 * (dx**2 + dy**2) ** 0.5  # half diagonal
        else:
            max_distance = 1000.0
    print(f"Using max_distance = {max_distance} for nearest point search.")
    # Map polygons by EZGNR for speed
    poly_map = {row.EZGNR: row.geometry for row in gdf_polygons.itertuples(index=False)}
    for ezg in missing_ezg_list:
        poly = poly_map.get(ezg)
        if poly is None:
            continue
        # choose interior point to avoid exterior centroid for thin shapes
        pt = poly.representative_point()
        # find nearest candidate index (1)
        try:
            nearest_idx = list(pts_sindex.nearest(pt.bounds, 1))[0]
        except Exception:
            continue
        nearest_row = gdf_grid_points.iloc[nearest_idx]
        dist = float(pt.distance(nearest_row.geometry))
        if dist <= max_distance:
            assigned.append(
                {
                    "EZGNR": int(ezg),
                    "x": float(nearest_row.x),
                    "y": float(nearest_row.y),
                    "method": "nearest",
                    "distance_m": float(dist),
                }
            )
    print(f"Assigned {len(assigned)} polygons by nearest point.")
    print(f"Average distance: {np.mean([a['distance_m'] for a in assigned])} m")
    return pd.DataFrame(assigned)

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
