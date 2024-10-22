import pathlib
import shutil
import tarfile
import tempfile
import time
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from src.readbin_wsl import read_data_gz
from src.utils_time import detect_date_in_filename
from src.var_attributes import ATTRIBUTES, HYDRO_NETCDF_ENCODINGS

AVAILABLE_PRODUCTS = ["ETR", "FCP", "P", "R2", "RGS", "SSO", "SWA"]


def extract_date_hydro_tgz(filename: str, prefix: str = "") -> datetime:
    """Extracts the date from the name of a gz PREVAH data file.

    Parameters
    ----------
    filename : str
        Name of the gz PREVAH data file
    prefix : str, optional
        String that identifies the type of simulation (e.g. Mob500),
        that precedes the date in the filename, by default ""

    Returns
    -------
    datetime
        Datetime object corresponding to the date in the
        gz PREVAH data filename.
    """
    date_str = filename
    if prefix:
        date_str = date_str.split(prefix)[1]
    if "." in date_str:
        date_str = date_str.split(".")[0]
    return datetime.strptime(detect_date_in_filename(date_str), "%Y%m%d")


def transform_coords_old_to_new_swiss(ds: xr.Dataset) -> xr.Dataset:
    """Transform the coordinates 'x' and 'y' in an xarray Dataset from
    the old Swiss System (LV03/EPSG:21871) to the new (LV95/EPSG:2056).

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset to transform

    Returns
    -------
    xr.Dataset
        xarray Dataset with coordinates in the new Swiss system
    """
    xv, yv = np.meshgrid(ds.x.values, ds.y.values, indexing="xy")

    transformer = Transformer.from_crs("EPSG:21781", "EPSG:2056")
    x2, y2 = transformer.transform(xv, yv)

    x_epsg_2056 = x2[0]
    y_epsg_2056 = y2[:, 0]

    ds["x"] = x_epsg_2056
    ds["y"] = y_epsg_2056

    return ds


def extract_hydro_tgz(
    file_path: pathlib.Path,
    temp_output_dir: pathlib.Path,
    product: str,
    prefix_filename: str = "",
) -> xr.Dataset:
    """Uncompresses a .tgz PREVAH file, extracts data from the resulting
    .gz file and stores them in an xarray Dataset.

    Parameters
    ----------
    file_path : pathlib.Path
        File path to the .tgz file
    temp_output_dir : pathlib.Path
        Temporary output directory for the uncompressed files
    product : str
        PREVAH data product to extract (e.g. RGS)
    prefix_filename : str, optional
        String that identifies the type of simulation (e.g. Mob500),
        that precedes the date in the filename, by default ""

    Returns
    -------
    xr.Dataset
        xarray Dataset with the extracted data from the uncompressed .gz file

    Raises
    ------
    ValueError
        If the product passed as argument is not yet considered
    """
    if product not in AVAILABLE_PRODUCTS:
        raise ValueError("The product you tried to read is not taken into account yet.")

    temp_hydro_tar_dir = temp_output_dir / file_path.stem
    temp_hydro_tar_dir.mkdir(exist_ok=True)
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(temp_hydro_tar_dir)

    filepath_product_gz = list(temp_hydro_tar_dir.glob(f"**/*{product}.gz"))[0]
    result = read_data_gz(filepath_product_gz)
    time = extract_date_hydro_tgz(filepath_product_gz.stem, prefix=prefix_filename)

    shutil.rmtree(temp_hydro_tar_dir)

    # Create xr.Dataset
    ds = xr.Dataset(
        data_vars={
            product.lower(): (["y", "x"], result.data),
        },
        coords=dict(
            time=time,
            x=(["x"], result.x),
            y=(["y"], result.y),
        ),
        attrs=ATTRIBUTES[product],
    )

    return ds


def combine_ds_to_netcdf(
    list_ds: List[xr.Dataset], convert_coords: bool, output_dir_path: pathlib.Path, output_filename: str = ""
) -> None:
    """Combines multiple xarray Datasets into one and saves the combined
    xarray Dataset into a netcdf file.

    Parameters
    ----------
    list_ds : List[xr.Dataset]
        List of xarray Datasets to combine
    convert_coords : bool
        Whether to conver the coordinates from the old Swiss
        system (LV95) to the new system (LV03)
    output_dir_path : pathlib.Path
        Path to output directory where the netcdf will be saved
    output_filename : str, optional
        Name of the netcdf file to save, by default ""

    Raises
    ------
    ValueError
        If the list of xarray Datasets is empty
    """
    if len(list_ds) == 0:
        raise ValueError("list_ds is empty.")

    product = list_ds[0].attrs["product"].upper()
    encoding = {
        var: HYDRO_NETCDF_ENCODINGS.copy() for var in list(list_ds[0].data_vars.keys())
    }
    concat_ds = xr.concat(list_ds, dim="time", coords="all").sortby("time")
    years = sorted(set(concat_ds.time.dt.year.values))
    print(f" | Years {years} | ", end="")
    if len(years) > 1:
        print(f"| More than 1 year in this dataset ({years}) | ", end="")
        concat_ds = concat_ds.sel(time=str(years[0]))

    if convert_coords:
        concat_ds = transform_coords_old_to_new_swiss(concat_ds)

    if output_filename == "":
        output_filename = (
            f"{product}_{pd.to_datetime(concat_ds.time[0].values).year}.nc"
        )

    output_filepath = output_dir_path / output_filename
    if output_filepath.is_file():
        output_filepath.unlink()

    concat_ds.to_netcdf(output_filepath, mode="w", encoding=encoding)


def batch_extraction_prevah(
    base_dir: pathlib.Path,
    netcdf_dir: pathlib.Path,
    product: str,
    prefix_filename_tgz: str = "",
    prefix_filename_gz: str = "",
    convert_coords: bool = True,
    # fill_value: float = np.nan,
    num_workers: int = 2,
):
    """Extracts all PREVAH data in a directory by uncompressing all
    .tgz archives, extracting data from their resulting files into an xarray
    Dataset and saving each year of data in a separate netcdf file.

    Parameters
    ----------
    base_dir : pathlib.Path
        Base directory containing the PREVAH data files. Its subdirectories
        are each year in the PREVAH simulation and under each of those folders
        contains the .tgz archives of the data for this year.
    netcdf_dir : pathlib.Path
        Path to output directory where the netcdf will be saved
    product : str
        PREVAH data product to extract (e.g. RGS)
    prefix_filename_tgz : str, optional
        String that precedes the date in the filename, by default ""
    prefix_filename_gz : str, optional
        String that identifies the type of simulation (e.g. Mob500),
        that precedes the date in the filename, by default ""
    convert_coords : bool, optional
        Whether to conver the coordinates from the old Swiss
        system (LV95) to the new system (LV03), by default True
    num_workers : int, optional
        Number of parallel workers to speed up the extraction
        process, by default 2

    Raises
    ------
    ValueError
        If the product passed as argument is not yet considered
    """
    start_time = time.time()
    if product not in AVAILABLE_PRODUCTS:
        raise ValueError("The product you tried to read is not taken into account yet.")

    output_compression_path = pathlib.Path(tempfile.mkdtemp())
    print(output_compression_path)
    list_years = set(
        [extract_date_hydro_tgz(file.stem).year for file in sorted(base_dir.glob("*"))]
    )

    for year in list_years:
        start_time_year = time.time()
        print(f"{year}.. ", end="")
        output_compression_year_path = output_compression_path / str(year)
        output_compression_year_path.mkdir(exist_ok=True)

        days = list(sorted(base_dir.glob(f"*{prefix_filename_tgz}{year}*")))
        if len(days) == 0:
            print(f"No files for year {year}, passing to next year.")
            pass

        with Pool(num_workers) as p:
            list_ds = [
                ds
                for ds in p.starmap(
                    extract_hydro_tgz,
                    zip(
                        days,
                        repeat(output_compression_year_path),
                        repeat(product),
                        repeat(prefix_filename_gz),
                    ),
                )
            ]
            print(f"| Days: {len(list_ds)} | ", end="")

            output_filename = f"{product}_{year}.nc"
            if prefix_filename_gz != "":
                output_filename = f"{prefix_filename_gz}_" + output_filename
            combine_ds_to_netcdf(list_ds, convert_coords, netcdf_dir, output_filename)

        del list_ds

        elapsed_time = (time.time() - start_time_year) / 60.0
        print(f"done.\t Time taken: {elapsed_time:.2f} minutes.")

    shutil.rmtree(output_compression_path)
    elapsed_time = (time.time() - start_time) / 60.0
    print(f"done.\t Time taken: {elapsed_time:.2f} minutes.")
