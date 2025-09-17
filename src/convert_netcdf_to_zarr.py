from pathlib import Path

import dask
import numpy as np
import rioxarray
import xarray as xr
from dask.diagnostics import ProgressBar
from pyproj import Transformer
from src.var_attributes import ACCUM_HYDRO_ZARR_ENCODING
from src.extract_runoff_prevah import transform_coords_old_to_new_swiss

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

data_dir = Path("/landclim/yhaddad/paper1/data/prevah/netcdf")
nc_files = sorted(str(p) for p in data_dir.glob("*.nc"))
if not nc_files:
    raise SystemExit(f"No netCDF files found in {data_dir}")

ds = xr.open_mfdataset(nc_files, combine="by_coords", parallel=True)

# Build xarray encoding dict from each variable's "zarr" attribute (if present).
chunks = {"time": 365, "y": 256, "x": 256}
encoding = {
    var: {
        **ACCUM_HYDRO_ZARR_ENCODING,
        "chunks": tuple(chunks.get(dim, -1) for dim in ds[var].dims),
    }
    for var in ds.data_vars
}

# # Define source and target CRS (old -> new Swiss coordinate system)
# src_crs = "EPSG:21781"  # LV03 (old)
# dst_crs = "EPSG:2056"  # LV95 (new)

ds = transform_coords_old_to_new_swiss(ds)

with ProgressBar(dt=10):
    # if ds.rio.crs is None:
    #     ds = ds.rio.write_crs(src_crs)

    # # make sure the data are chunked BEFORE reprojection so reprojection can be
    # # scheduled as many lazy tasks rather than computed eagerly
    ds = ds.chunk(chunks)

    # # ask Rasterio to use multiple threads for the reproject kernel (gets passed
    # # to rasterio.warp.reproject). Adjust num_threads to match your machine.
    # ds_reproj = ds.rio.reproject(dst_crs, num_threads=2)

    out = Path("/landclim/yhaddad/hydropower/hydropower_projections/ror/prevah_obs/rgs.zarr")
    # this will execute the lazy reproject + write in parallel using dask
    ds.to_zarr(out, mode="w", encoding=encoding)
ds.close()
