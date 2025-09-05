import zarr

ATTRIBUTES = {
    "RGS": {
        "product": "RGS",
        "standard_name": "total_discharge",
        "long_name": "Total discharge",
        "units": "mm/day"
    }
}

HYDRO_NETCDF_ENCODINGS = {
    'zlib': True,
    'shuffle': True,
    'complevel': 1,
    'fletcher32': False,
    'contiguous': False,
    'chunksizes': (30, 153, 143),
}


ACCUM_HYDRO_NETCDF_ENCODINGS = {
    'zlib': True,
    'shuffle': True,
    'complevel': 1,
    'fletcher32': False,
    'contiguous': False,
    'chunksizes': (30, 1000),
}

ACCUM_HYDRO_ZARR_ENCODING = {
    "compressor": zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
}

HYDROPOWER_NETCDF_ENCODINGS = {
    'zlib': True,
    'shuffle': True,
    'complevel': 1,
    'fletcher32': False,
    'contiguous': False,
    'chunksizes': (365, 100),
}