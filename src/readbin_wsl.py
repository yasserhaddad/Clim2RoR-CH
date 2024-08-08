"""Module containing functions to read WSL binary data."""

import struct
import gzip
import pathlib
from collections import namedtuple
import numpy as np

wslBinResult = namedtuple("wslBinResult", "data x y dist")


def read_data_gz(file_path: pathlib.Path) -> wslBinResult:
    '''Read gz binary PREVAH data from a file'''
    with gzip.open(file_path, "rb") as file:
        size=4
        count=6
        buffer = file.read(size*count)
        col, row, xu, yu, dist, nodata = struct.unpack(f'<{count}f', buffer)
        # print(col, row, xu, yu, dist, nodata)
        count = int(col*row)
        # print(count)
        # buffer = file.read(size*count+24*size)
        count2 = size*count+24
        # print(count2)
        buffer = file.read(count2)
        # print(len(buffer))
        values = struct.unpack(f'<{count+6}f', buffer)
        arr = np.asarray(values)
        arr = arr[6:]
        # print(arr.size)
        if np.any(arr == nodata):
            arr[arr == nodata] = np.nan
        dem = np.reshape(arr, (int(row),int(col)))
        # dem=dem[1:,:]
        dem = np.flip(dem, axis=0)
        x = np.arange(int(xu),(int(xu) +(int(dist) * (int(col) - 1)))+1,int(dist))
        y = np.arange(int(yu),(int(yu) + (int(dist) * (int(row) - 1)))+1,int(dist))

    return wslBinResult(dem[::-1, :], x, y[::-1], dist)