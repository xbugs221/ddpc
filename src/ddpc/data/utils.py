"""Utility functions for data processing."""

import os
import sys
from typing import Tuple, Union, cast

import numpy as np


def get_h5_str(f: str, key: str) -> list:
    """Read string data from HDF5 file and return as list of elements.

    Lazy imports h5py to avoid unnecessary dependency loading.
    """
    try:
        import h5py
    except ImportError as err:
        raise ImportError(
            "Reading HDF5 files requires 'h5py'. Install with: pip install ddpc-data"
        ) from err

    data: h5py.File
    if isinstance(f, h5py.File):
        data = f
    elif isinstance(f, str):
        absh5 = os.path.abspath(f)
        data = h5py.File(absh5)
    else:
        raise TypeError(f)

    _bytes = np.asarray(data.get(key))
    tempdata = np.asarray([i.decode() for i in _bytes])
    tempdata_str: str = cast(str, "".join(tempdata))

    return tempdata_str.split(";")


def _split_atomindex_orbital(s: str) -> Tuple[int, str]:
    """Split a string into atom index and orbital designation."""
    first_letter_index = -1
    for i, char in enumerate(s):
        if not char.isdigit():
            first_letter_index = i
            break

    if first_letter_index == -1:
        return int(s), ""
    atom_index_str = s[:first_letter_index]
    orbital_str = s[first_letter_index:]
    return int(atom_index_str), orbital_str


def _get_ao_spin(k: str) -> Tuple[str, str]:
    """Parse atomic orbital and spin information from formatted key strings."""
    ls = k.split("-")
    if len(ls) == 1:  # nospin
        return ls[0], ""
    if len(ls) == 2:  # spin-polarized
        return ls[0], ls[1]
    print(f"get_ao_spin error: {k=}")
    sys.exit(1)


def _inplace_update_data(_data: dict, key: str, v: Union[np.ndarray, list]) -> None:
    """Update data dictionary by adding values to existing keys or creating new ones."""
    if key in _data:
        _data[key] += np.asarray(v)
    else:
        _data[key] = np.asarray(v)
