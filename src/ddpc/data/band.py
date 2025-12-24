"""Read band data from output files."""

import sys
from json import load
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from ddpc._utils import absf
from ddpc.data.processors import _refactor_band
from ddpc.data.utils import get_h5_str


def read_band(
    p: Union[str, Path],
    mode: int = 5,
    fmt: str = "8.3f",
) -> Tuple[Dict[str, np.ndarray], float, bool]:
    """Read and process electronic band structure data from HDF5 or JSON files.

    Parameters
    ----------
    p : str or pathlib.Path
        Path to the band structure data file. Supported formats are HDF5 (.h5)
        and JSON (.json) files from DFT calculations.
    mode : int, default 5
        Projection mode for projected band structure data. Only relevant when
        the file contains orbital-projected information.

    Returns
    -------
    tuple of (Dict[str, np.ndarray], float, bool)

        - Dict mapping column names to numpy arrays with k-points and energies
        - Fermi energy in eV
        - Boolean indicating whether the data contains orbital projections

    Raises
    ------
    TypeError
        If the input file is neither HDF5 nor JSON format.
    """
    absfile = str(absf(p))

    if absfile.endswith(".h5"):
        df, efermi, isproj = read_band_h5(absfile, mode)
    elif absfile.endswith(".json"):
        df, efermi, isproj = read_band_json(absfile, mode)
    else:
        raise TypeError(f"{absfile} must be h5 or json file!")

    return df, efermi, isproj


def read_band_h5(absfile: str, mode: int) -> Tuple[Dict[str, np.ndarray], float, bool]:
    """Read band structure data from HDF5 file format.

    Lazy imports h5py to avoid unnecessary dependency loading.
    """
    try:
        import h5py
    except ImportError as err:
        raise ImportError(
            "Reading HDF5 files requires 'h5py'. Install with: pip install ddpc-data"
        ) from err

    with h5py.File(absfile, "r") as band:
        bandinfo = band["BandInfo"]
        if isinstance(bandinfo, h5py.Group):
            efermi_list = bandinfo["EFermi"]
            if isinstance(efermi_list, h5py.Dataset):
                efermi = efermi_list[0]
            else:
                print("ERROR: cannot read /BandInfo/EFermi")
                sys.exit(1)

            proj = bandinfo["IsProject"]
            if isinstance(proj, h5py.Dataset):
                iproj = proj[0]
            else:
                print("ERROR: cannot read /BandInfo/IsProject")
                sys.exit(1)

            if mode == 0:
                df = read_tband(band)
            elif iproj:
                df = read_pband_h5(band, mode)
            else:
                df = read_tband(band)
        else:
            raise TypeError("h5 file must contain 'BandInfo' group!")

    return df, efermi, bool(iproj)


def read_band_json(absfile: str, mode: int) -> Tuple[Dict[str, np.ndarray], float, bool]:
    """Read band structure data from JSON file format."""
    with open(absfile, encoding="utf-8") as fin:
        band = load(fin)
        efermi = band["BandInfo"]["EFermi"]

    iproj = band["BandInfo"]["IsProject"]
    if mode == 0:
        df = read_tband(band, h5=False)
    elif iproj:
        df = read_pband_json(band, mode)
    else:
        df = read_tband(band, h5=False)

    return df, efermi, bool(iproj)


def read_tband(band, h5: bool = True) -> Dict[str, np.ndarray]:
    """Read total (non-projected) band structure data from file."""
    kc: List[float] = band["BandInfo"]["CoordinatesOfKPoints"]
    nok: int = band["BandInfo"]["NumberOfKpoints"]
    if isinstance(nok, int):
        nkpt = nok
    else:
        nkpt = nok[0]

    nob = band["BandInfo"]["NumberOfBand"]
    if isinstance(nob, int):
        nband = nob
    else:
        nband = nob[0]
    kcoord = np.array(kc).reshape(nkpt, 3)
    kx = kcoord[:, 0]
    ky = kcoord[:, 1]
    kz = kcoord[:, 2]
    # distance should be sum of diff
    diff = np.diff(kcoord, axis=0)  # n-1
    dist = [0.0]
    dist.extend(np.cumsum(np.linalg.norm(diff, axis=1)).tolist())

    if h5:
        spin_type_list = get_h5_str(band, "/BandInfo/SpinType")
        collinear = spin_type_list[0] == "collinear"
        sk: List[str] = get_h5_str(band, "/BandInfo/SymmetryKPoints")
        # h5py bands is a nband*nkpt 2d array with C order, have to flatten and reshape it
        bands = (
            np.asarray(band["BandInfo"]["Spin1"]["BandEnergies"])
            .flatten()
            .reshape(nband, nkpt, order="F")
        )
    else:
        collinear = band["BandInfo"]["SpinType"] == "collinear"
        sk = band["BandInfo"]["SymmetryKPoints"]
        bands = np.asarray(band["BandInfo"]["Spin1"]["BandEnergies"]).reshape(
            nband, nkpt, order="F"
        )
    ski = band["BandInfo"]["SymmetryKPointsIndex"]
    sk_column = [""] * nkpt
    for i, symbol in zip(ski, sk):
        sk_column[i - 1] = symbol
    data = {
        "label": np.array(sk_column),
        "kx": kx,
        "ky": ky,
        "kz": kz,
        "dist": np.array(dist),
    }

    # only collinear system has Spin2
    if collinear:
        for i in range(bands.shape[0]):
            data[f"band{i + 1}-up"] = bands[i, :]
        bands = np.asarray(band["BandInfo"]["Spin2"]["BandEnergies"]).reshape(
            nband, nkpt, order="F"
        )
        for i in range(bands.shape[0]):
            data[f"band{i + 1}-down"] = bands[i, :]
    else:
        for i in range(bands.shape[0]):
            data[f"band{i + 1}"] = bands[i, :]

    return data


def read_pband_h5(band, mode: int) -> Dict[str, np.ndarray]:
    """Read orbital-projected band structure data from HDF5 file."""
    kc = band["/BandInfo/CoordinatesOfKPoints"]
    nkpt: int = band["/BandInfo/NumberOfKpoints"][0]
    nband: int = band["/BandInfo/NumberOfBand"][0]
    sk: List[str] = get_h5_str(band, "/BandInfo/SymmetryKPoints")
    ski: List[int] = band["BandInfo/SymmetryKPointsIndex"]

    sk_column = [""] * nkpt
    for i, symbol in zip(ski, sk):
        sk_column[i - 1] = symbol

    kcoord = np.array(kc).reshape(nkpt, 3)
    kx = kcoord[:, 0]
    ky = kcoord[:, 1]
    kz = kcoord[:, 2]
    # distance should be sum of diff
    diff = np.diff(kcoord, axis=0)  # n-1
    dist = [0.0]
    dist.extend(np.cumsum(np.linalg.norm(diff, axis=1)).tolist())

    data = {
        "label": np.array(sk_column),
        "kx": kx,
        "ky": ky,
        "kz": kz,
        "dist": np.array(dist),
    }
    orbitals: List[str] = get_h5_str(band, "/BandInfo/Orbit")
    atom_index = band["/BandInfo/Spin1/ProjectBand/AtomIndex"][0]
    orb_index = band["/BandInfo/Spin1/ProjectBand/OrbitIndexs"][0]

    # only collinear system has Spin2
    spin_type_list = get_h5_str(band, "/BandInfo/SpinType")
    if spin_type_list[0] == "collinear":
        for ai in range(atom_index):
            for oi in range(orb_index):
                data.update(
                    {
                        f"{ai + 1}{orbitals[oi]}-up": np.asarray(
                            band[f"/BandInfo/Spin1/ProjectBand/{ai + 1}/{oi + 1}"]
                        ).flatten()
                    }
                )
                data.update(
                    {
                        f"{ai + 1}{orbitals[oi]}-down": np.asarray(
                            band[f"/BandInfo/Spin2/ProjectBand/{ai + 1}/{oi + 1}"]
                        ).flatten()
                    }
                )
    else:
        for ai in range(atom_index):
            for oi in range(orb_index):
                data.update(
                    {
                        f"{ai + 1}{orbitals[oi]}": np.asarray(
                            band[f"/BandInfo/Spin1/ProjectBand/1/{ai + 1}/{oi + 1}"]
                        ).flatten()
                    }
                )

    elements: List[str] = get_h5_str(band, "/AtomInfo/Elements")
    _data = _refactor_band(data, nkpt, nband, elements, mode)

    return _data


def read_pband_json(band: Dict, mode: int) -> Dict[str, np.ndarray]:
    """Read orbital-projected band structure data from JSON file."""
    kc: List[float] = band["BandInfo"]["CoordinatesOfKPoints"]
    nkpt: int = band["BandInfo"]["NumberOfKpoints"]
    nband: int = band["BandInfo"]["NumberOfBand"]

    sk: List[str] = band["BandInfo"]["SymmetryKPoints"]
    ski: List[int] = band["BandInfo"]["SymmetryKPointsIndex"]
    sk_column = [""] * nkpt
    for i, symbol in zip(ski, sk):
        sk_column[i - 1] = symbol

    kcoord = np.array(kc).reshape(nkpt, 3)
    kx = kcoord[:, 0]
    ky = kcoord[:, 1]
    kz = kcoord[:, 2]
    # distance should be sum of diff
    diff = np.diff(kcoord, axis=0)  # n-1
    dist = [0.0]
    dist.extend(np.cumsum(np.linalg.norm(diff, axis=1)).tolist())

    data = {
        "label": np.array(sk_column),
        "kx": kx,
        "ky": ky,
        "kz": kz,
        "dist": np.array(dist),
    }
    orbitals: List[str] = band["BandInfo"]["Orbit"]
    if band["BandInfo"]["SpinType"] == "collinear":
        project1 = band["BandInfo"]["Spin1"]["ProjectBand"]
        project2 = band["BandInfo"]["Spin2"]["ProjectBand"]
        for p in project1:
            atom_index = p["AtomIndex"]
            orb_index = p["OrbitIndex"] - 1
            contrib = p["Contribution"]
            data.update({f"{atom_index}{orbitals[orb_index]}-up": contrib})
        for p in project2:
            atom_index = p["AtomIndex"]
            orb_index = p["OrbitIndex"] - 1
            contrib = p["Contribution"]
            data.update({f"{atom_index}{orbitals[orb_index]}-down": contrib})
    else:
        project = band["BandInfo"]["Spin1"]["ProjectBand"]
        for p in project:
            atom_index = p["AtomIndex"]
            orb_index = p["OrbitIndex"] - 1
            contrib = p["Contribution"]
            data.update({f"{atom_index}{orbitals[orb_index]}": contrib})

    elements: List[str] = [atom["Element"] for atom in band["AtomInfo"]["Atoms"]]
    _data = _refactor_band(data, nkpt, nband, elements, mode)

    return _data
