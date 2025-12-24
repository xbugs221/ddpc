"""Read density of states data from output files."""

import sys
from json import load
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from ddpc._utils import absf
from ddpc.data.processors import _refactor_dos
from ddpc.data.utils import get_h5_str


def read_dos(
    p: Union[str, Path],
    mode: int = 5,
) -> Tuple[Dict[str, np.ndarray], float, bool]:
    """Read and process electronic density of states data from HDF5 or JSON files.

    Parameters
    ----------
    p : str or pathlib.Path
        Path to the DOS data file. Supported formats are HDF5 (.h5) and
        JSON (.json) files from DFT calculations.
    mode : int, default 5
        Projection mode for projected density of states data.

    Returns
    -------
    tuple of (Dict[str, np.ndarray], float, bool)

        - Dict mapping column names to numpy arrays with energy and DOS values
        - Fermi energy in eV
        - Boolean indicating whether the data contains orbital projections

    Raises
    ------
    TypeError
        If the input file is neither HDF5 nor JSON format.
    """
    absfile = str(absf(p))

    if absfile.endswith(".h5"):
        df, efermi, isproj = read_dos_h5(absfile, mode)
    elif absfile.endswith(".json"):
        df, efermi, isproj = read_dos_json(absfile, mode)
    else:
        raise TypeError(f"{absfile} must be h5 or json file!")

    return df, efermi, isproj


def read_dos_h5(absfile: str, mode: int) -> Tuple[Dict[str, np.ndarray], float, bool]:
    """Read density of states data from HDF5 file format.

    Lazy imports h5py to avoid unnecessary dependency loading.
    """
    try:
        import h5py
    except ImportError as err:
        raise ImportError(
            "Reading HDF5 files requires 'h5py'. Install with: pip install ddpc-data"
        ) from err

    with h5py.File(absfile, "r") as dos:
        dosinfo = dos["DosInfo"]
        if isinstance(dosinfo, h5py.Group):
            efermi_list = dosinfo["EFermi"]
            if isinstance(efermi_list, h5py.Dataset):
                efermi = efermi_list[0]
            else:
                print("ERROR: cannot read /DosInfo/EFermi")
                sys.exit(1)

            proj = dosinfo["Project"]
            if isinstance(proj, h5py.Dataset):
                iproj = proj[0]
            else:
                print("ERROR: cannot read /DosInfo/Project")
                sys.exit(1)
            if mode == 0:
                df = read_tdos(dos)
            elif iproj:
                df = read_pdos_h5(dos, mode)
            else:
                df = read_tdos(dos)
        else:
            raise TypeError("h5 file must contain 'DosInfo' group!")

    return df, efermi, bool(iproj)


def read_dos_json(absfile: str, mode: int) -> Tuple[Dict[str, np.ndarray], float, bool]:
    """Read density of states data from JSON file format."""
    with open(absfile, encoding="utf-8") as fin:
        dos = load(fin)
        efermi = dos["DosInfo"]["EFermi"]
    iproj = dos["DosInfo"]["Project"]
    if mode == 0:
        df = read_tdos(dos, h5=False)
    elif iproj:
        df = read_pdos_json(dos, mode)
    else:
        df = read_tdos(dos, h5=False)

    return df, efermi, bool(iproj)


def read_tdos(dos, h5: bool = True) -> Dict[str, np.ndarray]:
    """Read total (non-projected) density of states data."""
    energies = np.asarray(dos["DosInfo"]["DosEnergy"])

    if h5:
        spin_type = dos["DosInfo"]["SpinType"][0]
    else:
        spin_type = dos["DosInfo"]["SpinType"]

    if spin_type == "collinear":
        densities = {
            "energy": energies,
            "up": np.asarray(dos["DosInfo"]["Spin1"]["Dos"]),
            "down": np.asarray(dos["DosInfo"]["Spin2"]["Dos"]),
        }
    else:
        densities = {
            "energy": energies,
            "dos": np.asarray(dos["DosInfo"]["Spin1"]["Dos"]),
        }
    return densities


def read_pdos_h5(dos, mode: int) -> Dict[str, np.ndarray]:
    """Read orbital-projected density of states data from HDF5 file."""
    energies: List[float] = dos["/DosInfo/DosEnergy"]
    data = {}
    orbitals: List[str] = get_h5_str(dos, "/DosInfo/Orbit")

    atom_index: int = dos["/DosInfo/Spin1/ProjectDos/AtomIndexs"][0]  # 2
    orb_index: int = dos["/DosInfo/Spin1/ProjectDos/OrbitIndexs"][0]  # 9
    spin_type_list = get_h5_str(dos, "/DosInfo/SpinType")
    if spin_type_list[0] == "collinear":
        data.update(
            {
                "tdos-up": np.asarray(dos["/DosInfo/Spin1/Dos"]),
                "tdos-down": np.asarray(dos["/DosInfo/Spin2/Dos"]),
            },
        )
        for ai in range(atom_index):
            for oi in range(orb_index):
                data.update(
                    {
                        f"{ai + 1}{orbitals[oi]}-up": dos[
                            f"/DosInfo/Spin1/ProjectDos{ai + 1}/{oi + 1}"
                        ]
                    }
                )
                data.update(
                    {
                        f"{ai + 1}{orbitals[oi]}-down": dos[
                            f"/DosInfo/Spin2/ProjectDos{ai + 1}/{oi + 1}"
                        ]
                    }
                )
    else:
        data.update(
            {"tdos": np.asarray(dos["/DosInfo/Spin1/Dos"])},
        )
        for ai in range(atom_index):
            for oi in range(orb_index):
                data.update(
                    {f"{ai + 1}{orbitals[oi]}": dos[f"/DosInfo/Spin1/ProjectDos{ai + 1}/{oi + 1}"]}
                )

    if mode == 3:
        elements: List[str] = get_h5_str(dos, "/AtomInfo/Elements")
    else:
        elements = []
    _data = _refactor_dos(energies, data, mode, elements)

    return _data


def read_pdos_json(dos: Dict, mode: int) -> Dict[str, np.ndarray]:
    """Read orbital-projected density of states data from JSON file."""
    energies: List[float] = dos["DosInfo"]["DosEnergy"]
    data = {}
    orbitals: List[str] = dos["DosInfo"]["Orbit"]

    if dos["DosInfo"]["SpinType"] == "collinear":
        data.update(
            {
                "tdos-up": dos["DosInfo"]["Spin1"]["Dos"],
                "tdos-down": dos["DosInfo"]["Spin2"]["Dos"],
            },
        )
        project = dos["DosInfo"]["Spin1"]["ProjectDos"]
        for p in project:
            atom_index = p["AtomIndex"]
            orb_index = p["OrbitIndex"] - 1
            contrib = p["Contribution"]
            data.update({f"{atom_index}{orbitals[orb_index]}-up": contrib})
        project = dos["DosInfo"]["Spin2"]["ProjectDos"]
        for p in project:
            atom_index = p["AtomIndex"]
            orb_index = p["OrbitIndex"] - 1
            contrib = p["Contribution"]
            data.update({f"{atom_index}{orbitals[orb_index]}-down": contrib})

    else:
        data.update(
            {"tdos": dos["DosInfo"]["Spin1"]["Dos"]},
        )
        project = dos["DosInfo"]["Spin1"]["ProjectDos"]
        for p in project:
            atom_index = p["AtomIndex"]
            orb_index = p["OrbitIndex"] - 1
            contrib = p["Contribution"]
            data.update({f"{atom_index}{orbitals[orb_index]}": contrib})

    if mode == 3:
        elements: List[str] = [atom["Element"] for atom in dos["AtomInfo"]["Atoms"]]
    else:
        elements = []
    _data = _refactor_dos(energies, data, mode, elements)

    return _data
