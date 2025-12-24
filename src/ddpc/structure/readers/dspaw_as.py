"""Read DS-PAW .as format files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import numpy as np

from ddpc._utils import absf, remove_comments

if TYPE_CHECKING:
    from ase.atoms import Atoms


def read(p: Union[Path, str] = "structure.as") -> Atoms:
    """Read DS-PAW .as format and convert to ASE Atoms.

    Supports lattice constraints, atomic constraints, and magnetic moments.
    """
    from ase.atoms import Atoms

    absfile = absf(p)
    lines = remove_comments(absfile, "#")

    natom = int(lines[1])
    lattice = _get_lat(lines)
    lat_fixs = _get_latfixs(lines)
    elements, coords = _get_ele_pos(lines, natom)
    atom_fix, mag = _get_mag_fix(lines, natom)

    if "Mag" in mag:
        magmoms = mag.pop("Mag")
    elif "Mag_x" in mag and "Mag_y" in mag and "Mag_z" in mag:
        magmoms = np.array([mag.pop("Mag_x"), mag.pop("Mag_y"), mag.pop("Mag_z")]).T.tolist()
    else:
        magmoms = None

    freedom = {"lat": lat_fixs} | atom_fix if any(lat_fixs) else atom_fix
    atoms = Atoms(symbols=elements, cell=lattice, info=freedom, magmoms=magmoms, pbc=True)
    cd = lines[6].strip().split()[0]
    _set_poses(atoms, cd, coords)

    return atoms


def _set_poses(atoms: Atoms, cd: str, coords: np.ndarray) -> None:
    """Set atomic positions based on coordinate type."""
    if cd == "Direct":
        atoms.set_scaled_positions(coords)
    elif cd == "Cartesian":
        atoms.set_positions(coords)
    else:
        raise ValueError("Structure file format error!")


def _get_latfixs(lines: List[str]) -> List[bool]:
    """Extract lattice constraint information."""
    lat_fixs: List[bool] = []
    if lines[2].strip() != "Lattice":
        lattice_fix_info = lines[2].strip().split()[1:]
        if lattice_fix_info == ["Fix_x", "Fix_y", "Fix_z"]:
            for line in lines[3:6]:
                lfs = line.strip().split()[3:6]
                for lf in lfs:
                    if lf.startswith("T"):
                        lat_fixs.append(True)
                    elif lf.startswith("F"):
                        lat_fixs.append(False)
        elif lattice_fix_info == ["Fix"]:
            for line in lines[3:6]:
                lf = line.strip().split()[3]
                if lf.startswith("T"):
                    lat_fixs.extend([True, True, True])
                elif lf.startswith("F"):
                    lat_fixs.extend([False, False, False])
        else:
            raise ValueError("Lattice fix info error!")

    return lat_fixs


def _get_lat(lines: List[str]) -> np.ndarray:
    """Parse lattice vectors."""
    lattice = []
    for line in lines[3:6]:
        vector = line.split()
        lattice.extend([float(vector[0]), float(vector[1]), float(vector[2])])
    return np.asarray(lattice).reshape(3, 3)


def _get_ele_pos(lines: List[str], natom: int) -> Tuple[List[str], np.ndarray]:
    """Extract element symbols and positions."""
    elements = []
    positions = []
    for i in range(natom):
        atom_data = lines[i + 7].strip().split()
        elements.append(atom_data[0])
        positions.extend([float(atom_data[1]), float(atom_data[2]), float(atom_data[3])])
    coords = np.asarray(positions).reshape(-1, 3)
    elements = [re.sub(r"_", "", e) for e in elements]

    return elements, coords


def _get_mag_fix(lines: List[str], natom: int) -> Tuple[Dict, Dict]:
    """Extract magnetic moments and atomic constraints."""
    l6 = lines[6].strip()
    mf_info = l6.split()[1:]
    for item in mf_info:
        assert item in [
            "Mag",
            "Mag_x",
            "Mag_y",
            "Mag_z",
            "Fix",
            "Fix_x",
            "Fix_y",
            "Fix_z",
        ]

    def handle_fix_value(val_str):
        return val_str.startswith("T")

    mag_fix_dict: Dict[str, List] = {}
    for mf_index, item in enumerate(mf_info):
        values: List[Any] = []
        for i in range(natom):
            atom_data = lines[i + 7].strip().split()
            mf_data = atom_data[4:]

            if item == "Fix":
                values.append(
                    [
                        handle_fix_value(mf_data[mf_index]),
                        handle_fix_value(mf_data[mf_index + 1]),
                        handle_fix_value(mf_data[mf_index + 2]),
                    ]
                )
            elif item.startswith("Fix_"):
                values.append(handle_fix_value(mf_data[mf_index]))
            else:
                values.append(float(mf_data[mf_index]))

        mag_fix_dict[item] = values

    atom_fix = {k: v for k, v in mag_fix_dict.items() if k.startswith("Fix")}
    mag = {k: v for k, v in mag_fix_dict.items() if k.startswith("Mag")}

    return atom_fix, mag
