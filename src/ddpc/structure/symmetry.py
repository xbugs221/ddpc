"""Symmetry analysis for crystal structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.atoms import Atoms


def get_symmetry(
    atoms: Atoms,
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
    hall_number: int = 0,
    symbol_type: int = 0,
) -> dict:
    """Get symmetry information using spglib.

    Args:
        atoms: Input structure
        symprec: Symmetry precision
        hall_number : If a serial number of Hall symbol (>0) is given, the database corresponding to the Hall symbol is made.
            The mapping from Hall symbols to a space-group-type is the many-to-one mapping.
            Without specifying this option (i.e., in the case of ``hall_number=0``), always the first one
            (the smallest serial number corresponding to the space-group-type in [list of space groups (Seto's web site)](https://yseto.net/en/sg/sg1))
            among possible choices and settings is chosen as default.
            This argument is useful when the other choice (or setting) is expected to be hooked. This affects to the obtained values of `international`, `hall`, `choice`, `transformation_matrix`,
            `origin shift`, `wyckoffs`, `std_lattice`, `std_positions`, `std_types` and `std_rotation_matrix`, but not to `rotations`
            and `translations` since the later set is defined with respect to the basis vectors of user's input (the `cell` argument).
        symbol_type: With ``symbol_type=1``, Schoenflies symbol is given instead of international symbol.

    Returns
    -------
        Dict with symmetry info:
        - spacegroup: International symbol (str)
        - spacegroup_number: Space group number (int)
        - point_group: Point group symbol (str)
        - crystal_system: Crystal system (str)
        - hall_number: Hall symbol number (int)

    Raises
    ------
        ImportError: If spglib not installed
        ValueError: If structure invalid
    """
    try:
        from spglib import get_spacegroup, get_symmetry_dataset
    except ImportError as e:
        raise ImportError(
            "'get_symmetry' requires 'spglib'.\nInstall with: pip install ddpc-structure"
        ) from e

    if not atoms.cell.array.any():
        raise ValueError("Input structure has no cell information")

    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    dataset = get_symmetry_dataset(cell, symprec, angle_tolerance, hall_number)

    if dataset is None:
        raise ValueError("spglib failed to determine symmetry")

    spacegroup_str = get_spacegroup(cell, symprec, angle_tolerance, symbol_type)

    return {
        "spacegroup": spacegroup_str,
        "spacegroup_number": dataset.number,
        "point_group": dataset.pointgroup,
        "crystal_system": _get_crystal_system(dataset.number),
        "hall_number": dataset.hall_number,
    }


def _get_crystal_system(spacegroup_number: int) -> str:
    """Map space group number to crystal system."""
    if 1 <= spacegroup_number <= 2:
        return "triclinic"
    if 3 <= spacegroup_number <= 15:
        return "monoclinic"
    if 16 <= spacegroup_number <= 74:
        return "orthorhombic"
    if 75 <= spacegroup_number <= 142:
        return "tetragonal"
    if 143 <= spacegroup_number <= 167:
        return "trigonal"
    if 168 <= spacegroup_number <= 194:
        return "hexagonal"
    if 195 <= spacegroup_number <= 230:
        return "cubic"
    return "unknown"
