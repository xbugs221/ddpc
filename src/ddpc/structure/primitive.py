"""Find primitive cell of crystal structures."""

from ase.atoms import Atoms


def find_primitive(atoms: Atoms, symprec: float = 1e-5, angle_tolerance: float = -1.0) -> Atoms:
    """Find primitive cell of crystal structure.

    Args:
        atoms: Input structure (may be supercell)
        symprec: Symmetry precision in Angstrom
        angle_tolerance : Symmetry search tolerance in the unit of angle deg. If negative, an internally optimized routine is used to judge symmetry.

    Returns
    -------
        Primitive cell as ase.Atoms

    Raises
    ------
        ImportError: If spglib not installed
        ValueError: If structure invalid
    """
    try:
        from spglib import find_primitive as spg_find_prim
    except ImportError as e:
        raise ImportError(
            "'find_primitive' requires 'spglib'.\nInstall with: pip install ddpc-structure"
        ) from e

    if not atoms.cell.array.any():
        raise ValueError("Input structure has no cell information")

    lat = tuple(atoms.cell.array)
    spo = tuple(atoms.get_scaled_positions())
    num = tuple(atoms.numbers)
    cell = (lat, spo, num)
    res = spg_find_prim(cell, symprec, angle_tolerance)

    if res and len(res) == 3:
        lattice, scaled_positions, numbers = res
    else:
        raise ValueError("spglib failed to find primitive cell")

    prim = Atoms(
        numbers=numbers,
        cell=lattice,
        scaled_positions=scaled_positions,
    )

    return prim
