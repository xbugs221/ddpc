"""Write DS-PAW .as format files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ddpc._utils import absf

if TYPE_CHECKING:
    from ase.atoms import Atoms


def write(p: str, atoms: Atoms) -> str:
    """Write ASE Atoms to DS-PAW .as format.

    Preserves lattice constraints, atomic constraints, and magnetic moments.
    """
    lines = "Total number of atoms\n"
    lines += "%d\n" % len(atoms)

    freedom = atoms.info
    lines = _add_lat_lines(freedom, lines, atoms)
    lines = _add_atom_lines(freedom, lines, atoms)
    lines = _write_to_file(p, lines)

    return lines


def _add_lat_lines(freedom: dict, lines: str, atoms: Atoms) -> str:
    """Add lattice vector lines."""
    if "lat" in freedom:
        lat_fix = freedom.pop("lat")
        lines += "Lattice Fix_x Fix_y Fix_z\n"
        formatted_fts = []
        for ft in lat_fix:
            ft_formatted = "T" if ft else "F"
            formatted_fts.append(ft_formatted)
        fix_str1 = " ".join(formatted_fts[:3])
        fix_str2 = " ".join(formatted_fts[3:6])
        fix_str3 = " ".join(formatted_fts[6:9])
        fix_strs = [fix_str1, fix_str2, fix_str3]
        for v, fs in zip(atoms.cell.array, fix_strs):
            lines += f"{v[0]: 10.4f} {v[1]: 10.4f} {v[2]: 10.4f} {fs}\n"

    else:
        lines += "Lattice\n"
        for v in atoms.cell.array:
            lines += f"{v[0]: 10.4f} {v[1]: 10.4f} {v[2]: 10.4f}\n"

    return lines


def _add_atom_lines(freedom, lines, atoms):
    """Add atomic information lines."""
    key_str = " ".join(freedom.keys())
    magmoms = atoms.get_initial_magnetic_moments()
    init_mag = True
    if len(magmoms.shape) == 1:
        if not any(magmoms):
            init_mag = False
        else:
            key_str += " Mag"
    else:
        key_str += " Mag_x Mag_y Mag_z"
    lines += f"Cartesian {key_str}\n"
    elements = atoms.symbols
    positions = atoms.positions
    atom_fix = []
    for i in range(len(atoms.symbols)):
        raw = ""
        for val_column in freedom.values():
            if val_column[i]:
                raw += "T "
            else:
                raw += "F "
        atom_fix.append(raw.strip())

    for ele, pos, af, magmom in zip(elements, positions, atom_fix, magmoms):
        if isinstance(magmom, np.ndarray):
            init_magmom = np.array2string(
                magmom,
                formatter={"float_kind": lambda x: f"{x:7.3f}"},
            ).strip("[]")
        elif magmom:
            init_magmom = f"{float(magmom): 7.3f}"
        else:
            init_magmom = "  0.000" if init_mag else ""
        lines += f"{ele:<2} {pos[0]: 10.4f} {pos[1]: 10.4f} {pos[2]: 10.4f} {af} {init_magmom}\n"

    return lines


def _write_to_file(filename, lines) -> str:
    """Write content to file or return as string."""
    if not filename:
        return lines

    if filename == "-":
        pass
    else:
        absfile = absf(filename)
        absfile.parent.mkdir(parents=True, exist_ok=True)

        with open(absfile, "w", encoding="utf-8") as file:
            file.write(lines)

    return lines
