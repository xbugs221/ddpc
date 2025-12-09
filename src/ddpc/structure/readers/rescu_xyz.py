"""Read RESCU XYZ format files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ddpc._utils import absf, remove_comments

if TYPE_CHECKING:
    from ase.atoms import Atoms


def read(p: str | Path) -> Atoms:
    """Read RESCU XYZ format and convert to ASE Atoms.

    Supports magnetic moments and position constraints.
    """
    from ase.atoms import Atoms

    absfile = absf(p)
    lines = remove_comments(absfile, "#")

    (
        nele,
        elements,
        pos,
        mags,
        moveable_indices_x,
        moveable_indices_y,
        moveable_indices_z,
    ) = _read_prop(lines)

    if not (nele == len(pos) == len(elements)):
        raise ValueError(f"Inconsistent atom count: {nele=}, {len(pos)=}, {len(elements)=}")
    if mags and nele != len(mags):
        raise ValueError(f"Inconsistent mag count: {nele=}, {len(mags)=}")
    if moveable_indices_x and nele != len(moveable_indices_x):
        raise ValueError(f"Inconsistent fix count: {nele=}, {len(moveable_indices_x)=}")

    fix_info = {
        "atom_fix": np.array([moveable_indices_x, moveable_indices_y, moveable_indices_z]).T
    }
    return Atoms(symbols=elements, positions=pos, magmoms=mags, pbc=True, info=fix_info)


def _read_prop(lines: list[str]):
    """Parse atomic properties from RESCU XYZ lines."""
    nele = 0
    elements = []
    pos = []
    mags: list[float | list[float]] = []
    moveable_indices_x = []
    moveable_indices_y = []
    moveable_indices_z = []

    for i, _line in enumerate(lines):
        line = _line.strip()
        if "#" in line:
            line = line[: line.index("#")]
        if "%" in line:
            line = line[: line.index("%")]

        if i == 0:
            nele = int(line)
        elif i > 1 and line:
            split_items = line.split()
            if len(split_items) == 4:
                ele, x, y, z = split_items
                elements.append(ele)
                pos.append([float(x), float(y), float(z)])
            elif len(split_items) == 5:
                ele, x, y, z, m = split_items
                elements.append(ele)
                pos.append([float(x), float(y), float(z)])
                mags.append(float(m))
            elif len(split_items) == 7:
                ele, x, y, z, mag_x, mag_y, mag_z = split_items
                elements.append(ele)
                pos.append([float(x), float(y), float(z)])
                mags.append([float(mag_x), float(mag_y), float(mag_z)])
            elif len(split_items) == 8:
                ele, x, y, z, mag, moveable_x, moveable_y, moveable_z = split_items
                elements.append(ele)
                pos.append([float(x), float(y), float(z)])
                mags.append(float(mag))
                moveable_indices_x.append(int(moveable_x))
                moveable_indices_y.append(int(moveable_y))
                moveable_indices_z.append(int(moveable_z))
            elif len(split_items) == 10:
                (
                    ele,
                    x,
                    y,
                    z,
                    mag_x,
                    mag_y,
                    mag_z,
                    moveable_x,
                    moveable_y,
                    moveable_z,
                ) = split_items
                elements.append(ele)
                pos.append([float(x), float(y), float(z)])
                mags.append([float(mag_x), float(mag_y), float(mag_z)])
                moveable_indices_x.append(int(moveable_x))
                moveable_indices_y.append(int(moveable_y))
                moveable_indices_z.append(int(moveable_z))
            else:
                raise ValueError(f"Invalid line: {line}")

    return (
        nele,
        elements,
        pos,
        mags,
        moveable_indices_x,
        moveable_indices_y,
        moveable_indices_z,
    )
