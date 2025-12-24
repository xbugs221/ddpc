"""Write RESCU XYZ format files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ddpc._utils import absf

if TYPE_CHECKING:
    from ase.atoms import Atoms


def write(f: str, atoms: Atoms) -> str:
    """Write ASE Atoms to RESCU XYZ format.

    Preserves magnetic moments and position constraints.
    """
    ret = f"{len(atoms)}\nAuto-generated xyz file\n"
    mags = atoms.get_initial_magnetic_moments()
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    fix_info = atoms.info.get("atom_fix", None)
    ret = _add_atom_lines(symbols, positions, fix_info, mags, ret)

    if f == "-":
        pass
    else:
        absxyz = absf(f)
        absxyz.parent.mkdir(parents=True, exist_ok=True)

        with open(absxyz, "w", encoding="utf-8") as _f:
            _f.write(ret)

    return ret


def _add_atom_lines(symbols, positions, fix_info, mags, ret) -> str:
    """Add atomic information lines."""
    if mags.any():
        ret = _add_with_mag(symbols, positions, fix_info, mags, ret)
    elif fix_info is not None and fix_info.any():
        for symbol, pos, moveable_xyz in zip(symbols, positions, fix_info):
            moveable = f"{moveable_xyz[0]} {moveable_xyz[1]} {moveable_xyz[2]}"
            ret += f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} 0 0 0 {moveable}\n"
    else:
        for symbol, pos in zip(symbols, positions):
            ret += f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"

    return ret


def _add_with_mag(symbols, positions, fix_info, mags, ret) -> str:
    """Add atomic lines with magnetic moments."""
    if mags.shape == (len(symbols), 3):
        if fix_info is not None and fix_info.any():
            for symbol, pos, mag, moveable_xyz in zip(
                symbols, positions, mags, fix_info, strict=True
            ):
                moveable = f"{moveable_xyz[0]} {moveable_xyz[1]} {moveable_xyz[2]}"
                _mag = f"{mag[0]:.2f} {mag[1]:.2f} {mag[2]:.2f}"
                ret += f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {_mag} {moveable}\n"
        else:
            for symbol, pos, mag in zip(symbols, positions, mags):
                _mag = f"{mag[0]:.2f} {mag[1]:.2f} {mag[2]:.2f}"
                ret += f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {_mag}\n"
    elif mags.shape == (len(symbols),):
        if fix_info is not None and fix_info.any():
            for symbol, pos, mag, moveable_xyz in zip(
                symbols, positions, mags, fix_info, strict=True
            ):
                moveable = f"{moveable_xyz[0]} {moveable_xyz[1]} {moveable_xyz[2]}"
                ret += f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {mag:.2f} {moveable}\n"
        else:
            for symbol, pos, mag in zip(symbols, positions, mags):
                ret += f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {mag:.2f}\n"
    else:
        raise ValueError(f"Invalid magnetic moment shape: {mags.shape}")

    return ret
