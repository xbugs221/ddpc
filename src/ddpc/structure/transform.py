"""Coordinate transformations for crystal structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.atoms import Atoms


def scale_positions(atoms: Atoms) -> Atoms:
    """Convert atomic positions to fractional coordinates.

    Args:
        atoms: Input structure

    Returns
    -------
        New structure with scaled coordinates
    """
    scaled = atoms.copy()
    scaled.set_scaled_positions(atoms.get_scaled_positions())
    return scaled
