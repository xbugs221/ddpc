"""Test orthogonal supercell finding."""

import pytest
from ase import Atoms

from ddpc.structure import find_orthogonal


def test_find_orthogonal_basic():
    """Test basic orthogonal supercell finding."""
    # Create a non-orthogonal structure
    cell = [[3.0, 0.0, 0.0], [1.5, 2.6, 0.0], [0.0, 0.0, 4.0]]
    atoms = Atoms("C2", positions=[[0, 0, 0], [1, 1, 2]], cell=cell, pbc=True)

    ortho = find_orthogonal(atoms, max_length=15.0)

    # Check that result is orthogonal
    angles = ortho.cell.angles()
    assert all(abs(angle - 90.0) < 1.0 for angle in angles)


def test_find_orthogonal_no_cell():
    """Test error when structure has no cell."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])

    with pytest.raises(ValueError, match="no cell information"):
        find_orthogonal(atoms)


def test_find_orthogonal_with_kwargs():
    """Test orthogonal supercell with various kwargs."""
    cell = [[3.0, 0.0, 0.0], [1.5, 2.6, 0.0], [0.0, 0.0, 4.0]]
    atoms = Atoms("C2", positions=[[0, 0, 0], [1, 1, 2]], cell=cell, pbc=True)

    ortho = find_orthogonal(
        atoms,
        min_length=10.0,
        max_length=20.0,
        angle_tolerance=0.01,
        step_size=0.2,
    )

    angles = ortho.cell.angles()
    assert all(abs(angle - 90.0) < 1.0 for angle in angles)
