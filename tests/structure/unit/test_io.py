"""Test structure I/O functions."""

import pytest
from ase.atoms import Atoms

from ddpc.structure import read_structure, write_structure


def test_read_as_format(structures_dir):
    """Test reading DS-PAW .as format."""
    as_file = structures_dir / "mag.as"
    if not as_file.exists():
        pytest.skip("Test structure not found")

    atoms = read_structure(as_file)
    assert isinstance(atoms, Atoms)
    assert len(atoms) > 0


def test_read_xyz_format(structures_dir):
    """Test reading XYZ format."""
    xyz_file = structures_dir / "Si.xyz"
    if not xyz_file.exists():
        pytest.skip("Test structure not found")

    atoms = read_structure(xyz_file)
    assert isinstance(atoms, Atoms)
    assert len(atoms) > 0


def test_write_read_roundtrip(structures_dir, temp_output_dir):
    """Test that writing and reading preserves structure."""
    input_file = structures_dir / "mag.as"
    if not input_file.exists():
        pytest.skip("Test structure not found")

    atoms_original = read_structure(input_file)
    output_file = temp_output_dir / "test_output.vasp"

    write_structure(output_file, atoms_original, format="vasp")
    atoms_read = read_structure(output_file)

    assert len(atoms_read) == len(atoms_original)
    assert atoms_read.get_chemical_formula() == atoms_original.get_chemical_formula()
