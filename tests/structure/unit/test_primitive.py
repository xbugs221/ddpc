"""Test primitive cell finding."""

import pytest
from ase import Atoms

from ddpc.structure import find_primitive, read_structure


def test_find_primitive_reduces_size(structures_dir):
    """Test that primitive cell has fewer or equal atoms."""
    input_file = structures_dir / "all.as"
    if not input_file.exists():
        pytest.skip("Test structure not found")

    original = read_structure(input_file)
    prim = find_primitive(original, symprec=1e-5)

    assert len(prim) <= len(original)


def test_find_primitive_preserves_composition(structures_dir):
    """Test that primitive cell preserves chemical composition ratios."""
    input_file = structures_dir / "mag.as"
    if not input_file.exists():
        pytest.skip("Test structure not found")

    original = read_structure(input_file)
    prim = find_primitive(original, symprec=1e-5)

    # Primitive cell should preserve element types
    assert set(prim.get_chemical_symbols()) == set(original.get_chemical_symbols())


def test_find_primitive_no_cell():
    """Test error when structure has no cell."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])

    with pytest.raises(ValueError, match="no cell information"):
        find_primitive(atoms)


def test_find_primitive_with_angle_tolerance(structures_dir):
    """Test primitive cell finding with custom angle tolerance."""
    input_file = structures_dir / "mag.as"
    if not input_file.exists():
        pytest.skip("Test structure not found")

    original = read_structure(input_file)
    prim = find_primitive(original, symprec=1e-5, angle_tolerance=5.0)

    assert len(prim) <= len(original)
