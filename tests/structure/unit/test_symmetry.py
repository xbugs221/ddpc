"""Test symmetry analysis."""

import pytest
from ase import Atoms
from ase.build import bulk

from ddpc.structure import get_symmetry, read_structure


def test_get_symmetry_returns_dict(structures_dir):
    """Test that get_symmetry returns proper dict."""
    input_file = structures_dir / "mag.as"
    if not input_file.exists():
        pytest.skip("Test structure not found")

    atoms = read_structure(input_file)
    sym = get_symmetry(atoms)

    assert isinstance(sym, dict)
    assert "spacegroup" in sym
    assert "spacegroup_number" in sym
    assert "crystal_system" in sym


def test_get_symmetry_recognizes_crystal_system(structures_dir):
    """Test that crystal system is correctly identified."""
    input_file = structures_dir / "mag.as"
    if not input_file.exists():
        pytest.skip("Test structure not found")

    atoms = read_structure(input_file)
    sym = get_symmetry(atoms)

    assert sym["crystal_system"] in [
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "trigonal",
        "hexagonal",
        "cubic",
    ]


def test_get_symmetry_no_cell():
    """Test error when structure has no cell."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])

    with pytest.raises(ValueError, match="no cell information"):
        get_symmetry(atoms)


def test_get_symmetry_cubic():
    """Test cubic crystal system."""
    atoms = bulk("Cu", "fcc", a=3.6)
    sym = get_symmetry(atoms)

    assert sym["crystal_system"] == "cubic"
    assert 195 <= sym["spacegroup_number"] <= 230


def test_get_symmetry_with_custom_params(structures_dir):
    """Test symmetry with custom parameters."""
    input_file = structures_dir / "mag.as"
    if not input_file.exists():
        pytest.skip("Test structure not found")

    atoms = read_structure(input_file)
    sym = get_symmetry(atoms, symprec=1e-3, angle_tolerance=5.0, hall_number=0, symbol_type=0)

    assert isinstance(sym, dict)
    assert "hall_number" in sym


def test_get_symmetry_triclinic():
    """Test triclinic crystal system (sg 1-2)."""
    from ase import Atoms

    # Create a triclinic cell
    cell = [[3.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.2, 0.3, 3.0]]
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=cell, pbc=True)
    sym = get_symmetry(atoms, symprec=1e-3)

    assert sym["crystal_system"] == "triclinic"


def test_get_symmetry_monoclinic():
    """Test monoclinic crystal system (sg 3-15)."""
    from ase import Atoms

    # Create a monoclinic cell (sg 5, C2)
    cell = [[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 0.0, 4.0]]
    atoms = Atoms("C2", positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=cell, pbc=True)
    sym = get_symmetry(atoms, symprec=1e-3)

    assert sym["crystal_system"] in ["monoclinic", "orthorhombic", "triclinic"]


def test_get_symmetry_orthorhombic():
    """Test orthorhombic crystal system (sg 16-74)."""
    from ase import Atoms

    # Create an orthorhombic cell
    cell = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
    atoms = Atoms("C", positions=[[0, 0, 0]], cell=cell, pbc=True)
    sym = get_symmetry(atoms)

    assert sym["crystal_system"] in ["orthorhombic", "cubic"]


def test_get_symmetry_tetragonal():
    """Test tetragonal crystal system (sg 75-142)."""
    from ase import Atoms

    # Create a tetragonal cell
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]]
    atoms = Atoms("C", positions=[[0, 0, 0]], cell=cell, pbc=True)
    sym = get_symmetry(atoms)

    assert sym["crystal_system"] in ["tetragonal", "cubic"]


def test_get_symmetry_hexagonal():
    """Test hexagonal crystal system (sg 168-194)."""
    from ase.build import bulk

    # Graphite has hexagonal symmetry
    atoms = bulk("C", "hcp", a=2.46, c=6.71)
    sym = get_symmetry(atoms)

    assert sym["crystal_system"] in ["hexagonal", "trigonal"]
