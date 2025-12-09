"""Test RESCU XYZ format reader and writer."""

import numpy as np
import pytest
from ase.atoms import Atoms

from ddpc.structure.readers import rescu_xyz
from ddpc.structure.writers import rescu_xyz as rescu_xyz_writer


def test_read_basic_xyz(temp_output_dir):
    """Test reading basic XYZ format (4 columns: element x y z)."""
    xyz_file = temp_output_dir / "basic.xyz"
    content = """2
Auto-generated xyz file
Si 0.000000 0.000000 0.000000
Si 1.357500 1.357500 1.357500
"""
    xyz_file.write_text(content)

    atoms = rescu_xyz.read(xyz_file)
    assert len(atoms) == 2
    assert atoms.get_chemical_symbols() == ["Si", "Si"]
    assert np.allclose(atoms.positions[0], [0, 0, 0])
    assert np.allclose(atoms.positions[1], [1.357500, 1.357500, 1.357500])


def test_read_xyz_with_scalar_mag(temp_output_dir):
    """Test reading XYZ with scalar magnetic moments (5 columns)."""
    xyz_file = temp_output_dir / "mag_scalar.xyz"
    content = """2
XYZ with scalar magnetic moments
Fe 0.000000 0.000000 0.000000 2.5
Fe 1.435000 1.435000 1.435000 -2.5
"""
    xyz_file.write_text(content)

    atoms = rescu_xyz.read(xyz_file)
    assert len(atoms) == 2
    assert atoms.get_chemical_symbols() == ["Fe", "Fe"]
    mags = atoms.get_initial_magnetic_moments()
    assert len(mags) == 2
    assert np.allclose(mags, [2.5, -2.5])


def test_read_xyz_with_vector_mag(temp_output_dir):
    """Test reading XYZ with vector magnetic moments (7 columns)."""
    xyz_file = temp_output_dir / "mag_vector.xyz"
    content = """2
XYZ with vector magnetic moments
Fe 0.000000 0.000000 0.000000 2.5 0.0 0.5
Fe 1.435000 1.435000 1.435000 -2.5 0.0 -0.5
"""
    xyz_file.write_text(content)

    atoms = rescu_xyz.read(xyz_file)
    assert len(atoms) == 2
    mags = atoms.get_initial_magnetic_moments()
    assert len(mags) == 2
    assert np.allclose(mags[0], [2.5, 0.0, 0.5])
    assert np.allclose(mags[1], [-2.5, 0.0, -0.5])


def test_read_xyz_with_scalar_mag_and_fix(temp_output_dir):
    """Test reading XYZ with scalar mag and fix info (8 columns)."""
    xyz_file = temp_output_dir / "mag_fix.xyz"
    content = """2
XYZ with mag and fix
Fe 0.000000 0.000000 0.000000 2.5 0 0 0
Fe 1.435000 1.435000 1.435000 -2.5 1 1 1
"""
    xyz_file.write_text(content)

    atoms = rescu_xyz.read(xyz_file)
    assert len(atoms) == 2
    mags = atoms.get_initial_magnetic_moments()
    assert np.allclose(mags, [2.5, -2.5])

    fix_info = atoms.info.get("atom_fix")
    assert fix_info is not None
    assert np.allclose(fix_info[0], [0, 0, 0])
    assert np.allclose(fix_info[1], [1, 1, 1])


def test_read_xyz_with_vector_mag_and_fix(temp_output_dir):
    """Test reading XYZ with vector mag and fix info (10 columns)."""
    xyz_file = temp_output_dir / "mag_vector_fix.xyz"
    content = """2
XYZ with vector mag and fix
Fe 0.000000 0.000000 0.000000 2.5 0.0 0.5 0 0 0
Fe 1.435000 1.435000 1.435000 -2.5 0.0 -0.5 1 1 1
"""
    xyz_file.write_text(content)

    atoms = rescu_xyz.read(xyz_file)
    assert len(atoms) == 2
    mags = atoms.get_initial_magnetic_moments()
    assert np.allclose(mags[0], [2.5, 0.0, 0.5])
    assert np.allclose(mags[1], [-2.5, 0.0, -0.5])

    fix_info = atoms.info.get("atom_fix")
    assert fix_info is not None
    assert np.allclose(fix_info[0], [0, 0, 0])
    assert np.allclose(fix_info[1], [1, 1, 1])


def test_read_xyz_with_comments(temp_output_dir):
    """Test reading XYZ with comments."""
    xyz_file = temp_output_dir / "with_comments.xyz"
    content = """2  # number of atoms
Auto-generated xyz file  # comment
Si 0.000000 0.000000 0.000000  # first atom
Si 1.357500 1.357500 1.357500  % second atom
"""
    xyz_file.write_text(content)

    atoms = rescu_xyz.read(xyz_file)
    assert len(atoms) == 2


def test_read_xyz_invalid_atom_count(temp_output_dir):
    """Test error handling for inconsistent atom count."""
    xyz_file = temp_output_dir / "invalid.xyz"
    content = """3
Should have 3 atoms but only 2 provided
Si 0.000000 0.000000 0.000000
Si 1.357500 1.357500 1.357500
"""
    xyz_file.write_text(content)

    with pytest.raises(ValueError, match="Inconsistent atom count"):
        rescu_xyz.read(xyz_file)


def test_read_xyz_invalid_mag_count(temp_output_dir):
    """Test error handling for inconsistent mag count."""
    xyz_file = temp_output_dir / "invalid_mag.xyz"
    content = """2
Invalid mag count
Si 0.000000 0.000000 0.000000 2.5
Si 1.357500 1.357500 1.357500
"""
    xyz_file.write_text(content)

    with pytest.raises(ValueError, match="Inconsistent mag count"):
        rescu_xyz.read(xyz_file)


def test_read_xyz_invalid_fix_count(temp_output_dir):
    """Test error handling for inconsistent fix count."""
    xyz_file = temp_output_dir / "invalid_fix.xyz"
    content = """2
Invalid fix count
Si 0.000000 0.000000 0.000000 2.5 0 0 0
Si 1.357500 1.357500 1.357500 2.5
"""
    xyz_file.write_text(content)

    with pytest.raises(ValueError, match="Inconsistent fix count"):
        rescu_xyz.read(xyz_file)


def test_read_xyz_invalid_line(temp_output_dir):
    """Test error handling for invalid line format."""
    xyz_file = temp_output_dir / "invalid_line.xyz"
    content = """1
Invalid line
Si 0.000000 0.000000
"""
    xyz_file.write_text(content)

    with pytest.raises(ValueError, match="Invalid line"):
        rescu_xyz.read(xyz_file)


def test_write_basic_xyz(temp_output_dir):
    """Test writing basic XYZ format."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.3575, 1.3575, 1.3575]], pbc=True)
    output_file = temp_output_dir / "output.xyz"

    rescu_xyz_writer.write(str(output_file), atoms)

    assert output_file.exists()
    lines = output_file.read_text().strip().split("\n")
    assert lines[0] == "2"
    assert "Si" in lines[2]
    assert "Si" in lines[3]


def test_write_xyz_with_scalar_mag(temp_output_dir):
    """Test writing XYZ with scalar magnetic moments."""
    atoms = Atoms(
        "Fe2", positions=[[0, 0, 0], [1.435, 1.435, 1.435]], magmoms=[2.5, -2.5], pbc=True
    )
    output_file = temp_output_dir / "output_mag.xyz"

    rescu_xyz_writer.write(str(output_file), atoms)

    assert output_file.exists()
    content = output_file.read_text()
    assert "2.50" in content
    assert "-2.50" in content


def test_write_xyz_with_vector_mag(temp_output_dir):
    """Test writing XYZ with vector magnetic moments."""
    atoms = Atoms(
        "Fe2",
        positions=[[0, 0, 0], [1.435, 1.435, 1.435]],
        magmoms=[[2.5, 0.0, 0.5], [-2.5, 0.0, -0.5]],
        pbc=True,
    )
    output_file = temp_output_dir / "output_mag_vec.xyz"

    rescu_xyz_writer.write(str(output_file), atoms)

    assert output_file.exists()
    content = output_file.read_text()
    assert "2.50" in content
    assert "0.50" in content
    assert "-0.50" in content


def test_write_xyz_with_fix_info(temp_output_dir):
    """Test writing XYZ with fix information."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.3575, 1.3575, 1.3575]], pbc=True)
    atoms.info["atom_fix"] = np.array([[0, 0, 0], [1, 1, 1]])
    output_file = temp_output_dir / "output_fix.xyz"

    rescu_xyz_writer.write(str(output_file), atoms)

    assert output_file.exists()
    content = output_file.read_text()
    # Should include zeros for mag and fix info
    lines = content.strip().split("\n")
    # First atom should have "0 0 0 0 0 0" (no mag, fixed in all directions)
    assert "0 0 0 0 0 0" in lines[2]
    # Second atom should have "0 0 0 1 1 1" (no mag, moveable in all directions)
    assert "0 0 0 1 1 1" in lines[3]


def test_write_xyz_with_scalar_mag_and_fix(temp_output_dir):
    """Test writing XYZ with scalar mag and fix info."""
    atoms = Atoms(
        "Fe2", positions=[[0, 0, 0], [1.435, 1.435, 1.435]], magmoms=[2.5, -2.5], pbc=True
    )
    atoms.info["atom_fix"] = np.array([[0, 0, 0], [1, 1, 1]])
    output_file = temp_output_dir / "output_mag_fix.xyz"

    rescu_xyz_writer.write(str(output_file), atoms)

    assert output_file.exists()
    content = output_file.read_text()
    content.strip().split("\n")
    # Check that both mag and fix info are present
    assert "2.50" in content
    assert "0 0 0" in content
    assert "1 1 1" in content


def test_write_xyz_with_vector_mag_and_fix(temp_output_dir):
    """Test writing XYZ with vector mag and fix info."""
    atoms = Atoms(
        "Fe2",
        positions=[[0, 0, 0], [1.435, 1.435, 1.435]],
        magmoms=[[2.5, 0.0, 0.5], [-2.5, 0.0, -0.5]],
        pbc=True,
    )
    atoms.info["atom_fix"] = np.array([[0, 0, 0], [1, 1, 1]])
    output_file = temp_output_dir / "output_mag_vec_fix.xyz"

    rescu_xyz_writer.write(str(output_file), atoms)

    assert output_file.exists()
    content = output_file.read_text()
    # Check vector mag components and fix info
    assert "2.50" in content
    assert "0.50" in content
    assert "0 0 0" in content
    assert "1 1 1" in content


def test_write_xyz_invalid_mag_shape(temp_output_dir):
    """Test error handling for invalid magnetic moment shape."""
    atoms = Atoms(
        "Fe2",
        positions=[[0, 0, 0], [1.435, 1.435, 1.435]],
        magmoms=[[2.5, 0.0], [-2.5, 0.0]],
        pbc=True,
    )  # Invalid 2D shape
    output_file = temp_output_dir / "invalid_mag.xyz"

    with pytest.raises(ValueError, match="Invalid magnetic moment shape"):
        rescu_xyz_writer.write(str(output_file), atoms)


def test_roundtrip_basic(temp_output_dir):
    """Test write-read roundtrip for basic structure."""
    atoms_orig = Atoms("Si2", positions=[[0, 0, 0], [1.3575, 1.3575, 1.3575]], pbc=True)
    xyz_file = temp_output_dir / "roundtrip.xyz"

    rescu_xyz_writer.write(str(xyz_file), atoms_orig)
    atoms_read = rescu_xyz.read(xyz_file)

    assert len(atoms_read) == len(atoms_orig)
    assert atoms_read.get_chemical_symbols() == atoms_orig.get_chemical_symbols()
    assert np.allclose(atoms_read.positions, atoms_orig.positions, atol=1e-5)


def test_roundtrip_with_mag(temp_output_dir):
    """Test write-read roundtrip with magnetic moments."""
    atoms_orig = Atoms(
        "Fe2", positions=[[0, 0, 0], [1.435, 1.435, 1.435]], magmoms=[2.5, -2.5], pbc=True
    )
    xyz_file = temp_output_dir / "roundtrip_mag.xyz"

    rescu_xyz_writer.write(str(xyz_file), atoms_orig)
    atoms_read = rescu_xyz.read(xyz_file)

    assert len(atoms_read) == len(atoms_orig)
    assert np.allclose(
        atoms_read.get_initial_magnetic_moments(),
        atoms_orig.get_initial_magnetic_moments(),
        atol=1e-2,
    )


def test_roundtrip_with_vector_mag(temp_output_dir):
    """Test write-read roundtrip with vector magnetic moments."""
    atoms_orig = Atoms(
        "Fe2",
        positions=[[0, 0, 0], [1.435, 1.435, 1.435]],
        magmoms=[[2.5, 0.0, 0.5], [-2.5, 0.0, -0.5]],
        pbc=True,
    )
    xyz_file = temp_output_dir / "roundtrip_mag_vec.xyz"

    rescu_xyz_writer.write(str(xyz_file), atoms_orig)
    atoms_read = rescu_xyz.read(xyz_file)

    assert len(atoms_read) == len(atoms_orig)
    assert np.allclose(
        atoms_read.get_initial_magnetic_moments(),
        atoms_orig.get_initial_magnetic_moments(),
        atol=1e-2,
    )


def test_roundtrip_with_fix(temp_output_dir):
    """Test write-read roundtrip with fix information."""
    atoms_orig = Atoms("Si2", positions=[[0, 0, 0], [1.3575, 1.3575, 1.3575]], pbc=True)
    atoms_orig.info["atom_fix"] = np.array([[0, 0, 0], [1, 1, 1]])
    xyz_file = temp_output_dir / "roundtrip_fix.xyz"

    rescu_xyz_writer.write(str(xyz_file), atoms_orig)
    atoms_read = rescu_xyz.read(xyz_file)

    assert len(atoms_read) == len(atoms_orig)
    assert np.allclose(atoms_read.info["atom_fix"], atoms_orig.info["atom_fix"])
