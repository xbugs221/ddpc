"""Integration tests: Verify complete workflow for multiple structure format conversions."""

import tempfile
from pathlib import Path

import pytest
from ase.build import bulk

from ddpc.structure import read_structure, write_structure


class TestFormatConversion:
    """Integration tests for structure format conversion."""

    @pytest.fixture
    def sample_structure(self):
        """Create test structure (simple cubic)."""
        return bulk("Cu", "fcc", a=3.6)

    def test_vasp_to_cif_conversion(self, sample_structure):
        """Test VASP to CIF format conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write VASP format
            vasp_file = tmpdir / "structure.vasp"
            write_structure(vasp_file, sample_structure, format="vasp")
            assert vasp_file.exists()

            # Read and convert to CIF
            atoms = read_structure(vasp_file)
            cif_file = tmpdir / "structure.cif"
            write_structure(cif_file, atoms, format="cif")
            assert cif_file.exists()

            # Verify structure consistency after reading CIF
            atoms_cif = read_structure(cif_file)
            assert len(atoms_cif) == len(sample_structure)
            assert atoms_cif.get_chemical_formula() == sample_structure.get_chemical_formula()

    def test_vasp_to_xyz_conversion(self, sample_structure):
        """Test VASP to XYZ format conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # VASP -> XYZ
            vasp_file = tmpdir / "structure.vasp"
            write_structure(vasp_file, sample_structure)

            atoms = read_structure(vasp_file)
            xyz_file = tmpdir / "structure.xyz"
            write_structure(xyz_file, atoms)

            # Verify XYZ
            atoms_xyz = read_structure(xyz_file)
            assert len(atoms_xyz) == len(sample_structure)

    def test_cif_to_vasp_conversion(self, sample_structure):
        """Test CIF to VASP format conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write CIF
            cif_file = tmpdir / "structure.cif"
            write_structure(cif_file, sample_structure, format="cif")

            # Read and convert to VASP
            atoms = read_structure(cif_file)
            vasp_file = tmpdir / "POSCAR"
            write_structure(vasp_file, atoms, format="vasp", vasp5=True)

            # Verify
            atoms_vasp = read_structure(vasp_file)
            assert len(atoms_vasp) == len(sample_structure)

    def test_multiple_format_roundtrip(self, sample_structure):
        """Test round-trip conversion for multiple formats."""
        formats = ["vasp", "cif", "xyz", "xsf"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for fmt in formats:
                # Write
                output_file = tmpdir / f"structure.{fmt}"
                write_structure(output_file, sample_structure, format=fmt)
                assert output_file.exists(), f"Failed to write {fmt}"

                # Read
                atoms = read_structure(output_file)
                assert len(atoms) == len(sample_structure)
                assert atoms.get_chemical_formula() == sample_structure.get_chemical_formula()

    def test_conversion_preserves_cell(self, sample_structure):
        """Test if format conversion preserves cell parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # VASP -> CIF -> VASP
            vasp1 = tmpdir / "input.vasp"
            write_structure(vasp1, sample_structure)

            atoms1 = read_structure(vasp1)
            cif_file = tmpdir / "temp.cif"
            write_structure(cif_file, atoms1)

            atoms2 = read_structure(cif_file)
            vasp2 = tmpdir / "output.vasp"
            write_structure(vasp2, atoms2)

            atoms3 = read_structure(vasp2)

            # Verify cell parameters
            import numpy as np

            np.testing.assert_allclose(
                atoms3.cell.cellpar(), sample_structure.cell.cellpar(), rtol=1e-5
            )

    def test_conversion_with_supercell(self):
        """Test format conversion for supercell structures."""
        # Create 2x2x2 supercell
        structure = bulk("Si", "diamond", a=5.43)
        supercell = structure * (2, 2, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Convert to multiple formats
            for fmt in ["vasp", "cif", "xyz"]:
                output = tmpdir / f"supercell.{fmt}"
                write_structure(output, supercell, format=fmt)

                atoms = read_structure(output)
                assert len(atoms) == len(supercell)  # 8 atoms * 8 = 64 atoms

    def test_dspaw_as_format_if_available(self):
        """Test DS-PAW .as format read/write (if implemented)."""
        try:
            from ddpc.structure.readers.dspaw_as import read
            from ddpc.structure.writers.dspaw_as import write

            structure = bulk("Fe", "bcc", a=2.87)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                as_file = tmpdir / "structure.as"

                # Try to write and read
                write(as_file, structure)
                if as_file.exists():
                    atoms = read(as_file)
                    assert len(atoms) == len(structure)
        except ImportError:
            pytest.skip("DS-PAW format not fully implemented")

    def test_conversion_error_handling(self):
        """Test error handling in format conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Try to read nonexistent file
            with pytest.raises(FileNotFoundError):
                read_structure(tmpdir / "nonexistent.vasp")

            # Try to write invalid format
            from ase.io.formats import UnknownFileTypeError

            atoms = bulk("Au", "fcc", a=4.08)
            with pytest.raises((ValueError, UnknownFileTypeError)):
                write_structure(tmpdir / "test.unknown", atoms, format="invalid_format")
