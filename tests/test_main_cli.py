"""Integration tests for the unified CLI entry point (ddpc.cli).

This tests the main CLI that delegates to subpackages (data, structure).
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase.build import bulk
from ase.io import write as ase_write
from click.testing import CliRunner

from ddpc.cli import cli


class TestMainCLI:
    """Test the main unified CLI."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_cli_version(self, runner):
        """Test --version option."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "2026.1.0" in result.output

    def test_cli_help(self, runner):
        """Test --help option."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "DDPC" in result.output
        assert "data" in result.output
        assert "structure" in result.output

    def test_cli_no_args_shows_help(self, runner):
        """Test shows help message when no arguments provided."""
        result = runner.invoke(cli, [])
        # Click groups without a default command return exit code 0
        # when invoked without arguments (shows help)
        assert result.exit_code in (0, 2)  # Both are acceptable
        assert "Usage:" in result.output or "DDPC" in result.output


class TestDataSubcommands:
    """Test data subcommands through unified CLI."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def sample_band_file(self):
        """Create test band HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            band_file = Path(tmpdir) / "band.h5"

            nkpt = 10
            nband = 5
            with h5py.File(band_file, "w") as f:
                # Create BandInfo group
                bandinfo = f.create_group("BandInfo")
                bandinfo.create_dataset("EFermi", data=[5.0])
                bandinfo.create_dataset("IsProject", data=[0])
                bandinfo.create_dataset("NumberOfKpoints", data=[nkpt])
                bandinfo.create_dataset("NumberOfBand", data=[nband])

                # Create k-point coordinates
                kpoint = np.random.rand(nkpt, 3)
                bandinfo.create_dataset("CoordinatesOfKPoints", data=kpoint.flatten())

                # Symmetry point info
                bandinfo.create_dataset("SymmetryKPoints", data=np.array(list("G"), dtype="S1"))
                bandinfo.create_dataset("SymmetryKPointsIndex", data=[1])
                bandinfo["SpinType"] = np.array(list("spinless"), dtype="S1")

                # Create Spin1 group and band data
                spin1 = bandinfo.create_group("Spin1")
                bands = np.random.rand(nband, nkpt) + np.arange(nband)[:, None]
                spin1.create_dataset("BandEnergies", data=bands.T)

            yield band_file

    @pytest.fixture
    def sample_dos_file(self):
        """Create test DOS HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dos_file = Path(tmpdir) / "dos.h5"

            with h5py.File(dos_file, "w") as f:
                # Create DosInfo group
                dosinfo = f.create_group("DosInfo")
                dosinfo.create_dataset("EFermi", data=[5.5])
                dosinfo.create_dataset("Project", data=[0])
                dosinfo["SpinType"] = np.array(list("spinless"), dtype="S1")

                # Create energy and DOS data
                energy = np.linspace(-10, 10, 100)
                dos_total = np.exp(-((energy - 5.5) ** 2) / 2.0)

                dosinfo.create_dataset("DosEnergy", data=energy)

                # Create Spin1 group and DOS data
                spin1 = dosinfo.create_group("Spin1")
                spin1.create_dataset("Dos", data=dos_total)

            yield dos_file

    def test_data_help(self, runner):
        """Test data --help."""
        result = runner.invoke(cli, ["data", "--help"])
        assert result.exit_code == 0
        assert "Electronic structure data" in result.output

    def test_data_band_help(self, runner):
        """Test data band --help."""
        result = runner.invoke(cli, ["data", "band", "--help"])
        assert result.exit_code == 0
        assert "Band structure commands" in result.output

    def test_data_band_read(self, runner, sample_band_file):
        """Test data band read command."""
        result = runner.invoke(cli, ["data", "band", "read", str(sample_band_file)])
        assert result.exit_code == 0
        assert "Fermi energy:" in result.output

    def test_data_band_info(self, runner, sample_band_file):
        """Test data band info command."""
        result = runner.invoke(cli, ["data", "band", "info", str(sample_band_file)])
        assert result.exit_code == 0
        assert "Band Structure Information" in result.output

    def test_data_dos_help(self, runner):
        """Test data dos --help."""
        result = runner.invoke(cli, ["data", "dos", "--help"])
        assert result.exit_code == 0
        assert "Density of states commands" in result.output

    def test_data_dos_read(self, runner, sample_dos_file):
        """Test data dos read command."""
        result = runner.invoke(cli, ["data", "dos", "read", str(sample_dos_file)])
        assert result.exit_code == 0
        assert "Fermi energy:" in result.output

    def test_data_dos_info(self, runner, sample_dos_file):
        """Test data dos info command."""
        result = runner.invoke(cli, ["data", "dos", "info", str(sample_dos_file)])
        assert result.exit_code == 0
        assert "DOS Information" in result.output


class TestStructureSubcommands:
    """Test structure subcommands through unified CLI."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def sample_structure_file(self):
        """Create test structure file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = bulk("Si", "diamond", a=5.43)
            vasp_file = Path(tmpdir) / "POSCAR"
            ase_write(vasp_file, structure, format="vasp")
            yield vasp_file

    def test_structure_help(self, runner):
        """Test structure --help."""
        result = runner.invoke(cli, ["structure", "--help"])
        assert result.exit_code == 0
        assert "Crystal structure commands" in result.output

    def test_structure_convert(self, runner, sample_structure_file):
        """Test structure convert command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.cif"
            result = runner.invoke(
                cli,
                ["structure", "convert", str(sample_structure_file), str(output_file), "--format", "cif"],
            )
            assert result.exit_code == 0
            assert output_file.exists()

    def test_structure_info(self, runner, sample_structure_file):
        """Test structure info command."""
        result = runner.invoke(cli, ["structure", "info", str(sample_structure_file)])
        assert result.exit_code == 0
        assert "Structure Information" in result.output
        assert "Si" in result.output

    def test_structure_primitive(self, runner):
        """Test structure primitive command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a 2x2x2 supercell
            structure = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
            input_file = tmpdir / "supercell.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "primitive.vasp"

            result = runner.invoke(
                cli,
                ["structure", "primitive", str(input_file), "-o", str(output_file)],
            )

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Reduced by" in result.output

    def test_structure_orthogonal(self, runner):
        """Test structure orthogonal command."""
        from ase import Atoms

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a non-orthogonal structure
            cell = [[3.0, 0.0, 0.0], [1.5, 2.6, 0.0], [0.0, 0.0, 4.0]]
            structure = Atoms("C2", positions=[[0, 0, 0], [1, 1, 2]], cell=cell, pbc=True)

            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "orthogonal.vasp"

            result = runner.invoke(
                cli,
                ["structure", "orthogonal", str(input_file), "-o", str(output_file)],
            )

            assert result.exit_code == 0
            assert output_file.exists()

    def test_structure_scale(self, runner, sample_structure_file):
        """Test structure scale command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "scaled.vasp"
            result = runner.invoke(
                cli,
                ["structure", "scale", str(sample_structure_file), "-o", str(output_file)],
            )
            assert result.exit_code == 0
            assert output_file.exists()


class TestCLIErrorHandling:
    """Test CLI error handling."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_invalid_command(self, runner):
        """Test invalid command."""
        result = runner.invoke(cli, ["invalid_command"])
        assert result.exit_code != 0

    def test_invalid_subcommand(self, runner):
        """Test invalid subcommand."""
        result = runner.invoke(cli, ["data", "invalid_subcommand"])
        assert result.exit_code != 0

    def test_invalid_structure_subcommand(self, runner):
        """Test invalid structure subcommand."""
        result = runner.invoke(cli, ["structure", "invalid_subcommand"])
        assert result.exit_code != 0


@pytest.mark.integration
class TestCLIWorkflows:
    """Test complete CLI workflows."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_structure_conversion_workflow(self, runner):
        """Test a complete structure conversion workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create initial structure
            structure = bulk("Fe", "bcc", a=2.87)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            # Step 1: Get info
            result = runner.invoke(cli, ["structure", "info", str(input_file)])
            assert result.exit_code == 0

            # Step 2: Convert to CIF
            cif_file = tmpdir / "output.cif"
            result = runner.invoke(
                cli,
                ["structure", "convert", str(input_file), str(cif_file), "--format", "cif"],
            )
            assert result.exit_code == 0
            assert cif_file.exists()

            # Step 3: Get info from CIF
            result = runner.invoke(cli, ["structure", "info", str(cif_file)])
            assert result.exit_code == 0

    def test_data_export_workflow(self, runner):
        """Test a complete data export workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test DOS file
            dos_file = tmpdir / "dos.h5"
            with h5py.File(dos_file, "w") as f:
                dosinfo = f.create_group("DosInfo")
                dosinfo.create_dataset("EFermi", data=[5.5])
                dosinfo.create_dataset("Project", data=[0])
                dosinfo["SpinType"] = np.array(list("spinless"), dtype="S1")

                energy = np.linspace(-10, 10, 100)
                dos_total = np.exp(-((energy - 5.5) ** 2) / 2.0)

                dosinfo.create_dataset("DosEnergy", data=energy)
                spin1 = dosinfo.create_group("Spin1")
                spin1.create_dataset("Dos", data=dos_total)

            # Step 1: Get info
            result = runner.invoke(cli, ["data", "dos", "info", str(dos_file)])
            assert result.exit_code == 0

            # Step 2: Export to CSV
            csv_file = tmpdir / "dos.csv"
            result = runner.invoke(
                cli,
                ["data", "dos", "read", str(dos_file), "-o", str(csv_file), "--format", "csv"],
            )
            assert result.exit_code == 0
            assert csv_file.exists()
