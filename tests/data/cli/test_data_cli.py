"""CLI command tests: Test command line interface using Click CliRunner."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from ddpc.data.cli import cli


class TestCLIBasics:
    """Test basic CLI functionality."""

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
        assert "Electronic structure data" in result.output
        assert "band" in result.output
        assert "dos" in result.output

    def test_cli_no_args_shows_help(self, runner):
        """Test shows help message when no arguments provided."""
        result = runner.invoke(cli, [])
        assert result.exit_code == 0


class TestBandCommands:
    """Test band command group."""

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
                bandinfo.create_dataset("IsProject", data=[0])  # No projection
                bandinfo.create_dataset("NumberOfKpoints", data=[nkpt])
                bandinfo.create_dataset("NumberOfBand", data=[nband])

                # Create k-point coordinates
                kpoint = np.random.rand(nkpt, 3)
                bandinfo.create_dataset("CoordinatesOfKPoints", data=kpoint.flatten())

                # Symmetry point info - use single symmetry point to avoid complexity
                bandinfo.create_dataset("SymmetryKPoints", data=np.array(list("G"), dtype="S1"))
                bandinfo.create_dataset("SymmetryKPointsIndex", data=[1])
                # Use byte array format like real files
                bandinfo["SpinType"] = np.array(list("spinless"), dtype="S1")

                # Create Spin1 group and band data
                spin1 = bandinfo.create_group("Spin1")
                # Band energies: nband x nkpt in Fortran order
                bands = np.random.rand(nband, nkpt) + np.arange(nband)[:, None]
                spin1.create_dataset("BandEnergies", data=bands.T)  # Transpose for C order storage

            yield band_file

    def test_band_help(self, runner):
        """Test band --help."""
        result = runner.invoke(cli, ["band", "--help"])
        assert result.exit_code == 0
        assert "Band structure commands" in result.output

    def test_band_read_help(self, runner):
        """Test band read --help."""
        result = runner.invoke(cli, ["band", "read", "--help"])
        assert result.exit_code == 0
        assert "Read band structure data" in result.output
        assert "--mode" in result.output
        assert "--format" in result.output

    def test_band_read_preview(self, runner, sample_band_file):
        """Test reading band data and preview."""
        result = runner.invoke(cli, ["band", "read", str(sample_band_file)])

        assert result.exit_code == 0
        assert "Fermi energy:" in result.output
        assert "5.0" in result.output
        assert "K-points:" in result.output

    def test_band_read_export_csv(self, runner, sample_band_file):
        """Test reading band data and export to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "band.csv"

            result = runner.invoke(
                cli,
                [
                    "band",
                    "read",
                    str(sample_band_file),
                    "-o",
                    str(output_file),
                    "--format",
                    "csv",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Saved to" in result.output

    def test_band_read_export_npz(self, runner, sample_band_file):
        """Test reading band data and export to NPZ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "band.npz"

            result = runner.invoke(
                cli,
                [
                    "band",
                    "read",
                    str(sample_band_file),
                    "-o",
                    str(output_file),
                    "--format",
                    "npz",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()

    def test_band_read_different_mode(self, runner, sample_band_file):
        """Test using different projection mode."""
        result = runner.invoke(cli, ["band", "read", str(sample_band_file), "--mode", "0"])

        assert result.exit_code == 0
        assert "Projection mode:" in result.output
        assert "0" in result.output

    def test_band_info(self, runner, sample_band_file):
        """Test band info command."""
        result = runner.invoke(cli, ["band", "info", str(sample_band_file)])

        assert result.exit_code == 0
        assert "Band Structure Information" in result.output
        assert "Fermi Energy" in result.output
        assert "5.0" in result.output

    def test_band_read_nonexistent_file(self, runner):
        """Test reading nonexistent file."""
        result = runner.invoke(cli, ["band", "read", "/nonexistent/file.h5"])
        # Click will error when file does not exist
        assert result.exit_code != 0

    def test_band_read_missing_arguments(self, runner):
        """Test missing required arguments."""
        result = runner.invoke(cli, ["band", "read"])
        assert result.exit_code == 2
        assert "Usage:" in result.output


class TestDOSCommands:
    """Test dos command group."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def sample_dos_file(self):
        """Create test DOS HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dos_file = Path(tmpdir) / "dos.h5"

            with h5py.File(dos_file, "w") as f:
                # Create DosInfo group
                dosinfo = f.create_group("DosInfo")
                dosinfo.create_dataset("EFermi", data=[5.5])
                dosinfo.create_dataset("Project", data=[0])  # No projection
                # Use byte array format like real files
                dosinfo["SpinType"] = np.array(list("spinless"), dtype="S1")

                # Create energy and DOS data
                energy = np.linspace(-10, 10, 100)
                dos_total = np.exp(-((energy - 5.5) ** 2) / 2.0)  # Gaussian-like

                dosinfo.create_dataset("DosEnergy", data=energy)

                # Create Spin1 group and DOS data
                spin1 = dosinfo.create_group("Spin1")
                spin1.create_dataset("Dos", data=dos_total)

            yield dos_file

    def test_dos_help(self, runner):
        """Test dos --help."""
        result = runner.invoke(cli, ["dos", "--help"])
        assert result.exit_code == 0
        assert "Density of states commands" in result.output

    def test_dos_read_help(self, runner):
        """Test dos read --help."""
        result = runner.invoke(cli, ["dos", "read", "--help"])
        assert result.exit_code == 0
        assert "Read density of states data" in result.output

    def test_dos_read_preview(self, runner, sample_dos_file):
        """Test reading DOS data and preview."""
        result = runner.invoke(cli, ["dos", "read", str(sample_dos_file)])

        assert result.exit_code == 0
        assert "Fermi energy:" in result.output
        assert "5.5" in result.output
        assert "Energy points:" in result.output

    def test_dos_read_export_csv(self, runner, sample_dos_file):
        """Test reading DOS data and export to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "dos.csv"

            result = runner.invoke(
                cli,
                [
                    "dos",
                    "read",
                    str(sample_dos_file),
                    "-o",
                    str(output_file),
                    "--format",
                    "csv",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Saved to" in result.output

    def test_dos_read_export_npz(self, runner, sample_dos_file):
        """Test reading DOS data and export to NPZ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "dos.npz"

            result = runner.invoke(
                cli,
                [
                    "dos",
                    "read",
                    str(sample_dos_file),
                    "-o",
                    str(output_file),
                    "--format",
                    "npz",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()

    def test_dos_read_with_mode(self, runner, sample_dos_file):
        """Test using different mode parameter."""
        result = runner.invoke(cli, ["dos", "read", str(sample_dos_file), "--mode", "0"])

        assert result.exit_code == 0
        assert "Projection mode:" in result.output

    def test_dos_info(self, runner, sample_dos_file):
        """Test dos info command."""
        result = runner.invoke(cli, ["dos", "info", str(sample_dos_file)])

        assert result.exit_code == 0
        assert "DOS Information" in result.output
        assert "Fermi Energy" in result.output
        assert "5.5" in result.output
        assert "Energy Points" in result.output

    def test_dos_info_shows_energy_range(self, runner, sample_dos_file):
        """Test info command shows energy range."""
        result = runner.invoke(cli, ["dos", "info", str(sample_dos_file)])

        assert result.exit_code == 0
        assert "Energy Range" in result.output
        # Should contain range from -10 to 10
        assert "-10" in result.output or "10" in result.output


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
        result = runner.invoke(cli, ["band", "invalid_subcommand"])
        assert result.exit_code != 0

    def test_invalid_format_option(self, runner):
        """Test invalid format option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple test file
            test_file = Path(tmpdir) / "test.h5"
            with h5py.File(test_file, "w") as f:
                f.create_dataset("efermi", data=0.0)

            result = runner.invoke(
                cli,
                [
                    "band",
                    "read",
                    str(test_file),
                    "-o",
                    str(Path(tmpdir) / "out.txt"),
                    "--format",
                    "invalid",
                ],
            )

            # Click Choice will error on invalid option
            assert result.exit_code != 0


class TestCLIWithRealData:
    """Integration tests using real test data (if available)."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def real_data_dir(self):
        """Get real test data directory."""
        # Try to find test data in project root directory
        test_data = Path(__file__).parent.parent / "raw"
        if test_data.exists():
            return test_data
        return None

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "raw").exists(),
        reason="Real test data not available",
    )
    def test_band_read_real_data(self, runner, real_data_dir):
        """Test with real band data (if available)."""
        # Find first .h5 file
        h5_files = list(real_data_dir.glob("*band*.h5"))
        if not h5_files:
            pytest.skip("No real band HDF5 files found")

        band_file = h5_files[0]
        result = runner.invoke(cli, ["band", "info", str(band_file)])

        # Should be able to read successfully
        assert result.exit_code == 0
        assert "Fermi Energy" in result.output

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "raw").exists(),
        reason="Real test data not available",
    )
    def test_dos_read_real_data(self, runner, real_data_dir):
        """Test with real DOS data (if available)."""
        # Find first DOS .h5 file
        h5_files = list(real_data_dir.glob("*dos*.h5"))
        if not h5_files:
            pytest.skip("No real DOS HDF5 files found")

        dos_file = h5_files[0]
        result = runner.invoke(cli, ["dos", "info", str(dos_file)])

        # Should be able to read successfully
        assert result.exit_code == 0
        assert "Fermi Energy" in result.output
