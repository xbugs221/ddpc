"""CLI command tests: Test command line interface using Click CliRunner."""

import tempfile
from pathlib import Path

import pytest
from ase.build import bulk
from ase.io import write as ase_write
from click.testing import CliRunner

from ddpc.structure.cli import cli


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
        assert "Crystal structure I/O" in result.output
        assert "convert" in result.output
        assert "info" in result.output
        assert "primitive" in result.output

    def test_cli_no_args_shows_help(self, runner):
        """Test shows help message when no arguments provided."""
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output


class TestConvertCommand:
    """Test convert command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def sample_vasp_file(self):
        """Create test VASP file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = bulk("Cu", "fcc", a=3.6)
            vasp_file = Path(tmpdir) / "POSCAR"
            ase_write(vasp_file, structure, format="vasp")
            yield vasp_file

    def test_convert_help(self, runner):
        """Test convert --help."""
        result = runner.invoke(cli, ["convert", "--help"])
        assert result.exit_code == 0
        assert "Convert structure file formats" in result.output
        assert "--format" in result.output

    def test_convert_vasp_to_cif(self, runner):
        """Test VASP to CIF conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input file
            structure = bulk("Si", "diamond", a=5.43)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "output.cif"

            result = runner.invoke(
                cli, ["convert", str(input_file), str(output_file), "--format", "cif"]
            )

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Converted to" in result.output

    def test_convert_vasp_to_xyz(self, runner):
        """Test VASP to XYZ conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            structure = bulk("Fe", "bcc", a=2.87)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "output.xyz"

            result = runner.invoke(cli, ["convert", str(input_file), str(output_file)])

            assert result.exit_code == 0
            assert output_file.exists()

    def test_convert_with_vasp5_flag(self, runner):
        """Test using --vasp5 flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            structure = bulk("Au", "fcc", a=4.08)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "POSCAR"

            result = runner.invoke(cli, ["convert", str(input_file), str(output_file), "--vasp5"])

            assert result.exit_code == 0
            assert output_file.exists()

    def test_convert_missing_input_shows_help(self, runner):
        """Test error handling when input file is missing."""
        result = runner.invoke(cli, ["convert"])
        # According to FriendlyCommand implementation, should show help and return exit code 2
        assert result.exit_code == 2
        assert "Usage:" in result.output

    def test_convert_nonexistent_file(self, runner):
        """Test converting nonexistent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                cli,
                [
                    "convert",
                    str(Path(tmpdir) / "nonexistent.vasp"),
                    str(Path(tmpdir) / "output.cif"),
                ],
            )
            # Click will error when file does not exist
            assert result.exit_code != 0


class TestInfoCommand:
    """Test info command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_info_help(self, runner):
        """Test info --help."""
        result = runner.invoke(cli, ["info", "--help"])
        assert result.exit_code == 0
        assert "Display structure information" in result.output
        assert "--symprec" in result.output

    def test_info_basic(self, runner):
        """Test basic info command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            structure = bulk("Si", "diamond", a=5.43)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            result = runner.invoke(cli, ["info", str(input_file)])

            assert result.exit_code == 0
            assert "Structure Information" in result.output
            assert "Si" in result.output  # Formula
            assert "Volume" in result.output

    def test_info_with_symmetry(self, runner):
        """Test info command with symmetry information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            structure = bulk("Cu", "fcc", a=3.6)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            result = runner.invoke(cli, ["info", str(input_file), "--show-symmetry"])

            assert result.exit_code == 0
            assert "Space group" in result.output or "Warning" in result.output

    def test_info_custom_symprec(self, runner):
        """Test custom symmetry precision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            structure = bulk("Al", "fcc", a=4.05)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            result = runner.invoke(cli, ["info", str(input_file), "--symprec", "0.001"])

            assert result.exit_code == 0

    def test_info_no_cell(self, runner):
        """Test info with structure without cell."""
        from ase import Atoms

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Structure without cell
            structure = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
            input_file = tmpdir / "input.xyz"
            ase_write(input_file, structure, format="xyz")

            result = runner.invoke(cli, ["info", str(input_file)])

            assert result.exit_code == 0
            assert "Unknown" in result.output

    def test_info_symmetry_with_custom_params(self, runner):
        """Test symmetry with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            structure = bulk("Si", "diamond", a=5.43)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            result = runner.invoke(
                cli,
                [
                    "info",
                    str(input_file),
                    "--show-symmetry",
                    "--angle-tolerance",
                    "5.0",
                    "--hall-number",
                    "0",
                    "--symbol-type",
                    "0",
                ],
            )

            assert result.exit_code == 0


class TestPrimitiveCommand:
    """Test primitive command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_primitive_help(self, runner):
        """Test primitive --help."""
        result = runner.invoke(cli, ["primitive", "--help"])
        assert result.exit_code == 0
        assert "Find primitive cell" in result.output

    def test_primitive_basic(self, runner):
        """Test basic primitive command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a 2x2x2 supercell
            structure = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
            input_file = tmpdir / "supercell.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "primitive.vasp"

            result = runner.invoke(cli, ["primitive", str(input_file), "-o", str(output_file)])

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Reduced by" in result.output

    def test_primitive_custom_output(self, runner):
        """Test custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            structure = bulk("Fe", "bcc", a=2.87) * (2, 2, 2)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "custom" / "prim.vasp"

            result = runner.invoke(cli, ["primitive", str(input_file), "-o", str(output_file)])

            assert result.exit_code == 0
            assert output_file.exists()

    def test_primitive_different_format(self, runner):
        """Test output in different format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            structure = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "primitive.cif"

            result = runner.invoke(
                cli,
                [
                    "primitive",
                    str(input_file),
                    "-o",
                    str(output_file),
                    "--format",
                    "cif",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()

    def test_primitive_with_angle_tolerance(self, runner):
        """Test primitive with angle tolerance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            structure = bulk("Fe", "bcc", a=2.87) * (2, 2, 2)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "primitive.vasp"

            result = runner.invoke(
                cli,
                [
                    "primitive",
                    str(input_file),
                    "-o",
                    str(output_file),
                    "--angle-tolerance",
                    "5.0",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()


class TestOrthogonalCommand:
    """Test orthogonal command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_orthogonal_help(self, runner):
        """Test orthogonal --help."""
        result = runner.invoke(cli, ["orthogonal", "--help"])
        assert result.exit_code == 0
        assert "Find orthogonal supercell" in result.output

    def test_orthogonal_basic(self, runner):
        """Test basic orthogonal command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a non-orthogonal structure
            from ase import Atoms

            cell = [[3.0, 0.0, 0.0], [1.5, 2.6, 0.0], [0.0, 0.0, 4.0]]
            structure = Atoms("C2", positions=[[0, 0, 0], [1, 1, 2]], cell=cell, pbc=True)

            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "orthogonal.vasp"

            result = runner.invoke(cli, ["orthogonal", str(input_file), "-o", str(output_file)])

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Supercell expansion" in result.output

    def test_orthogonal_with_options(self, runner):
        """Test orthogonal with various options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            from ase import Atoms

            cell = [[3.0, 0.0, 0.0], [1.5, 2.6, 0.0], [0.0, 0.0, 4.0]]
            structure = Atoms("C2", positions=[[0, 0, 0], [1, 1, 2]], cell=cell, pbc=True)

            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "orthogonal.vasp"

            result = runner.invoke(
                cli,
                [
                    "orthogonal",
                    str(input_file),
                    "-o",
                    str(output_file),
                    "--min-atoms",
                    "4",
                    "--max-atoms",
                    "100",
                    "--min-length",
                    "10.0",
                    "--force-diagonal",
                    "--force-90-degrees",
                    "--allow-orthorhombic",
                    "--angle-tolerance",
                    "0.01",
                    "--step-size",
                    "0.2",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()


class TestScaleCommand:
    """Test scale command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_scale_help(self, runner):
        """Test scale --help."""
        result = runner.invoke(cli, ["scale", "--help"])
        assert result.exit_code == 0
        assert "fractional coordinates" in result.output

    def test_scale_basic(self, runner):
        """Test basic scale command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            structure = bulk("Si", "diamond", a=5.43)
            input_file = tmpdir / "input.vasp"
            ase_write(input_file, structure, format="vasp")

            output_file = tmpdir / "scaled.vasp"

            result = runner.invoke(cli, ["scale", str(input_file), "-o", str(output_file)])

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
        assert "Error" in result.output or "Usage" in result.output

    def test_missing_required_arguments(self, runner):
        """Test missing required arguments."""
        result = runner.invoke(cli, ["convert"])
        assert result.exit_code == 2  # FriendlyCommand returns 2
        assert "Usage:" in result.output
