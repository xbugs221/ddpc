"""Integration tests: Verify complete workflow for all export formats."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ddpc.data.export import to_csv, to_npz


class TestExportFormats:
    """Integration tests for all export formats."""

    @pytest.fixture
    def sample_data(self):
        """Create test data."""
        return {
            "kpoint": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            "energy": np.array([1.0, 2.0, 3.0]),
            "weight": np.array([1.0, 1.0, 1.0]),
        }

    def test_csv_export_roundtrip(self, sample_data):
        """Test complete CSV export and read workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test.csv"

            # Export
            to_csv(sample_data, csv_file)
            assert csv_file.exists()

            # Read and verify
            import csv

            with open(csv_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3
            assert "energy" in rows[0]
            assert float(rows[0]["energy"]) == pytest.approx(1.0)

    def test_npz_export_roundtrip(self, sample_data):
        """Test complete NPZ export and read workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_file = Path(tmpdir) / "test.npz"

            # Export
            to_npz(sample_data, npz_file)
            assert npz_file.exists()

            # Read and verify
            loaded = np.load(npz_file)
            np.testing.assert_array_equal(loaded["energy"], sample_data["energy"])
            np.testing.assert_array_equal(loaded["kpoint"], sample_data["kpoint"])

    def test_all_formats_consistency(self, sample_data):
        """Test consistency across all formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Export all formats
            to_csv(sample_data, tmpdir / "data.csv")
            to_npz(sample_data, tmpdir / "data.npz")

            # Verify all files exist
            assert (tmpdir / "data.csv").exists()
            assert (tmpdir / "data.npz").exists()

            # Verify NPZ format data is correct
            loaded_npz = np.load(tmpdir / "data.npz")
            np.testing.assert_array_equal(loaded_npz["energy"], sample_data["energy"])

    def test_export_with_projections(self):
        """Test export with projection data."""
        data_with_proj = {
            "kpoint": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            "energy": np.array([1.0, 2.0]),
            "weight": np.array([1.0, 1.0]),
            "proj_s": np.array([0.1, 0.2]),
            "proj_p": np.array([0.3, 0.4]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # ExportNPZ
            to_npz(data_with_proj, tmpdir / "proj.npz")
            loaded = np.load(tmpdir / "proj.npz")

            assert "proj_s" in loaded
            assert "proj_p" in loaded
            np.testing.assert_array_equal(loaded["proj_s"], data_with_proj["proj_s"])

    def test_export_empty_data(self):
        """Test edge case of exporting empty data."""
        empty_data = {
            "kpoint": np.array([]).reshape(0, 3),
            "energy": np.array([]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # NPZ should handle empty data
            to_npz(empty_data, tmpdir / "empty.npz")
            loaded = np.load(tmpdir / "empty.npz")
            assert len(loaded["energy"]) == 0
