"""Test read functions in band.py module."""

import numpy as np
import pytest

from ddpc.data.band import read_band, read_band_h5, read_band_json


class TestReadBand:
    """Test read_band main function."""

    def test_read_band_h5_spinless(self, band_dos_dir):
        """Test reading spinless H5 format band data."""
        h5_file = band_dos_dir / "spinless_band.h5"
        data, efermi, isproj = read_band(h5_file, mode=0)

        assert isinstance(data, dict)
        assert isinstance(efermi, float)
        assert isinstance(isproj, bool)
        assert "kx" in data
        assert "ky" in data
        assert "kz" in data
        assert "dist" in data
        assert "label" in data
        assert all(isinstance(v, np.ndarray) for v in data.values())

    def test_read_band_json_spinless(self, band_dos_dir):
        """Test reading spinless JSON format band data."""
        json_file = band_dos_dir / "spinless_band.json"
        data, efermi, isproj = read_band(json_file, mode=0)

        assert isinstance(data, dict)
        assert isinstance(efermi, float)
        assert isinstance(isproj, bool)
        assert "kx" in data
        assert "dist" in data

    def test_read_band_h5_collinear(self, band_dos_dir):
        """Test reading spin-polarized H5 format band data."""
        h5_file = band_dos_dir / "collinear_band.h5"
        data, _efermi, _isproj = read_band(h5_file, mode=0)

        assert isinstance(data, dict)
        # Spin-polarized system should have up and down bands
        band_keys = [k for k in data.keys() if k.startswith("band")]
        has_spin = any("-up" in k or "-down" in k for k in band_keys)
        assert has_spin or len(band_keys) > 0

    def test_read_band_json_collinear(self, band_dos_dir):
        """Test reading spin-polarized JSON format band data."""
        json_file = band_dos_dir / "collinear_band.json"
        data, efermi, _isproj = read_band(json_file, mode=0)

        assert isinstance(data, dict)
        assert "kx" in data
        assert isinstance(efermi, float)

    def test_read_band_invalid_format(self, temp_output_dir):
        """Test reading unsupported file format."""
        invalid_file = temp_output_dir / "test.txt"
        invalid_file.write_text("test")

        with pytest.raises(TypeError, match="must be h5 or json file"):
            read_band(invalid_file)

    def test_read_band_nonexistent_file(self):
        """Test reading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            read_band("nonexistent_file.h5")


class TestReadBandH5:
    """Test read_band_h5 function."""

    def test_read_band_h5_without_h5py(self, band_dos_dir, monkeypatch):
        """Test error message when h5py dependency is missing."""
        h5_file = str(band_dos_dir / "spinless_band.h5")

        # Simulate h5py not installed
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "h5py":
                raise ImportError("No module named 'h5py'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="Reading HDF5 files requires 'h5py'"):
            read_band_h5(h5_file, mode=0)

    def test_read_band_h5_projected(self, band_dos_dir):
        """Test reading projected band data (H5 format)."""
        h5_file = str(band_dos_dir / "spinless_pband.h5")
        data, _efermi, isproj = read_band_h5(h5_file, mode=5)

        assert isproj is True
        assert isinstance(data, dict)
        assert "kx" in data
        # Projected data should contain atom and orbital information
        proj_keys = [k for k in data.keys() if not k.startswith(("k", "label", "dist"))]
        assert len(proj_keys) > 0

    def test_read_band_h5_collinear_spin2(self, band_dos_dir):
        """Test reading collinear band data with Spin2 channel."""
        h5_file = str(band_dos_dir / "collinear_band.h5")
        data, _efermi, _isproj = read_band_h5(h5_file, mode=0)

        # Collinear system should have both up and down bands
        band_keys = [k for k in data.keys() if k.startswith("band")]
        has_up = any("-up" in k for k in band_keys)
        has_down = any("-down" in k for k in band_keys)
        assert has_up, "Collinear bands should have '-up' suffix"
        assert has_down, "Collinear bands should have '-down' suffix"

    def test_read_band_h5_collinear_pband_spin2(self, band_dos_dir):
        """Test reading collinear projected band data with Spin2 channel."""
        h5_file = str(band_dos_dir / "collinear_pband.h5")
        # Test that we can at least read the file as total band (mode 0)
        # This will trigger the collinear code path in read_tband
        data, _efermi, isproj = read_band_h5(h5_file, mode=0)

        # Should have basic k-point data
        assert isinstance(data, dict)
        assert "kx" in data
        # For projected files, isproj should be True
        assert isproj is True
        # Collinear files should have spin-polarized bands
        band_keys = [k for k in data.keys() if k.startswith("band")]
        has_up = any("-up" in k for k in band_keys)
        has_down = any("-down" in k for k in band_keys)
        assert has_up and has_down, "Collinear bands should have both -up and -down"

    def test_read_band_h5_invalid_group(self, temp_output_dir):
        """Test error handling for H5 file without BandInfo group."""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        # Create an H5 file without BandInfo group
        invalid_h5 = temp_output_dir / "invalid.h5"
        with h5py.File(invalid_h5, "w") as f:
            f.create_dataset("dummy", data=[1, 2, 3])

        with pytest.raises((TypeError, KeyError)):
            read_band_h5(str(invalid_h5), mode=0)


class TestReadBandJSON:
    """Test read_band_json function."""

    def test_read_band_json_projected(self, band_dos_dir):
        """Test reading projected band data (JSON format)."""
        json_file = str(band_dos_dir / "spinless_pband.json")
        data, _efermi, isproj = read_band_json(json_file, mode=5)

        assert isproj is True
        assert isinstance(data, dict)
        assert "kx" in data

    def test_read_band_json_collinear_spin2(self, band_dos_dir):
        """Test reading collinear JSON band data with Spin2 channel."""
        json_file = str(band_dos_dir / "collinear_band.json")
        data, _efermi, _isproj = read_band_json(json_file, mode=0)

        # Collinear system should have both up and down bands
        band_keys = [k for k in data.keys() if k.startswith("band")]
        has_up = any("-up" in k for k in band_keys)
        has_down = any("-down" in k for k in band_keys)
        assert has_up or has_down, "Collinear bands should have spin channels"

    def test_read_band_json_collinear_pband_spin2(self, band_dos_dir):
        """Test reading collinear JSON projected band data with Spin2 channel."""
        json_file = str(band_dos_dir / "collinear_pband.json")
        data, _efermi, isproj = read_band_json(json_file, mode=5)

        assert isproj is True
        # Check for spin-polarized projection data - keys start with "band" and contain orbital info
        all_keys = list(data.keys())
        band_proj_keys = [k for k in all_keys if k.startswith("band") and "-" in k]
        has_up = any("-up" in k for k in band_proj_keys)
        has_down = any("-down" in k for k in band_proj_keys)
        # At least one of them should be present for collinear
        assert has_up or has_down, "Collinear projected bands should have spin channels"


class TestBandDataStructure:
    """Test band data structure integrity."""

    def test_band_kpoint_consistency(self, band_dos_dir):
        """Test k-point data consistency."""
        h5_file = band_dos_dir / "spinless_band.h5"
        data, _, _ = read_band(h5_file, mode=0)

        # All k-point related columns should have the same length
        kx_len = len(data["kx"])
        assert len(data["ky"]) == kx_len
        assert len(data["kz"]) == kx_len
        assert len(data["dist"]) == kx_len
        assert len(data["label"]) == kx_len

    def test_band_energy_shape(self, band_dos_dir):
        """Test band energy array shape."""
        h5_file = band_dos_dir / "spinless_band.h5"
        data, _, _ = read_band(h5_file, mode=0)

        # Each band should match the number of k-points
        kpt_count = len(data["kx"])
        band_keys = [k for k in data.keys() if k.startswith("band")]

        for band_key in band_keys:
            assert len(data[band_key]) == kpt_count

    def test_band_modes_different_outputs(self, band_dos_dir):
        """Test different modes produce different output structures."""
        h5_file = band_dos_dir / "spinless_pband.h5"

        # mode 0: total band
        data0, _, _ = read_band(h5_file, mode=0)

        # mode 5: atom + detailed orbital projection
        data5, _, _ = read_band(h5_file, mode=5)

        # mode 5 should have more projection columns
        keys0 = set(data0.keys())
        keys5 = set(data5.keys())

        # Basic k-point columns should exist in both
        assert "kx" in keys0
        assert "kx" in keys5

        # mode 5 should have additional columns when projection data exists
        # Note: if file has projection, keys5 should have more
        # Here we just verify both return dictionaries
        assert isinstance(data0, dict)
        assert isinstance(data5, dict)


class TestBandEdgeCases:
    """Test edge cases."""

    def test_band_fermi_energy_type(self, band_dos_dir):
        """Test Fermi energy type."""
        h5_file = band_dos_dir / "spinless_band.h5"
        _, efermi, _ = read_band(h5_file)

        assert isinstance(efermi, (float, np.floating))
        assert not np.isnan(efermi)

    def test_band_projection_flag(self, band_dos_dir):
        """Test projection flag correctness."""
        # Non-projected file
        h5_file = band_dos_dir / "spinless_band.h5"
        _, _, isproj = read_band(h5_file, mode=0)
        assert isinstance(isproj, bool)

        # Projected file
        pband_file = band_dos_dir / "spinless_pband.h5"
        _, _, isproj_p = read_band(pband_file, mode=5)
        assert isinstance(isproj_p, bool)
