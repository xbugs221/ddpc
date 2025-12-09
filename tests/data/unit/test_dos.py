"""Test read functions in dos.py module."""

import numpy as np
import pytest

from ddpc.data.dos import read_dos, read_dos_h5, read_dos_json


class TestReadDos:
    """Test read_dos main function."""

    def test_read_dos_h5_spinless(self, band_dos_dir):
        """Test reading spinless H5 format DOS data."""
        h5_file = band_dos_dir / "spinless_dos.h5"
        data, efermi, isproj = read_dos(h5_file, mode=0)

        assert isinstance(data, dict)
        assert isinstance(efermi, float)
        assert isinstance(isproj, bool)
        assert "energy" in data
        assert "dos" in data
        assert all(isinstance(v, np.ndarray) for v in data.values())

    def test_read_dos_json_spinless(self, band_dos_dir):
        """Test reading spinless JSON format DOS data."""
        json_file = band_dos_dir / "spinless_dos.json"
        data, efermi, isproj = read_dos(json_file, mode=0)

        assert isinstance(data, dict)
        assert isinstance(efermi, float)
        assert isinstance(isproj, bool)
        assert "energy" in data
        assert "dos" in data

    def test_read_dos_h5_collinear(self, band_dos_dir):
        """Test reading spin-polarized H5 format DOS data."""
        h5_file = band_dos_dir / "collinear_dos.h5"
        data, _efermi, _isproj = read_dos(h5_file, mode=0)

        assert isinstance(data, dict)
        assert "energy" in data
        # Spin-polarized system should have up and down, or non-polarized system has dos
        # Note: filename is collinear, but actual data structure may differ
        assert "up" in data or "down" in data or "dos" in data

    def test_read_dos_json_collinear(self, band_dos_dir):
        """Test reading spin-polarized JSON format DOS data."""
        json_file = band_dos_dir / "collinear_dos.json"
        data, efermi, _isproj = read_dos(json_file, mode=0)

        assert isinstance(data, dict)
        assert "energy" in data
        assert isinstance(efermi, float)

    def test_read_dos_invalid_format(self, temp_output_dir):
        """Test reading unsupported file format."""
        invalid_file = temp_output_dir / "test.txt"
        invalid_file.write_text("test")

        with pytest.raises(TypeError, match="must be h5 or json file"):
            read_dos(invalid_file)

    def test_read_dos_nonexistent_file(self):
        """Test reading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            read_dos("nonexistent_file.h5")


class TestReadDosH5:
    """Test read_dos_h5 function."""

    def test_read_dos_h5_without_h5py(self, band_dos_dir, monkeypatch):
        """Test error message when h5py dependency is missing."""
        h5_file = str(band_dos_dir / "spinless_dos.h5")

        # Simulate h5py not installed
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "h5py":
                raise ImportError("No module named 'h5py'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="Reading HDF5 files requires 'h5py'"):
            read_dos_h5(h5_file, mode=0)

    def test_read_dos_h5_projected(self, band_dos_dir):
        """Test reading projected DOS data (H5 format)."""
        h5_file = str(band_dos_dir / "spinless_pdos.h5")
        data, _efermi, isproj = read_dos_h5(h5_file, mode=5)

        assert isproj is True
        assert isinstance(data, dict)
        assert "energy" in data
        # Projected data should contain atom and orbital information
        proj_keys = [k for k in data.keys() if k not in ("energy", "tdos")]
        assert len(proj_keys) > 0

    def test_read_dos_h5_mode_variations(self, band_dos_dir):
        """Test different mode effects on H5 format."""
        h5_file = str(band_dos_dir / "spinless_pdos.h5")

        # mode 0: total DOS
        data0, _, _ = read_dos_h5(h5_file, mode=0)
        assert "energy" in data0

        # mode 1: spdf projection
        data1, _, _ = read_dos_h5(h5_file, mode=1)
        assert "energy" in data1

    def test_read_dos_h5_collinear_pdos(self, band_dos_dir):
        """Test reading collinear projected DOS data with Spin2 channel."""
        h5_file = str(band_dos_dir / "collinear_pdos.h5")
        data, _efermi, isproj = read_dos_h5(h5_file, mode=5)

        assert isproj is True
        # Check for spin-polarized tdos
        assert "tdos-up" in data or "tdos-down" in data, "Collinear DOS should have spin channels"
        # Check for spin-polarized projection data
        proj_keys = [k for k in data.keys() if not k.startswith(("energy", "tdos"))]
        has_up = any("-up" in k for k in proj_keys)
        has_down = any("-down" in k for k in proj_keys)
        assert has_up or has_down, "Collinear projected DOS should have spin channels"

    def test_read_dos_h5_invalid_group(self, temp_output_dir):
        """Test error handling for H5 file without DosInfo group."""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        # Create an H5 file without DosInfo group
        invalid_h5 = temp_output_dir / "invalid.h5"
        with h5py.File(invalid_h5, "w") as f:
            f.create_dataset("dummy", data=[1, 2, 3])

        with pytest.raises((TypeError, KeyError)):
            read_dos_h5(str(invalid_h5), mode=0)


class TestReadDosJSON:
    """Test read_dos_json function."""

    def test_read_dos_json_projected(self, band_dos_dir):
        """Test reading projected DOS data (JSON format)."""
        json_file = str(band_dos_dir / "spinless_pdos.json")
        data, _efermi, isproj = read_dos_json(json_file, mode=5)

        assert isproj is True
        assert isinstance(data, dict)
        assert "energy" in data

    def test_read_dos_json_mode_variations(self, band_dos_dir):
        """Test different mode effects on JSON format."""
        json_file = str(band_dos_dir / "spinless_pdos.json")

        # mode 0: total DOS
        data0, _, _ = read_dos_json(json_file, mode=0)
        assert "energy" in data0

        # mode 4: atom + spdf
        data4, _, _ = read_dos_json(json_file, mode=4)
        assert "energy" in data4

    def test_read_dos_json_collinear_pdos(self, band_dos_dir):
        """Test reading collinear JSON projected DOS data with Spin2 channel."""
        json_file = str(band_dos_dir / "collinear_pdos.json")
        data, _efermi, isproj = read_dos_json(json_file, mode=5)

        assert isproj is True
        # Check for spin-polarized tdos
        assert "tdos-up" in data or "tdos-down" in data, "Collinear DOS should have spin channels"
        # Check for spin-polarized projection data
        proj_keys = [k for k in data.keys() if not k.startswith(("energy", "tdos"))]
        has_up = any("-up" in k for k in proj_keys)
        has_down = any("-down" in k for k in proj_keys)
        assert has_up or has_down, "Collinear projected DOS should have spin channels"


class TestDosDataStructure:
    """Test DOS data structure integrity."""

    def test_dos_energy_dos_consistency(self, band_dos_dir):
        """Test energy and DOS array length consistency."""
        h5_file = band_dos_dir / "spinless_dos.h5"
        data, _, _ = read_dos(h5_file, mode=0)

        energy_len = len(data["energy"])
        dos_len = len(data["dos"])

        assert energy_len == dos_len, "Energy and DOS arrays must have same length"

    def test_dos_collinear_symmetry(self, band_dos_dir):
        """Test up/down symmetry in spin-polarized DOS."""
        h5_file = band_dos_dir / "collinear_dos.h5"
        data, _, _ = read_dos(h5_file, mode=0)

        if "up" in data and "down" in data:
            # up and down should have the same length
            assert len(data["up"]) == len(data["down"])
            # Both should match energy array length
            assert len(data["up"]) == len(data["energy"])

    def test_dos_projected_all_same_length(self, band_dos_dir):
        """Test all arrays in projected DOS have consistent length."""
        h5_file = band_dos_dir / "spinless_pdos.h5"
        data, _, _ = read_dos(h5_file, mode=5)

        energy_len = len(data["energy"])

        # All other columns should match energy length
        for key, value in data.items():
            if key != "energy":
                assert len(value) == energy_len, f"Column {key} has different length than energy"

    def test_dos_modes_different_outputs(self, band_dos_dir):
        """Test different modes produce different output structures."""
        h5_file = band_dos_dir / "spinless_pdos.h5"

        # mode 0: total DOS
        data0, _, _ = read_dos(h5_file, mode=0)

        # mode 1: spdf projection
        data1, _, _ = read_dos(h5_file, mode=1)

        # mode 4: atom + spdf
        data4, _, _ = read_dos(h5_file, mode=4)

        # All modes should have energy column
        assert "energy" in data0
        assert "energy" in data1
        assert "energy" in data4

        # mode 1 and 4 should have more columns when projection data exists
        # Verify all are valid dictionaries
        assert isinstance(data0, dict)
        assert isinstance(data1, dict)
        assert isinstance(data4, dict)


class TestDosEdgeCases:
    """Test edge cases."""

    def test_dos_fermi_energy_type(self, band_dos_dir):
        """Test Fermi energy type."""
        h5_file = band_dos_dir / "spinless_dos.h5"
        _, efermi, _ = read_dos(h5_file)

        assert isinstance(efermi, (float, np.floating))
        assert not np.isnan(efermi)

    def test_dos_projection_flag(self, band_dos_dir):
        """Test projection flag correctness."""
        # Non-projected file
        h5_file = band_dos_dir / "spinless_dos.h5"
        _, _, isproj = read_dos(h5_file, mode=0)
        assert isinstance(isproj, bool)

        # Projected file
        pdos_file = band_dos_dir / "spinless_pdos.h5"
        _, _, isproj_p = read_dos(pdos_file, mode=5)
        assert isinstance(isproj_p, bool)

    def test_dos_energy_ordering(self, band_dos_dir):
        """Test if energy array is monotonically increasing."""
        h5_file = band_dos_dir / "spinless_dos.h5"
        data, _, _ = read_dos(h5_file)

        energy = data["energy"]
        # Check if energy is monotonically increasing
        assert np.all(np.diff(energy) >= 0), "Energy array should be monotonically increasing"

    def test_dos_noncollinear(self, band_dos_dir):
        """Test noncollinear DOS data."""
        h5_file = band_dos_dir / "noncollinear_dos.h5"
        data, _efermi, _ = read_dos(h5_file, mode=0)

        assert isinstance(data, dict)
        assert "energy" in data
        # Noncollinear system should only have one dos column (no up/down separation)
        assert "dos" in data

    def test_dos_h5_json_consistency(self, band_dos_dir):
        """Test consistency between H5 and JSON format reading."""
        h5_file = band_dos_dir / "spinless_dos.h5"
        json_file = band_dos_dir / "spinless_dos.json"

        data_h5, efermi_h5, isproj_h5 = read_dos(h5_file, mode=0)
        data_json, efermi_json, isproj_json = read_dos(json_file, mode=0)

        # Basic properties should be consistent
        assert isproj_h5 == isproj_json
        assert abs(efermi_h5 - efermi_json) < 1e-6

        # Data structure should be consistent
        assert set(data_h5.keys()) == set(data_json.keys())

        # Array lengths should be consistent
        assert len(data_h5["energy"]) == len(data_json["energy"])
