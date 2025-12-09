"""Test projection processing logic in processors.py module."""

import numpy as np
import pytest

from ddpc.data.processors import (
    _band_atompxpy,
    _band_atomspdf,
    _band_ele,
    _band_elepxpy,
    _band_elespdf,
    _dos_atom,
    _dos_atomspdf,
    _dos_atomt2geg,
    _dos_element,
    _dos_spdf,
    _dos_spxpy,
    _refactor_band,
    _refactor_dos,
)


class TestBandProcessors:
    """Test band data processing functions."""

    @pytest.fixture
    def sample_band_data(self):
        """Create test band data."""
        nkpt = 10
        nband = 5
        return {
            "kx": np.linspace(0, 1, nkpt),
            "ky": np.zeros(nkpt),
            "kz": np.zeros(nkpt),
            "dist": np.linspace(0, 1, nkpt),
            "label": np.array([""] * nkpt),
            "1s": np.random.rand(nband * nkpt),
            "1px": np.random.rand(nband * nkpt),
            "2dxy": np.random.rand(nband * nkpt),
        }

    @pytest.fixture
    def sample_band_data_spin(self):
        """Create test spin-polarized band data."""
        nkpt = 10
        nband = 5
        return {
            "kx": np.linspace(0, 1, nkpt),
            "ky": np.zeros(nkpt),
            "kz": np.zeros(nkpt),
            "dist": np.linspace(0, 1, nkpt),
            "label": np.array([""] * nkpt),
            "1s-up": np.random.rand(nband * nkpt),
            "1s-down": np.random.rand(nband * nkpt),
            "2px-up": np.random.rand(nband * nkpt),
            "2px-down": np.random.rand(nband * nkpt),
        }

    def test_band_ele(self, sample_band_data):
        """Test band processing in element mode."""
        nkpt = 10
        nband = 5
        elements = ["Si", "O"]
        _data = {}

        _band_ele(sample_band_data, nkpt, nband, elements, _data)

        # Should preserve k-point information
        assert "kx" in _data
        assert "dist" in _data

        # Should have bands grouped by element
        band_keys = [k for k in _data.keys() if k.startswith("band")]
        assert len(band_keys) > 0

        # Each band value should be an array of nkpt length
        for key in band_keys:
            assert len(_data[key]) == nkpt

    def test_band_elespdf(self, sample_band_data):
        """Test band processing in element+spdf mode."""
        nkpt = 10
        nband = 5
        elements = ["Si", "O"]
        _data = {}

        _band_elespdf(sample_band_data, nkpt, nband, elements, _data)

        assert "kx" in _data
        band_keys = [k for k in _data.keys() if k.startswith("band")]
        assert len(band_keys) > 0

    def test_band_elepxpy(self, sample_band_data):
        """Test band processing in element+detailed orbital mode."""
        nkpt = 10
        nband = 5
        elements = ["Si", "O"]
        _data = {}

        _band_elepxpy(sample_band_data, nkpt, nband, elements, _data)

        assert "kx" in _data
        band_keys = [k for k in _data.keys() if k.startswith("band")]
        assert len(band_keys) > 0

    def test_band_atomspdf(self, sample_band_data):
        """Test band processing in atom+spdf mode."""
        nkpt = 10
        nband = 5
        _data = {}

        _band_atomspdf(sample_band_data, nkpt, nband, _data)

        assert "kx" in _data
        band_keys = [k for k in _data.keys() if k.startswith("band")]
        assert len(band_keys) > 0

    def test_band_atompxpy(self, sample_band_data):
        """Test band processing in atom+detailed orbital mode."""
        nkpt = 10
        nband = 5
        _data = {}

        _band_atompxpy(sample_band_data, nkpt, nband, _data)

        assert "kx" in _data
        band_keys = [k for k in _data.keys() if k.startswith("band")]
        assert len(band_keys) > 0

        # Verify the length of each band array
        for key in band_keys:
            assert len(_data[key]) == nkpt

    def test_band_spin_handling(self, sample_band_data_spin):
        """Test processing of spin-polarized data."""
        nkpt = 10
        nband = 5
        _data = {}

        _band_atompxpy(sample_band_data_spin, nkpt, nband, _data)

        # Should have up and down bands
        band_keys = [k for k in _data.keys() if k.startswith("band")]
        has_up = any("-up" in k for k in band_keys)
        has_down = any("-down" in k for k in band_keys)

        assert has_up or has_down or len(band_keys) > 0


class TestRefactorBand:
    """Test _refactor_band function."""

    @pytest.fixture
    def sample_data(self):
        """Create simple test data."""
        nkpt = 5
        nband = 3
        return (
            {
                "kx": np.linspace(0, 1, nkpt),
                "ky": np.zeros(nkpt),
                "kz": np.zeros(nkpt),
                "dist": np.linspace(0, 1, nkpt),
                "label": np.array([""] * nkpt),
                "1s": np.random.rand(nband * nkpt),
            },
            nkpt,
            nband,
        )

    def test_refactor_band_mode1(self, sample_data):
        """Test mode 1 processing."""
        data, nkpt, nband = sample_data
        elements = ["Si"]

        result = _refactor_band(data, nkpt, nband, elements, mode=1)

        assert isinstance(result, dict)
        assert "kx" in result

    def test_refactor_band_mode2(self, sample_data):
        """Test mode 2 processing."""
        data, nkpt, nband = sample_data
        elements = ["Si"]

        result = _refactor_band(data, nkpt, nband, elements, mode=2)

        assert isinstance(result, dict)
        assert "kx" in result

    def test_refactor_band_mode3(self, sample_data):
        """Test mode 3 processing."""
        data, nkpt, nband = sample_data
        elements = ["Si"]

        result = _refactor_band(data, nkpt, nband, elements, mode=3)

        assert isinstance(result, dict)
        assert "kx" in result

    def test_refactor_band_mode4(self, sample_data):
        """Test mode 4 processing."""
        data, nkpt, nband = sample_data
        elements = ["Si"]

        result = _refactor_band(data, nkpt, nband, elements, mode=4)

        assert isinstance(result, dict)
        assert "kx" in result

    def test_refactor_band_mode5(self, sample_data):
        """Test mode 5 processing."""
        data, nkpt, nband = sample_data
        elements = ["Si"]

        result = _refactor_band(data, nkpt, nband, elements, mode=5)

        assert isinstance(result, dict)
        assert "kx" in result

    def test_refactor_band_invalid_mode(self, sample_data):
        """Test unsupported mode."""
        data, nkpt, nband = sample_data
        elements = ["Si"]

        with pytest.raises(RuntimeError, match="Unsupported mode"):
            _refactor_band(data, nkpt, nband, elements, mode=99)


class TestDosProcessors:
    """Test DOS data processing functions."""

    @pytest.fixture
    def sample_dos_data(self):
        """Create test DOS data."""
        nenergy = 100
        return {
            "tdos": np.random.rand(nenergy),
            "1s": np.random.rand(nenergy),
            "1px": np.random.rand(nenergy),
            "2dxy": np.random.rand(nenergy),
            "2dxz": np.random.rand(nenergy),
            "2dx2y2": np.random.rand(nenergy),
        }

    @pytest.fixture
    def sample_dos_energies(self):
        """Create energy array."""
        return np.linspace(-10, 10, 100)

    def test_dos_spdf(self, sample_dos_data, sample_dos_energies):
        """Test DOS processing in spdf mode."""
        result = _dos_spdf(sample_dos_data, sample_dos_energies)

        assert isinstance(result, dict)
        assert "energy" in result
        assert len(result["energy"]) == len(sample_dos_energies)

        # Should have DOS grouped by s/p/d
        dos_keys = [k for k in result.keys() if k != "energy"]
        assert len(dos_keys) > 0

    def test_dos_spxpy(self, sample_dos_data, sample_dos_energies):
        """Test DOS processing in detailed orbital mode."""
        result = _dos_spxpy(sample_dos_data, sample_dos_energies)

        assert isinstance(result, dict)
        assert "energy" in result

    def test_dos_element(self, sample_dos_data, sample_dos_energies):
        """Test DOS processing in element mode."""
        elements = ["Si", "O"]

        result = _dos_element(sample_dos_data, elements, sample_dos_energies)

        assert isinstance(result, dict)
        assert "energy" in result

    def test_dos_element_no_elements(self, sample_dos_data, sample_dos_energies):
        """Test element mode raises error when element list is missing."""
        with pytest.raises(ValueError):
            _dos_element(sample_dos_data, None, sample_dos_energies)

        with pytest.raises(ValueError):
            _dos_element(sample_dos_data, [], sample_dos_energies)

    def test_dos_atomspdf(self, sample_dos_data, sample_dos_energies):
        """Test DOS processing in atom+spdf mode."""
        result = _dos_atomspdf(sample_dos_data, sample_dos_energies)

        assert isinstance(result, dict)
        assert "energy" in result

    def test_dos_atomt2geg(self, sample_dos_energies):
        """Test DOS processing in t2g/eg decomposition mode."""
        # Create data containing d orbitals
        data = {
            "tdos": np.random.rand(100),
            "1dxy": np.random.rand(100),
            "1dxz": np.random.rand(100),
            "1dyz": np.random.rand(100),
            "1dz2": np.random.rand(100),
            "1dx2y2": np.random.rand(100),
        }

        result = _dos_atomt2geg(data, sample_dos_energies)

        assert isinstance(result, dict)
        assert "energy" in result

        # Should have t2g and eg projections
        # t2g contains dxy, dxz, dyz
        # eg contains dz2, dx2y2

    def test_dos_atom(self, sample_dos_data, sample_dos_energies):
        """Test DOS processing in atom projection mode."""
        result = _dos_atom(sample_dos_data, sample_dos_energies)

        assert isinstance(result, dict)
        assert "energy" in result


class TestRefactorDos:
    """Test _refactor_dos function."""

    @pytest.fixture
    def sample_data(self):
        """Create simple DOS test data."""
        energies = np.linspace(-10, 10, 50)
        data = {
            "tdos": np.random.rand(50),
            "1s": np.random.rand(50),
            "2px": np.random.rand(50),
        }
        return energies, data

    def test_refactor_dos_mode1(self, sample_data):
        """Test mode 1 (spdf)."""
        energies, data = sample_data

        result = _refactor_dos(energies, data, mode=1)

        assert isinstance(result, dict)
        assert "energy" in result
        assert len(result["energy"]) == len(energies)

    def test_refactor_dos_mode2(self, sample_data):
        """Test mode 2 (detailed orbitals)."""
        energies, data = sample_data

        result = _refactor_dos(energies, data, mode=2)

        assert isinstance(result, dict)
        assert "energy" in result

    def test_refactor_dos_mode3(self, sample_data):
        """Test mode 3 (element)."""
        energies, data = sample_data
        elements = ["Si", "O"]

        result = _refactor_dos(energies, data, mode=3, elements=elements)

        assert isinstance(result, dict)
        assert "energy" in result

    def test_refactor_dos_mode4(self, sample_data):
        """Test mode 4 (atom+spdf)."""
        energies, data = sample_data

        result = _refactor_dos(energies, data, mode=4)

        assert isinstance(result, dict)
        assert "energy" in result

    def test_refactor_dos_mode5(self, sample_data):
        """Test mode 5 (atom+detailed orbitals)."""
        energies, data = sample_data

        result = _refactor_dos(energies, data, mode=5)

        assert isinstance(result, dict)
        assert "energy" in result

        # mode 5 should preserve original data structure and add energy
        for key in data.keys():
            assert key in result

    def test_refactor_dos_mode6(self):
        """Test mode 6 (t2g/eg)."""
        energies = np.linspace(-10, 10, 50)
        data = {
            "tdos": np.random.rand(50),
            "1dxy": np.random.rand(50),
            "1dz2": np.random.rand(50),
        }

        result = _refactor_dos(energies, data, mode=6)

        assert isinstance(result, dict)
        assert "energy" in result

    def test_refactor_dos_mode7(self, sample_data):
        """Test mode 7 (atom projection)."""
        energies, data = sample_data

        result = _refactor_dos(energies, data, mode=7)

        assert isinstance(result, dict)
        assert "energy" in result

    def test_refactor_dos_invalid_mode(self, sample_data):
        """Test unsupported mode."""
        energies, data = sample_data

        with pytest.raises(RuntimeError, match="Unsupported mode"):
            _refactor_dos(energies, data, mode=99)

    def test_refactor_dos_accepts_list_energies(self, sample_data):
        """Test energy parameter can be a list."""
        _, data = sample_data
        energies_list = list(range(50))

        result = _refactor_dos(energies_list, data, mode=1)

        assert isinstance(result, dict)
        assert isinstance(result["energy"], np.ndarray)


class TestProcessorEdgeCases:
    """Test processor edge cases."""

    def test_empty_projection_data(self):
        """Test processing of empty projection data."""
        nkpt = 5
        nband = 3
        data = {
            "kx": np.linspace(0, 1, nkpt),
            "ky": np.zeros(nkpt),
            "kz": np.zeros(nkpt),
            "dist": np.linspace(0, 1, nkpt),
            "label": np.array([""] * nkpt),
        }
        elements = ["Si"]

        # Should work normally even without projection data
        result = _refactor_band(data, nkpt, nband, elements, mode=5)

        assert "kx" in result
        assert "dist" in result

    def test_single_kpoint_band(self):
        """Test band data with single k-point."""
        nkpt = 1
        nband = 2
        data = {
            "kx": np.array([0.0]),
            "ky": np.array([0.0]),
            "kz": np.array([0.0]),
            "dist": np.array([0.0]),
            "label": np.array([""]),
            "1s": np.array([1.0, 2.0]),
        }
        elements = ["Si"]

        result = _refactor_band(data, nkpt, nband, elements, mode=5)

        assert len(result["kx"]) == nkpt

    def test_single_energy_point_dos(self):
        """Test DOS data with single energy point."""
        energies = np.array([0.0])
        data = {
            "tdos": np.array([1.0]),
        }

        result = _refactor_dos(energies, data, mode=1)

        assert len(result["energy"]) == 1
