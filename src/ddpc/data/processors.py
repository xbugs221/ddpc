"""Projection mode processing for band and DOS data, internal use only."""

from typing import List, Union

import numpy as np

from ddpc.data.utils import (
    _get_ao_spin,
    _inplace_update_data,
    _split_atomindex_orbital,
)


# Band processing functions
def _band_ele(data: dict, nkpt: int, nband: int, elements: List[str], _data: dict) -> None:
    """Process band data in element-resolved mode (mode 1)."""
    for k, v in data.items():
        if k.startswith(("k", "label", "dist")):
            _data[k] = v
        else:
            cont = np.asarray(v).reshape(nband, nkpt, order="F")
            ao, updown = _get_ao_spin(k)
            a, _ = _split_atomindex_orbital(ao)
            for b in range(nband):
                if updown:
                    key = f"band{b + 1}-{elements[a - 1]}-{updown}"
                else:
                    key = f"band{b + 1}-{elements[a - 1]}"
                _inplace_update_data(_data, key, cont[b])


def _band_elespdf(data: dict, nkpt: int, nband: int, elements: List[str], _data: dict) -> None:
    """Process band data in element + spdf mode (mode 2)."""
    for k, v in data.items():
        if k.startswith(("k", "label", "dist")):
            _data[k] = v
        else:
            cont = np.asarray(v).reshape(nband, nkpt, order="F")
            ao, updown = _get_ao_spin(k)
            a, o = _split_atomindex_orbital(ao)
            for b in range(nband):
                if updown:
                    key = f"band{b + 1}-{elements[a - 1]}-{o[0]}-{updown}"
                else:
                    key = f"band{b + 1}-{elements[a - 1]}-{o[0]}"
                _inplace_update_data(_data, key, cont[b])


def _band_elepxpy(data: dict, nkpt: int, nband: int, elements: List[str], _data: dict) -> None:
    """Process band data in element + detailed orbital mode (mode 3)."""
    for k, v in data.items():
        if k.startswith(("k", "label", "dist")):
            _data[k] = v
        else:
            cont = np.asarray(v).reshape(nband, nkpt, order="F")
            ao, updown = _get_ao_spin(k)
            a, o = _split_atomindex_orbital(ao)
            for b in range(nband):
                if updown:
                    key = f"band{b + 1}-{elements[a - 1]}-{o}-{updown}"
                else:
                    key = f"band{b + 1}-{elements[a - 1]}-{o}"
                _inplace_update_data(_data, key, cont[b])


def _band_atomspdf(data: dict, nkpt: int, nband: int, _data: dict) -> None:
    """Process band data in atom + spdf mode (mode 4)."""
    for k, v in data.items():
        if k.startswith(("k", "label", "dist")):
            _data[k] = v
        else:
            cont = np.asarray(v).reshape(nband, nkpt, order="F")
            ao, updown = _get_ao_spin(k)
            a, o = _split_atomindex_orbital(ao)
            for b in range(nband):
                if updown:
                    key = f"band{b + 1}-{a}-{o[0]}-{updown}"
                else:
                    key = f"band{b + 1}-{a}-{o[0]}"
                _inplace_update_data(_data, key, cont[b])


def _band_atompxpy(data: dict, nkpt: int, nband: int, _data: dict) -> None:
    """Process band data in atom + detailed orbital mode (mode 5)."""
    for k, v in data.items():
        if k.startswith(("k", "label", "dist")):
            _data[k] = v
        else:
            cont = np.asarray(v).reshape(nband, nkpt, order="F")
            ao, updown = _get_ao_spin(k)
            a, o = _split_atomindex_orbital(ao)
            for b in range(nband):
                if updown:
                    key = f"band{b + 1}-{a}-{o}-{updown}"
                else:
                    key = f"band{b + 1}-{a}-{o}"
                _inplace_update_data(_data, key, cont[b])


def _refactor_band(data: dict, nkpt: int, nband: int, elements: List[str], mode: int) -> dict:
    """Refactor band data based on projection mode.

    Args:
        data: Raw band data dict
        nkpt: Number of k-points
        nband: Number of bands
        elements: List of element symbols
        mode: Projection mode (1-5)

    Returns
    -------
        Processed band data dict

    Raises
    ------
        RuntimeError: If mode is not supported
    """
    _data: dict = {}
    if mode == 1:
        _band_ele(data, nkpt, nband, elements, _data)
    elif mode == 2:
        _band_elespdf(data, nkpt, nband, elements, _data)
    elif mode == 3:
        _band_elepxpy(data, nkpt, nband, elements, _data)
    elif mode == 4:
        _band_atomspdf(data, nkpt, nband, _data)
    elif mode == 5:
        _band_atompxpy(data, nkpt, nband, _data)
    else:
        print(f"{mode=} not supported yet")
        raise RuntimeError(f"Unsupported mode: {mode}")

    return _data


# DOS processing functions
def _dos_spdf(data: dict, energies: np.ndarray) -> dict:
    """Process DOS data in spdf mode (mode 1)."""
    _data = {"energy": energies}
    for k, v in data.items():
        if k.startswith("tdos"):
            _data[k] = v
            continue
        ao, updown = _get_ao_spin(k)
        _, o = _split_atomindex_orbital(ao)
        if updown:
            key = f"{o[0]}-{updown}"
        else:
            key = f"{o[0]}"
        _inplace_update_data(_data, key, v)
    return _data


def _dos_spxpy(data: dict, energies: np.ndarray) -> dict:
    """Process DOS data in detailed orbital mode (mode 2)."""
    _data = {"energy": energies}
    for k, v in data.items():
        if k.startswith("tdos"):
            _data[k] = v
            continue
        ao, updown = _get_ao_spin(k)
        _, o = _split_atomindex_orbital(ao)
        if updown:
            key = f"{o}-{updown}"
        else:
            key = f"{o}"
        _inplace_update_data(_data, key, v)
    return _data


def _dos_element(data: dict, elements: Union[List[str], None], energies: np.ndarray) -> dict:
    """Process DOS data in element-resolved mode (mode 3)."""
    if not elements:
        raise ValueError(f"{elements=}")
    _data = {"energy": energies}
    for k, v in data.items():
        if k.startswith("tdos"):
            _data[k] = v
            continue
        ao, updown = _get_ao_spin(k)
        a, _ = _split_atomindex_orbital(ao)
        if updown:
            key = f"{elements[a - 1]}-{updown}"
        else:
            key = f"{elements[a - 1]}"
        _inplace_update_data(_data, key, v)
    return _data


def _dos_atomspdf(data: dict, energies: np.ndarray) -> dict:
    """Process DOS data in atom + spdf mode (mode 4)."""
    _data = {"energy": energies}
    for k, v in data.items():
        if k.startswith("tdos"):
            _data[k] = v
            continue
        ao, updown = _get_ao_spin(k)
        a, o = _split_atomindex_orbital(ao)
        if updown:
            key = f"{a}{o[0]}-{updown}"
        else:
            key = f"{a}{o[0]}"
        _inplace_update_data(_data, key, v)
    return _data


def _dos_atomt2geg(data: dict, energies: np.ndarray) -> dict:
    """Process DOS data in atom + t2g/eg mode (mode 6)."""
    _data = {"energy": energies}
    for k, v in data.items():
        if k.startswith("tdos"):
            _data[k] = v
            continue
        ao, updown = _get_ao_spin(k)
        a, o = _split_atomindex_orbital(ao)
        if o in ["dxy", "dxz", "dyz"]:
            if updown:
                key = f"{a}t2g-{updown}"
            else:
                key = f"{a}t2g"
            _inplace_update_data(_data, key, v)
        elif o in ["dz2", "dx2y2"]:
            if updown:
                key = f"{a}eg-{updown}"
            else:
                key = f"{a}eg"
            _inplace_update_data(_data, key, v)

    return _data


def _dos_atom(data: dict, energies: np.ndarray) -> dict:
    """Process DOS data in atom-projected mode (mode 7) - sum all orbitals per atom."""
    _data = {"energy": energies}
    for k, v in data.items():
        if k.startswith("tdos"):
            _data[k] = v
            continue
        ao, updown = _get_ao_spin(k)
        a, _ = _split_atomindex_orbital(ao)
        if updown:
            key = f"{a}-{updown}"
        else:
            key = f"{a}"
        _inplace_update_data(_data, key, v)
    return _data


def _refactor_dos(
    energies: Union[list, np.ndarray],
    data: dict,
    mode: int,
    elements: Union[List[str], None] = None,
) -> dict:
    """Refactor DOS data based on projection mode.

    Args:
        energies: Energy points
        data: Raw DOS data dict
        mode: Projection mode (1-7)
        elements: List of element symbols (required for mode 3)

    Returns
    -------
        Processed DOS data dict

    Raises
    ------
        RuntimeError: If mode is not supported
    """
    energies = np.asarray(energies)
    if mode == 1:  # spdf
        _data = _dos_spdf(data, energies)
    elif mode == 2:  # spxpy...
        _data = _dos_spxpy(data, energies)
    elif mode == 3:  # element
        _data = _dos_element(data, elements, energies)
    elif mode == 4:  # atom+spdf
        _data = _dos_atomspdf(data, energies)
    elif mode == 5:  # atom+spxpy...
        _data = {"energy": energies} | {key: dataset[:] for key, dataset in data.items()}
    elif mode == 6:  # atom+t2g/eg
        _data = _dos_atomt2geg(data, energies)
    elif mode == 7:  # atom-projected (sum all orbitals per atom)
        _data = _dos_atom(data, energies)
    else:
        print(f"{mode=} not supported yet")
        raise RuntimeError(f"Unsupported mode: {mode}")

    return _data
