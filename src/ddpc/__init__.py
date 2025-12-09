"""DDPC."""


def _has_data_deps():
    try:
        import h5py
        import numpy as np

        return True
    except ImportError:
        return False


def _has_structure_deps():
    try:
        import ase
        import spglib

        return True
    except ImportError:
        return False


def __getattr__(name):
    if name in ("read_band", "read_dos", "to_csv", "to_npz"):
        if not _has_data_deps():
            raise ImportError(f"Install ddpc[data] before using '{name}'")
        from ddpc import data

        return getattr(data, name)

    if name in (
        "read_structure",
        "write_structure",
        "find_primitive",
        "find_orthogonal",
        "scale_positions",
        "get_symmetry",
    ):
        if not _has_structure_deps():
            raise ImportError(f"Install ddpc[structure] before using '{name}'")
        from ddpc import structure

        return getattr(structure, name)

    raise AttributeError(f"'ddpc' has no '{name}'")
