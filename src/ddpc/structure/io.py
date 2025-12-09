"""Crystal structure I/O functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.atoms import Atoms


def read_structure(file: str | Path, format: str | None = None) -> Atoms:
    """Read crystal structure from file.

    Args:
        file: Path to structure file
        format: File format (auto-detected if None)

    Returns
    -------
        ase.Atoms object with structure

    Raises
    ------
        ImportError: If required parser not installed
        FileNotFoundError: If file does not exist
        ValueError: If unsupported format or corrupt file
    """
    from ase.io import read

    from ddpc.structure.readers import dspaw_as, rescu_xyz

    fn = str(file)
    if format == "dspaw" or fn.endswith(".as"):
        return dspaw_as.read(fn)
    if format == "rescu":
        return rescu_xyz.read(fn)
    if fn.endswith(".xyz"):
        # Try standard XYZ first, fallback to RESCU format
        try:
            return read(fn, format="xyz")
        except Exception:
            return rescu_xyz.read(fn)
    return read(fn, format=format)


def write_structure(
    file: str | Path,
    atoms: Atoms,
    format: str | None = None,
    **kwargs,
) -> None:
    """Write crystal structure to file.

    Args:
        file: Output file path
        atoms: Structure to write
        format: Output format (inferred from extension if None)
        **kwargs: Format-specific options

    Raises
    ------
        ImportError: If required writer not installed
        ValueError: If unsupported format
        IOError: If cannot write to path
    """
    from ase.io import write

    from ddpc.structure.writers import dspaw_as, rescu_xyz

    fn = str(file)
    if format == "dspaw" or fn.endswith(".as"):
        dspaw_as.write(fn, atoms)
        return
    if format == "rescu" or (format == "xyz" and "atom_fix" in atoms.info):
        rescu_xyz.write(fn, atoms)
        return

    write(fn, atoms, format=format, **kwargs)
