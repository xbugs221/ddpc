"""Export data to various formats."""

from pathlib import Path

import numpy as np


def to_csv(
    data: dict[str, np.ndarray],
    path: str | Path,
    *,
    delimiter: str = ",",
    header: bool = True,
) -> None:
    """Export data dict to CSV file.

    Args:
        data: Dict with numpy arrays of same length
        path: Output CSV file path
        delimiter: Column separator (default: ",")
        header: Include column names (default: True)

    Raises
    ------
        ValueError: If arrays have inconsistent shapes
        IOError: If cannot write to path
    """
    path = Path(path)
    if not data:
        raise ValueError("Empty data dict")

    # Flatten data structure for CSV
    flat_data = {}
    for key, arr in data.items():
        if arr.ndim == 1:
            flat_data[key] = arr
        elif arr.ndim == 2:
            # Expand 2D arrays into multiple columns
            for j in range(arr.shape[1]):
                flat_data[f"{key}{j}"] = arr[:, j]
        else:
            raise ValueError(f"Cannot export {arr.ndim}D array '{key}' to CSV")

    # Check consistency
    arrays = list(flat_data.values())
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Inconsistent array lengths: {dict(zip(flat_data.keys(), lengths, strict=False))}"
        )

    # Build CSV lines
    lines = []
    if header:
        lines.append(delimiter.join(flat_data.keys()))

    n_rows = lengths[0]
    for i in range(n_rows):
        row = [str(arr[i]) for arr in arrays]
        lines.append(delimiter.join(row))

    path.write_text("\n".join(lines), encoding="utf-8")


def to_npz(
    data: dict[str, np.ndarray],
    path: str | Path,
    *,
    compressed: bool = True,
) -> None:
    """Export data dict to NPZ file.

    Args:
        data: Dict with numpy arrays
        path: Output NPZ file path
        compressed: Use compression (default: True)

    Raises
    ------
        IOError: If cannot write to path
    """
    path = Path(path)
    if compressed:
        np.savez_compressed(path, **data)
    else:
        np.savez(path, **data)
