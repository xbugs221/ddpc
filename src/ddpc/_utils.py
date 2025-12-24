"""Common utility functions shared across ddpc modules."""

import re
from pathlib import Path
from typing import Union


def absf(p: Union[str, Path]) -> Path:
    """Return absolute path of a file or directory."""
    return Path(p).resolve()


def remove_comments(p: Union[str, Path], comment: str = "#") -> list:
    """Remove comments from a text file and return non-empty lines."""
    lines = []
    with open(p, encoding="utf-8") as file:
        while True:
            line = file.readline()
            if line:
                line = re.sub(comment + r".*$", "", line)
                line = line.strip()
                if line:
                    lines.append(line)
            else:
                break
    return lines
