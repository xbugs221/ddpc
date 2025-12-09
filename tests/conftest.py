from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir():
    return Path("tests/data/raw")


@pytest.fixture(scope="session")
def band_dos_dir():
    """Test data directory - band and dos data files."""
    return Path("tests/data/raw")


@pytest.fixture(scope="session")
def structures_dir():
    """Test data directory - structure data files."""
    return Path("tests/structure/raw")


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    return tmp_path


PROJ_TYPES = ["", "p"]
SPIN_TYPES = ["collinear", "noncollinear", "spinless"]
FILE_EXTS = ["h5", "json"]
DATA_TYPES = ["band", "dos"]


def _create_fixture_func(file_path_str: str):
    @pytest.fixture(scope="session")
    def path_fixture(data_dir: Path) -> Path:
        return data_dir / file_path_str

    return path_fixture


# Generate all possible file path strings
# This list will be used as parameters for a single fixture
ALL_FILE_PATHS = []
for proj in PROJ_TYPES:
    for spin in SPIN_TYPES:
        for ext in FILE_EXTS:
            for dt in DATA_TYPES:
                file_name = f"{spin}_{proj}{dt}.{ext}"
                ALL_FILE_PATHS.append(file_name)


@pytest.fixture(scope="session", params=ALL_FILE_PATHS)
def parametrized_data_file_path(data_dir: Path, request) -> Path:
    """Yield different file paths based on test parameter combinations.

    Generates paths from PROJ_TYPES, SPIN_TYPES, FILE_EXTS, and DATA_TYPES.
    """
    file_path_str = request.param
    full_path = data_dir / file_path_str
    print(f"\nPreparing data file path fixture: {full_path}")
    # In a real scenario, you might check if the file exists here
    # assert full_path.exists(), f"Test data file not found: {full_path}"
    yield full_path
    # Optional: cleanup or print after test
    print(f"Finished with data file path: {full_path}")
