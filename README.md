# DDPC - DFT Data Processing Core

[![codecov](https://codecov.io/gh/DeeliN221/ddpc/graph/badge.svg?token=7HLO183PVZ)](https://codecov.io/gh/DeeliN221/ddpc)
[![PyPI version](https://badge.fury.io/py/ddpc.svg)](https://badge.fury.io/py/ddpc)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DDPC is a Python library for processing and analyzing density functional theory (DFT) calculation data. It provides a unified interface for reading, writing, and manipulating crystal structures, electronic band structures, and density of states data from various DFT codes including VASP, DS-PAW, and RESCU.

**Status**: Early stage development. API may change.

[API Documentation](https://ddpc.readthedocs.io/en/latest/index.html)

## Features

- Universal structure I/O for multiple formats (VASP, DS-PAW, RESCU, CIF, XYZ)
- Electronic structure data processing (band structure, density of states)
- Structure manipulation utilities (primitive cell, orthogonal supercell)
- Full type annotations
- Command-line interface

## Quick Start

### Installation

```bash
pip install "ddpc[full]"
```

### Basic Usage

#### Structure I/O

```python
from ddpc import read_structure, write_structure

# Read crystal structure from various formats
atoms = read_structure("input.vasp")  # VASP POSCAR
atoms = read_structure("structure.as")  # DS-PAW format
atoms = read_structure("crystal.cif")   # CIF format

# Write to different formats
write_structure("output.vasp", atoms)
write_structure("structure.xyz", atoms)
```

#### Electronic Structure Analysis

Currently only support DS-PAW output hdf5/json format, will support others in the future.

```python
from ddpc import read_band, read_dos

# Read band structure data
df_band, fermi_energy, has_projections = read_band("band.h5", mode=5)
print(f"Fermi energy: {fermi_energy:.3f} eV")

# Read density of states
df_dos, fermi_energy, has_projections = read_dos("dos.json", mode=1)
```

#### Structure Utilities

```python
from ddpc import find_primitive, find_orthogonal, scale_positions
from ddpc import read_structure, write_structure

# Find primitive cell
atoms = read_structure("input.vasp")
prim = find_primitive(atoms, symprec=1e-5, angle_tolerance=-1.0)
write_structure("primitive.vasp", prim, format="vasp")

# Create orthogonal supercell
orth = find_orthogonal(atoms, min_length=15.0, max_length=20.0)
write_structure("ortho.vasp", orth, format="vasp")

# Convert to fractional coordinates
scaled = scale_positions(atoms)
write_structure("scaled.vasp", scaled, format="vasp", direct=True)
```

## Command-Line Interface

### Structure Commands

```bash
# Convert structure formats
ddpc structure convert input.vasp output.cif

# Show structure information
ddpc structure info input.vasp

# Find primitive cell
ddpc structure primitive input.vasp -o primitive.vasp

# Find orthogonal supercell
ddpc structure orthogonal input.vasp -o ortho.vasp

# Convert to fractional coordinates
ddpc structure scale input.vasp -o scaled.vasp
```

### Data Commands

```bash
# Read band structure data
ddpc data band read band.h5 -o band_data.csv

# Show band structure info
ddpc data band info band.h5

# Read DOS data
ddpc data dos read dos.json -o dos_data.csv

# Show DOS info
ddpc data dos info dos.json
```

## Supported Formats

### Crystal Structures

- **VASP**: POSCAR/CONTCAR files
- **DS-PAW**: Custom .as format with constraints and magnetism
- **RESCU**: Extended .xyz format with magnetic moments
- **Standard formats**: CIF, XYZ, and other ASE-supported formats

### Electronic Structure Data

- **HDF5 files**: Band structure and DOS data from DFT calculations
- **JSON files**: Alternative format for smaller datasets
- **Projected data**: Orbital-resolved band structures and DOS

## Advanced Features

### Constraint Handling

DDPC preserves and processes atomic and lattice constraints from specialized DFT codes:

```python
# DS-PAW format with constraints
atoms = read_structure("constrained.as")
print(atoms.info)  # Shows constraint information
```

### Magnetic Systems

Support for both collinear and non-collinear magnetic systems:

```python
# Read magnetic structure
atoms = read_structure("magnetic.xyz")
print(atoms.get_initial_magnetic_moments())
```

### Data Processing Modes

Multiple projection modes for electronic structure analysis:

```python
# Different projection modes for band structure
df, ef, proj = read_band("band.h5", mode=1)  # Element-resolved
df, ef, proj = read_band("band.h5", mode=2)  # Orbital-resolved (s,p,d,f)
df, ef, proj = read_band("band.h5", mode=5)  # Detailed orbital projections
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Changelog

### v2025.12.09

Recreate this package.