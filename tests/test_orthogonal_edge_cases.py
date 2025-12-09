"""Test edge cases and error handling for orthogonal supercell."""

import pytest
from ase.build import bulk

from ddpc.structure.orthogonal import (
    CubicSupercellTransformation,
    _proj,
    _round_and_make_arr_singular,
    find_orthogonal,
)
import numpy as np


class TestHelperFunctions:
    """Test helper functions."""

    def test_proj(self):
        """Test vector projection function."""
        a = np.array([1, 0, 0])
        b = np.array([1, 1, 0])
        proj = _proj(b, a)
        np.testing.assert_array_equal(proj, [1, 0, 0])

    def test_proj_orthogonal(self):
        """Test projection of orthogonal vectors."""
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        proj = _proj(b, a)
        np.testing.assert_array_equal(proj, [0, 0, 0])

    def test_round_and_make_arr_singular_identity(self):
        """Test rounding with identity matrix."""
        arr = np.eye(3)
        result = _round_and_make_arr_singular(arr)
        np.testing.assert_array_equal(result, np.eye(3, dtype=int))

    def test_round_and_make_arr_singular_zero_row(self):
        """Test rounding with a matrix that would have a zero row."""
        arr = np.array([[1.4, 0.1, 0.1], [0.1, 1.4, 0.1], [0.1, 0.1, 0.1]])
        result = _round_and_make_arr_singular(arr)
        # Should not have any zero rows
        assert not (result == 0).all(axis=1).any()

    def test_round_and_make_arr_singular_zero_col(self):
        """Test rounding with a matrix that would have a zero column."""
        arr = np.array([[1.4, 0.1, 0.1], [0.1, 1.4, 0.1], [0.1, 0.1, 1.4]])
        result = _round_and_make_arr_singular(arr)
        # Should not have any zero columns
        assert not (result == 0).all(axis=0).any()


class TestFindOrthogonal:
    """Test find_orthogonal function."""

    def test_no_cell(self):
        """Test error when no cell information."""
        from ase import Atoms

        atoms = Atoms("H")
        with pytest.raises(ValueError, match="no cell information"):
            find_orthogonal(atoms)

    def test_max_atoms_exceeded(self):
        """Test error when max_atoms is exceeded."""
        atoms = bulk("Cu", "fcc", a=3.6)
        with pytest.raises(ValueError, match="No orthogonal supercell found"):
            find_orthogonal(atoms, min_length=50.0, max_atoms=10)

    def test_max_length_required_for_orthorhombic(self):
        """Test that max_length is required for orthorhombic."""
        atoms = bulk("Cu", "fcc", a=3.6)
        with pytest.raises(ValueError):
            find_orthogonal(atoms, allow_orthorhombic=True, min_length=10.0)


class TestCubicSupercellTransformationEdgeCases:
    """Test edge cases for CubicSupercellTransformation."""

    def test_max_atoms_constraint(self):
        """Test that max_atoms constraint is respected."""
        atoms = bulk("Cu", "fcc", a=3.6)
        transformer = CubicSupercellTransformation(
            min_length=10.0, max_atoms=50, min_atoms=10
        )
        with pytest.raises(AttributeError, match="max number of atoms was exceeded"):
            transformer.apply_transformation(atoms)

    def test_max_length_constraint(self):
        """Test that max_length constraint is respected when possible."""
        atoms = bulk("Cu", "fcc", a=3.6)
        # Use reasonable constraints to get a valid supercell
        transformer = CubicSupercellTransformation(
            min_length=10.0, max_length=15.0, max_atoms=500
        )
        result = transformer.apply_transformation(atoms)
        # Verify all lattice vectors are within max_length
        lengths = result.cell.lengths()
        assert all(length <= 15.0 for length in lengths), f"Some lengths exceed max_length: {lengths}"
        # Verify at least one lattice vector is >= min_length
        assert any(length >= 10.0 for length in lengths), f"No length >= min_length: {lengths}"

    def test_min_atoms_constraint(self):
        """Test that min_atoms constraint is respected."""
        atoms = bulk("Cu", "fcc", a=3.6)
        transformer = CubicSupercellTransformation(
            min_length=10.0, min_atoms=50, max_atoms=500
        )
        result = transformer.apply_transformation(atoms)
        assert len(result) >= 50

    def test_force_diagonal_small_cell(self):
        """Test force_diagonal with small cell."""
        atoms = bulk("Si", "diamond", a=5.43)
        transformer = CubicSupercellTransformation(
            min_length=5.0, force_diagonal=True
        )
        result = transformer.apply_transformation(atoms)
        # Check that result is larger than original
        assert len(result) >= len(atoms)

    def test_orthorhombic_without_max_length(self):
        """Test that orthorhombic requires max_length."""
        atoms = bulk("Cu", "fcc", a=3.6)
        transformer = CubicSupercellTransformation(
            min_length=10.0, allow_orthorhombic=True
        )
        with pytest.raises(AttributeError, match="max_length is required"):
            transformer.apply_transformation(atoms)

    def test_transformation_matrix_set(self):
        """Test that transformation_matrix is set after transformation."""
        atoms = bulk("Cu", "fcc", a=3.6)
        transformer = CubicSupercellTransformation(
            min_length=10.0, max_atoms=500, force_diagonal=True
        )
        result = transformer.apply_transformation(atoms)
        assert transformer.transformation_matrix is not None
        assert isinstance(transformer.transformation_matrix, np.ndarray)
