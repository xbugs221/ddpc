"""Find orthogonal supercell of crystal structures. Based on pymatgen."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ase.atoms import Atoms


def find_orthogonal(atoms: Atoms, **kwargs) -> Atoms:
    """Find minimal orthogonal supercell.

    Args:
        atoms: Input structure
        kwargs: Pass to CubicSupercellTransformation. Common parameters:
            - min_atoms: Minimum number of atoms allowed in the supercell
            - max_atoms: Maximum number of atoms allowed in the supercell
            - min_length: Minimum length of the smallest supercell lattice vector (default: 15.0)
            - max_length: Maximum length of the larger supercell lattice vector
            - force_diagonal: If True, return a diagonal transformation matrix
            - force_90_degrees: If True, return a supercell with 90 degree angles
            - allow_orthorhombic: Allow orthorhombic cells instead of cubic
            - angle_tolerance: Tolerance for 90 degree angles (default: 1e-3)
            - step_size: Step size for increasing supercell size (default: 0.1)

    Returns
    -------
        Orthogonal supercell as ase.Atoms

    Raises
    ------
        ValueError: If no orthogonal cell found within constraints
        AttributeError: If constraints are exceeded during search
    """
    if not atoms.cell.array.any():
        raise ValueError("Input structure has no cell information")

    try:
        orthed_s = CubicSupercellTransformation(**kwargs).apply_transformation(atoms)
    except Exception as e:
        raise ValueError("No orthogonal supercell found. You may try other kwargs.") from e

    return orthed_s


def _proj(b: NDArray, a: NDArray) -> NDArray:
    """Get vector projection of vector b onto vector a.

    Args:
        b: Vector to be projected.
        a: Vector onto which `b` is projected.

    Returns
    -------
        Projection of `b` onto `a`.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    return (np.dot(b, a) / np.dot(a, a)) * a


def _round_and_make_arr_singular(arr: np.ndarray) -> np.ndarray:
    """This function rounds all elements of a matrix to the nearest integer,
    unless the rounding scheme causes the matrix to be singular, in which
    case elements of zero rows or columns in the rounded matrix with the
    largest absolute valued magnitude in the unrounded matrix will be
    rounded to the next integer away from zero rather than to the
    nearest integer.

    The transformation is as follows. First, all entries in 'arr' will be
    rounded to the nearest integer to yield 'arr_rounded'. If 'arr_rounded'
    has any zero rows, then one element in each zero row of 'arr_rounded'
    corresponding to the element in 'arr' of that row with the largest
    absolute valued magnitude will be rounded to the next integer away from
    zero (see the '_round_away_from_zero(x)' function) rather than the
    nearest integer. This process is then repeated for zero columns. Also
    note that if 'arr' already has zero rows or columns, then this function
    will not change those rows/columns.

    Args:
        arr: Input matrix

    Returns
    -------
        Transformed matrix.
    """

    def round_away_from_zero(x):
        """Get 'x' rounded to the next integer away from 0.
        If 'x' is zero, then returns zero.
        E.g. -1.2 rounds to -2.0. 1.2 rounds to 2.0.
        """
        abs_x = abs(x)
        return math.ceil(abs_x) * (abs_x / x) if x != 0 else 0

    arr_rounded = np.around(arr)

    # Zero rows in 'arr_rounded' make the array singular, so force zero rows to
    # be nonzero
    if (~arr_rounded.any(axis=1)).any():
        # Check for zero rows in T_rounded

        # indices of zero rows
        zero_row_idxs = np.where(~arr_rounded.any(axis=1))[0]

        for zero_row_idx in zero_row_idxs:  # loop over zero rows
            zero_row = arr[zero_row_idx, :]

            # Find the element of the zero row with the largest absolute
            # magnitude in the original (non-rounded) array (i.e. 'arr')
            matches = np.absolute(zero_row) == np.amax(np.absolute(zero_row))
            col_idx_to_fix = np.where(matches)[0]

            # Break ties for the largest absolute magnitude
            r_idx = np.random.default_rng().integers(len(col_idx_to_fix))
            col_idx_to_fix = col_idx_to_fix[r_idx]

            # Round the chosen element away from zero
            arr_rounded[zero_row_idx, col_idx_to_fix] = round_away_from_zero(
                arr[zero_row_idx, col_idx_to_fix]
            )

    # Repeat process for zero columns
    if (~arr_rounded.any(axis=0)).any():
        # Check for zero columns in T_rounded
        zero_col_idxs = np.where(~arr_rounded.any(axis=0))[0]
        for zero_col_idx in zero_col_idxs:
            zero_col = arr[:, zero_col_idx]
            matches = np.absolute(zero_col) == np.amax(np.absolute(zero_col))
            row_idx_to_fix = np.where(matches)[0]

            for idx in row_idx_to_fix:
                arr_rounded[idx, zero_col_idx] = round_away_from_zero(arr[idx, zero_col_idx])
    return arr_rounded.astype(int)


class CubicSupercellTransformation:
    """A transformation that aims to generate a nearly cubic supercell structure
    from a structure.

    The algorithm solves for a transformation matrix that makes the supercell
    cubic. The matrix must have integer entries, so entries are rounded (in such
    a way that forces the matrix to be non-singular). From the supercell
    resulting from this transformation matrix, vector projections are used to
    determine the side length of the largest cube that can fit inside the
    supercell. The algorithm will iteratively increase the size of the supercell
    until the largest inscribed cube's side length is at least 'min_length'
    and the number of atoms in the supercell falls in the range
    ``min_atoms < n < max_atoms``.
    """

    def __init__(
        self,
        min_atoms: int | None = None,
        max_atoms: int | None = None,
        min_length: float = 15.0,
        max_length: float | None = None,
        force_diagonal: bool = False,
        force_90_degrees: bool = False,
        allow_orthorhombic: bool = False,
        angle_tolerance: float = 1e-3,
        step_size: float = 0.1,
    ):
        """
        Args:
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.
            max_length: Maximum length of the larger supercell lattice vector.
            force_diagonal: If True, return a transformation with a diagonal
                transformation matrix.
            force_90_degrees: If True, return a transformation for a supercell
                with 90 degree angles (if possible). To avoid long run times,
                please use max_atoms or max_length
            allow_orthorhombic: Instead of a cubic cell, also orthorhombic cells
                are allowed. max_length is required for this option.
            angle_tolerance: tolerance to determine the 90 degree angles.
            step_size: step_size which is used to increase the supercell.
                If allow_orthorhombic and force_90_degrees is both set to True,
                the chosen step_size will be automatically multiplied by 5 to
                prevent a too long search for the possible supercell.
        """
        self.min_atoms = min_atoms or -np.inf
        self.max_atoms = max_atoms or np.inf
        self.min_length = min_length
        self.max_length = max_length
        self.force_diagonal = force_diagonal
        self.force_90_degrees = force_90_degrees
        self.allow_orthorhombic = allow_orthorhombic
        self.angle_tolerance = angle_tolerance
        self.transformation_matrix = None
        self.step_size = step_size

    def apply_transformation(self, atoms: Atoms) -> Atoms:
        """The algorithm solves for a transformation matrix that makes the
        supercell cubic. The matrix must have integer entries, so entries are
        rounded (in such a way that forces the matrix to be non-singular). From
        the supercell resulting from this transformation matrix, vector
        projections are used to determine the side length of the largest cube
        that can fit inside the supercell. The algorithm will iteratively
        increase the size of the supercell until the largest inscribed cube's
        side length is at least 'num_nn_dists' times the nearest neighbor
        distance and the number of atoms in the supercell falls in the range
        defined by min_atoms and max_atoms.

        Returns
        -------
            supercell: Transformed supercell.
        """
        lat_vecs = atoms.cell.array

        if self.max_length is None and self.allow_orthorhombic:
            raise AttributeError("max_length is required for orthorhombic cells")

        if self.force_diagonal:
            scale = self.min_length / np.array(atoms.cell.lengths())
            self.transformation_matrix = np.diag(np.ceil(scale).astype(int))
            from ase.build import make_supercell

            return make_supercell(atoms, self.transformation_matrix)

        if not self.allow_orthorhombic:
            # boolean for if a sufficiently large supercell has been created
            sc_not_found = True

            # target_threshold is used as the desired cubic side lengths
            target_sc_size = self.min_length
            while sc_not_found:
                target_sc_lat_vecs = np.eye(3, 3) * target_sc_size
                length_vecs, n_atoms, superstructure, self.transformation_matrix = (
                    self.get_possible_supercell(lat_vecs, atoms, target_sc_lat_vecs)
                )
                # Check if constraints are satisfied
                if self.check_constraints(
                    length_vecs=length_vecs,
                    n_atoms=n_atoms,
                    superstructure=superstructure,
                ):
                    return superstructure

                # Increase threshold until proposed supercell meets requirements
                target_sc_size += self.step_size
                self.check_exceptions(length_vecs, n_atoms)

            raise AttributeError("Unable to find cubic supercell")

        if self.force_90_degrees:
            # prevent a too long search for the supercell
            self.step_size *= 5

        combined_list = [
            [size_a, size_b, size_c]
            for size_a in np.arange(self.min_length, self.max_length, self.step_size)
            for size_b in np.arange(self.min_length, self.max_length, self.step_size)
            for size_c in np.arange(self.min_length, self.max_length, self.step_size)
        ]
        combined_list = sorted(combined_list, key=sum)

        for size_a, size_b, size_c in combined_list:
            target_sc_lat_vecs = np.array([[size_a, 0, 0], [0, size_b, 0], [0, 0, size_c]])
            length_vecs, n_atoms, superstructure, self.transformation_matrix = (
                self.get_possible_supercell(lat_vecs, atoms, target_sc_lat_vecs)
            )
            # Check if constraints are satisfied
            if self.check_constraints(
                length_vecs=length_vecs, n_atoms=n_atoms, superstructure=superstructure
            ):
                return superstructure

            self.check_exceptions(length_vecs, n_atoms)
        raise AttributeError("Unable to find orthorhombic supercell")

    def check_exceptions(self, length_vecs, n_atoms):
        """Check supercell exceptions."""
        if n_atoms > self.max_atoms:
            raise AttributeError(
                "While trying to solve for the supercell, the max "
                "number of atoms was exceeded. Try lowering the number"
                "of nearest neighbor distances."
            )
        if (
            self.max_length is not None
            and np.max(np.linalg.norm(length_vecs, axis=1)) >= self.max_length
        ):
            raise AttributeError(
                "While trying to solve for the supercell, the max length was exceeded."
            )

    def check_constraints(self, length_vecs, n_atoms, superstructure):
        """
        Check if the supercell constraints are met.

        Returns
        -------
            bool

        """
        return bool(
            (
                np.min(np.linalg.norm(length_vecs, axis=1)) >= self.min_length
                and self.min_atoms <= n_atoms <= self.max_atoms
            )
            and (
                not self.force_90_degrees
                or np.all(
                    np.absolute(np.array(superstructure.cell.angles()) - 90) < self.angle_tolerance
                )
            )
        )

    @staticmethod
    def get_possible_supercell(lat_vecs, atoms, target_sc_lat_vecs):
        """
        Get the supercell possible with the set conditions.

        Returns
        -------
            length_vecs, n_atoms, superstructure, transformation_matrix
        """
        from ase.build import make_supercell

        transformation_matrix = target_sc_lat_vecs @ np.linalg.inv(lat_vecs)
        # round the entries of T and force T to be non-singular
        transformation_matrix = _round_and_make_arr_singular(transformation_matrix)
        proposed_sc_lat_vecs = transformation_matrix @ lat_vecs
        # Find the shortest dimension length and direction
        a = proposed_sc_lat_vecs[0]
        b = proposed_sc_lat_vecs[1]
        c = proposed_sc_lat_vecs[2]
        length1_vec = c - _proj(c, a)  # a-c plane
        length2_vec = a - _proj(a, c)
        length3_vec = b - _proj(b, a)  # b-a plane
        length4_vec = a - _proj(a, b)
        length5_vec = b - _proj(b, c)  # b-c plane
        length6_vec = c - _proj(c, b)
        length_vecs = np.array(
            [
                length1_vec,
                length2_vec,
                length3_vec,
                length4_vec,
                length5_vec,
                length6_vec,
            ]
        )
        # Get number of atoms
        superstructure = make_supercell(atoms, transformation_matrix)
        n_atoms = len(superstructure)
        return length_vecs, n_atoms, superstructure, transformation_matrix
