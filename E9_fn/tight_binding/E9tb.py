import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

import sys
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util

class tbmodel_2D:
    """(Finite, 2D) tight binding model.
    
    Once the 2D version works it shouldn't be too hard to generalize to other dimensions."""
    def __init__(self,
                 lat_vec: list[np.ndarray],
                 basis_vec: list[np.ndarray],
                 lat_dim: list[int],
                 lat_bc: list[{0, 1}],
                 sublat_offsets: list[complex],
                 hoppings: list[complex],):
        """
        Args:
            lat_vec:        list of lattice vectors
            basis_vec:      list of basis vectors, in the unit of lattice vectors
            lat_dim:        list that specifies the number of unit cell in each lattice vector direction
            lat_bc:         tuple of 0 or 1, where 0 means open boundary condition, and 1 means closed bc.
            sublat_offsets: list of offsets on each sublattice
            hoppings:       (t, i, j, R). This establishes a hopping t between the i-th orbit in some unit cell
                (x, y), and the j-th orbit in unit cell (x + R[0], y + R[1])."""
        # Consistency checks
        self.dim = len(lat_vec)
        self.n_basis = len(basis_vec)
        if self.dim != 2:
            raise(Exception("I can only do two dimensions now"))
        if len(lat_dim) != self.dim:
            raise(Exception("Dimensionality in lat_dim doesn't match that of lat_vec"))
        if len(lat_bc) != self.dim:
            raise(Exception("Dimensionality in lat_bc doesn't match that of lat_vec"))
        for h in hoppings:
            if len(h[3]) != self.dim:
                raise(Exception("Dimensionality in some hopping input doesn't match that of lat_vec"))
            if h[1] >= self.n_basis or h[2] >= self.n_basis:
                raise(Exception("Sublattice index in some hopping input is out of bound"))
        if len(sublat_offsets) != self.n_basis:
            raise(Exception("Number of sublattices in sublat_offsets doesn't match that of basis_vec"))
        
        # Copies of inputs
        self.lat_vec = lat_vec
        self.basis_vec = basis_vec
        self.lat_dim = lat_dim
        self.lat_bc = lat_bc
        self.sublat_offsets = sublat_offsets
        self.hoppings = hoppings

        # Attributes derived from inputs
        self.n_cells = lat_dim[0] * lat_dim[1]
        self.n_orbs = self.n_cells * self.n_basis
    
    def get_reduced_index(self, i, j, k):
        """Returns the reduced index that represents the k-th orbital in the (i, j)-th unit cell."""
        return self.n_basis * (self.lat_dim[0] * j + i) + k
    
    def get_natural_index(self, indices):
        """Unpacks the reduced indices as (i, j, k), the k-th orbital in the (i, j)-th unit cell."""
        indices = np.array(indices)
        if np.any(indices >= self.n_orbs):
            raise(Exception("index out of range"))
        k = (indices % self.n_basis)
        i = (indices - k) % self.lat_dim[0]
        j = (indices - k) // self.lat_dim[0]
        return i, j, k
    
    def build_H(self):
        """Construct the hamiltonian of the given tight binding model."""
        if self.lat_bc[0] == 1:
            raise(Exception("I haven't added closed bc for dimension 0 yet"))
        if self.lat_bc[1] == 1:
            raise(Exception("I haven't added closed bc for dimension 1 yet"))
        
        H = np.zeros((self.n_orbs, self.n_orbs))
        # Put offsets on each sites (they lie on diagonals)
        for i, offset in enumerate(self.sublat_offsets):
            diag_ind = np.arange(i, self.n_orbs, self.n_basis)
            H[diag_ind, diag_ind] = offset
        # Put hoppings between sites (they lie on diagonals with offsets)
        for t, i, j, R in self.hoppings:
            logging.debug("hopping: ({}, {}, {}, {})".format(t, i, j, R))
            uc1_natural_uc_ind = np.array([[(ii, jj) for jj in range(self.lat_dim[1])]
                                                     for ii in range(self.lat_dim[0])])
            uc2_natural_uc_ind = np.array([[(ii + R[0], jj + R[1]) for jj in range(self.lat_dim[1])]
                                                                   for ii in range(self.lat_dim[0])])
            uc2_allowed_uc_ind = np.logical_and.reduce((
                (uc2_natural_uc_ind >= 0)[:, :, 0],               # i >= 0
                (uc2_natural_uc_ind < self.lat_dim[0])[:, :, 0],  # i < self.lat_dim[0]
                (uc2_natural_uc_ind >= 0)[:, :, 1],               # j >= 0
                (uc2_natural_uc_ind < self.lat_dim[1])[:, :, 1])) # j < self.lat_dim[1]
            uc1_natural_uc_ind = uc1_natural_uc_ind[uc2_allowed_uc_ind, :]
            uc2_natural_uc_ind = uc2_natural_uc_ind[uc2_allowed_uc_ind, :]
            uc1_ind = self.get_reduced_index(uc1_natural_uc_ind[:,0], uc1_natural_uc_ind[:,1], i)
            uc2_ind = self.get_reduced_index(uc2_natural_uc_ind[:,0], uc2_natural_uc_ind[:,1], j)
            logging.debug(uc1_ind)
            logging.debug(uc2_ind)
            H[uc1_ind, uc2_ind] = t
            H[uc2_ind, uc1_ind] = t.conjugate() # make sure that H is Hermitian
        return H

if __name__ == "__main__":
    logpath = '' # '' if not logging to a file
    loglevel = logging.DEBUG
    logroot = logging.getLogger()
    list(map(logroot.removeHandler, logroot.handlers))
    list(map(logroot.removeFilter, logroot.filters))
    logging.basicConfig(filename = logpath, level = loglevel)

    test_square = tbmodel_2D(lat_vec = [np.array([0, 1]), np.array([1, 0])],
                             basis_vec = [np.array([0, 0])],
                             lat_dim = np.array([10,10]),
                             lat_bc = (0, 0),
                             sublat_offsets = [0],
                             hoppings = [(1, 0, 0, (1, 0)),
                                         (1, 0, 0, (0, 1)),],)
    test_kagome = tbmodel_2D(lat_vec = [np.array([0, 1]),
                                        np.array([-np.sqrt(3)/2, 1/2])],
                             basis_vec = [np.array([0, 0]),
                                          np.array([0.5, 0]),
                                          np.array([0, 0.5])],
                             lat_dim = np.array([20,20]),
                             lat_bc = (0, 0),
                             sublat_offsets = [0,0,0],
                             hoppings = [(-1, 0, 1, (0, 0)),
                                         (-1, 0, 2, (0, 0)),
                                         (-1, 1, 2, (0, 0)),
                                         (-1, 0, 1, (-1, 0)),
                                         (-1, 0, 2, (0, -1)),
                                         (-1, 1, 2, (1, -1))
                                         ],)
    H_model = test_kagome.build_H()
    fig_H, ax_H = util.make_simple_axes(fignum = 100)
    ax_H.matshow(H_model)

    eigvals, eigvecs = eigh(H_model)
    fig, ax = util.make_simple_axes()
    ax.scatter(np.arange(len(eigvals)), eigvals)