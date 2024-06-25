import logging
import numpy as np
import matplotlib as mpl
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
        self.H = self.build_H()
    
    def get_reduced_index(self, i, j, k):
        """Returns the reduced index that represents the k-th orbital in the (i, j)-th unit cell."""
        return self.n_basis * (self.lat_dim[0] * j + i) + k
    
    def get_natural_index(self, indices):
        """Unpacks the reduced indices as (i, j, k), the k-th orbital in the (i, j)-th unit cell."""
        indices = np.array(indices)
        if np.any(indices >= self.n_orbs):
            raise(Exception("index out of range"))
        k = (indices % self.n_basis)
        i = ((indices - k) // self.n_basis) % self.lat_dim[0]
        j = ((indices - k) // self.n_basis) // self.lat_dim[0]
        return i, j, k
    
    def build_H(self):
        """Construct the Hamiltonian of the given tight binding model."""
        if self.lat_bc[0] == 1:
            raise(Exception("I haven't added closed bc for dimension 0 yet"))
        if self.lat_bc[1] == 1:
            raise(Exception("I haven't added closed bc for dimension 1 yet"))
        
        H = np.zeros((self.n_orbs, self.n_orbs), dtype = complex)
        # Put offsets on each sites (they lie on diagonals)
        for i, offset in enumerate(self.sublat_offsets):
            diag_ind = np.arange(i, self.n_orbs, self.n_basis)
            H[diag_ind, diag_ind] = offset
        # Put hoppings between sites (they lie on diagonals with offsets)
        for t, i, j, R in self.hoppings:
            logging.debug("hopping: ({}, {}, {}, {})".format(t, i, j, R))
            # I may not need the inner list - consider removing it when generalize to other dimensions
            uc1_natural_uc_ind = np.array([[(ii, jj) for jj in range(self.lat_dim[1])]
                                                     for ii in range(self.lat_dim[0])])
            uc2_natural_uc_ind = np.array([[(ii + R[0], jj + R[1]) for jj in range(self.lat_dim[1])]
                                                                   for ii in range(self.lat_dim[0])])
            uc2_allowed_uc_ind = np.logical_and.reduce((            # Basically we require: (ii, jj) are cell indices
                (uc2_natural_uc_ind >= 0)[:, :, 0],                 # ii >= 0
                (uc2_natural_uc_ind < self.lat_dim[0])[:, :, 0],    # ii < self.lat_dim[0]
                (uc2_natural_uc_ind >= 0)[:, :, 1],                 # jj >= 0
                (uc2_natural_uc_ind < self.lat_dim[1])[:, :, 1]))   # jj < self.lat_dim[1]
            # Below it is the easiest to think of each natural_uc_ind as (m, n) arrays, each element being a 2-tuple
            uc1_natural_uc_ind = uc1_natural_uc_ind[uc2_allowed_uc_ind, :]
            uc2_natural_uc_ind = uc2_natural_uc_ind[uc2_allowed_uc_ind, :]
            uc1_ind = self.get_reduced_index(uc1_natural_uc_ind[:, 0], uc1_natural_uc_ind[:, 1], i)
            uc2_ind = self.get_reduced_index(uc2_natural_uc_ind[:, 0], uc2_natural_uc_ind[:, 1], j)
            logging.debug(uc1_ind)
            logging.debug(uc2_ind)
            H[uc1_ind, uc2_ind] = t
            H[uc2_ind, uc1_ind] = t.conjugate() # make sure that H is Hermitian
        return H
    
    def get_lat_pos(self, ind):
        """Return the position of the lattice site specified by ind.
        
        ind can either be a reduced index or a (tuple = (i, j, k) of) natural index."""
        if type(ind) is int:
            i, j, k = self.get_natural_index(ind)
        else:
            i, j, k = ind
        b0, b1 = self.basis_vec[k]
        logging.debug("ind = {}: i = {}, j = {}, k = {}".format(ind, i, j, k))
        return (i + b0) * self.lat_vec[0] + (j + b1) * self.lat_vec[1]
    
    def plot_H(self, ax = None, H = None):
        """Plot the lattice in the specified axes using the Hamiltonian.
        
        One can add stuff to the Hamiltonian after it is initialized, and use this fuction to plot
        the new Hamiltonian. Otherwise, this function plots the bare Hamiltonian.
        The lattice sites (links) are more transparent if they are at a lower energy (weaker).
        Imaginary hopping values are currently indicated with dashed lines, but their directions are
        not indicated by any means yet. Consider using an arrow.
        Lattice site offsets should be real, otherwise we are working with a non-Hermitian system.
        Args:
            H: Hamiltonian to be plotted."""
        if ax is None: _fig, ax = util.make_simple_axes()
        if H is None: H = self.H
        sublat_colors = ["#1f77b4", "#7fbf7f", "#ff5b2b", "ffcc3f"]
        
        # Find the maximum offset and bond strengths (for plotting purposes)
        Hd = H.diagonal().real
        V_offset_max, V_offset_min = max(Hd), min(Hd)
        V_offset_range = V_offset_max - V_offset_min
        if V_offset_range == 0:
            get_offset_alpha = lambda x: 1
        else:
            get_offset_alpha = lambda x: 0.8 * (x.real - V_offset_min) / V_offset_range + 0.2
        
        H_offdiag_abs = abs(H - np.diag(Hd)).flatten()
        t_max, t_min = max(H_offdiag_abs), min(H_offdiag_abs)
        t_range = t_max - t_min
        if t_range == 0:
            get_t_alpha = lambda x: 1
        else:
            get_t_alpha = lambda x: min(0.9 * (abs(x) - t_min) / t_range + 0.1, 1) # min needed for floating point error
        # Loop through all the elements in H
        for ri1 in range(self.n_orbs):
            for ri2 in range(self.n_orbs):
                if ri1 == ri2: # this element (if non-zero) is an on-site offset
                    i, j, k = self.get_natural_index(ri1)
                    pos = self.get_lat_pos((i, j, k))
                    ax.scatter(pos[0], pos[1], color = sublat_colors[k], s = 15, alpha = get_offset_alpha(H[ri1, ri2]))
                else:
                    if H[ri1, ri2] != 0: # There is some hopping between the two sites
                        pos1, pos2 = self.get_lat_pos(ri1), self.get_lat_pos(ri2)
                        if not np.isclose(H[ri1, ri2].imag, 0):
                            if np.angle(H[ri1, ri2]) > 0:
                                xa, ya, dxa, dya = pos1[0], pos1[1], pos2[0] - pos1[0], pos2[1] - pos1[1]
                            else:
                                xa, ya, dxa, dya = pos2[0], pos2[1], pos1[0] - pos2[0], pos1[1] - pos2[1]
                            ax.arrow(xa, ya, dxa, dya, color = "black", length_includes_head = True,
                                     width = 0.03, head_width = 0.15, head_length = 0.15, alpha = get_t_alpha(H[ri1, ri2]), zorder = -100)
                        else:
                            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], ls = "-", lw = 1, color = "black",
                                    alpha = get_t_alpha(H[ri1, ri2]), zorder = -100)
        ax.set_aspect("equal")
        return ax
    
    def plot_state(self, state, ax = None, cutoff = 1e-4):
        """Plot a state in the specified axes.
        
        The size of the markers is proportional to the weight of the wave function, and the orientation is
        determined by the phase of the wave function.
        For a time-symmetric Hamiltonian the wave function can be chosen to be real. See Littlejohn 22-17.
        Args:
            state:  state to be plotted
            cutoff: don't plot the arrow if the amplitude is below this value
        """
        if len(state) != self.n_orbs:
            raise(Exception("The size of the input state doesn't match the number of orbitals"))
        if ax is None: _fig, ax = util.make_simple_axes()
        
        for ri in range(self.n_orbs):
            if abs(state[ri]) < cutoff: continue
            pos = self.get_lat_pos(ri)
            arrow_marker = mpl.markers.MarkerStyle(marker = "^")
            arrow_marker._transform = arrow_marker.get_transform().scale(0.8, 1.2).rotate(np.angle(state[ri]))
            plt.scatter(pos[0], pos[1], s = abs(state[ri]) * 100, marker = arrow_marker, color = "cyan", zorder = 100)
        return ax

# Some default values for hopping amplitudes
tnn = -1
tnnn = -0.1
tnnnc = 0.1 * np.exp(2j * np.pi * (1/3))

model_dictionary = {    # This defines everything except for the lattice dimensions
    "square":{
        "lat_vec": [np.array([0, 1]), np.array([1, 0])],
        "basis_vec": [np.array([0, 0])],
        "lat_bc": (0, 0),
        "sublat_offsets": [0],
        "hoppings": [(tnn, 0, 0, (1, 0)),
                     (tnn, 0, 0, (0, 1)),]
    },

    "kagome":{
        "lat_vec": [np.array([0, 1]),
                   np.array([-np.sqrt(3)/2, 1/2])],
        "basis_vec": [np.array([0, 0]),
                     np.array([0.5, 0]),
                     np.array([0, 0.5])],
        "lat_bc": (0, 0),
        "sublat_offsets": [0, 0, 0],
        "hoppings": [(tnn, 0, 1, (0, 0)),
                     (tnn, 0, 2, (0, 0)),
                     (tnn, 1, 2, (0, 0)),
                     (tnn, 0, 1, (-1, 0)),
                     (tnn, 0, 2, (0, -1)),
                     (tnn, 1, 2, (1, -1))]
    },

    "kagome_Haldane":{ # a "Haldane-like" model where there are complex hoppings between nnn
        "lat_vec": [np.array([0, 1]),
                   np.array([-np.sqrt(3)/2, 1/2])],
        "basis_vec": [np.array([0, 0]),
                     np.array([0.5, 0]),
                     np.array([0, 0.5])],
        "lat_bc": (0, 0),
        "sublat_offsets": [0, 0, 0],
        "hoppings": [(tnn, 0, 1, (0, 0)),
                     (tnn, 0, 2, (0, 0)),
                     (tnn, 1, 2, (0, 0)),
                     (tnn, 0, 1, (-1, 0)),
                     (tnn, 0, 2, (0, -1)),
                     (tnn, 1, 2, (1, -1)),
                     (tnnnc, 0, 1, (-1, 1)),
                     (tnnnc, 0, 1, (0, -1)),
                     (tnnnc, 1, 2, (1, 0)),
                     (tnnnc, 1, 2, (0, -1)),
                     (tnnnc, 2, 0, (1, 0)),
                     (tnnnc, 2, 0, (-1, 1))]
    },

    "kagome_visualize":{    # kagome lattice with energies chosen to make visualizing the hamiltonian easier
        "lat_vec": [np.array([0, 1]),
                   np.array([-np.sqrt(3)/2, 1/2])],
        "basis_vec": [np.array([0, 0]),
                     np.array([0.5, 0]),
                     np.array([0, 0.5])],
        "lat_bc": (0, 0),
        "sublat_offsets": [-6, -6, -6],
        "hoppings": [(1, 0, 1, (0, 0)),
                     (2, 0, 2, (0, 0)),
                     (3, 1, 2, (0, 0)),
                     (4, 0, 1, (-1, 0)),
                     (5, 0, 2, (0, -1)),
                     (6, 1, 2, (1, -1))]
    },
}

if __name__ == "__main__":
    logpath = '' # '' if not logging to a file
    loglevel = logging.INFO
    logroot = logging.getLogger()
    list(map(logroot.removeHandler, logroot.handlers))
    list(map(logroot.removeFilter, logroot.filters))
    logging.basicConfig(filename = logpath, level = loglevel)

    ### Define the model and solve it
    lattice_str = "kagome"
    lattice_halfdim = 5
    lattice_dim = (lattice_halfdim * 2, lattice_halfdim * 2)
    tb_params = model_dictionary[lattice_str]
    my_tb_model= tbmodel_2D(lat_dim = lattice_dim, **tb_params)
    H_bare = my_tb_model.build_H()
    
    # Add offset to the bare model
    sys_halfdim = 3
    sys_range = (lattice_halfdim - sys_halfdim, lattice_halfdim + sys_halfdim)
    V_rsv_offset = -2
    # Find what unit cells are in the reservoir by excluding the unit cells in the system
    sys_natural_uc_ind = set([(ii, jj) for jj in range(my_tb_model.lat_dim[1]) if sys_range[0] <= jj and jj < sys_range[1]
                                       for ii in range(my_tb_model.lat_dim[0]) if sys_range[0] <= ii and ii < sys_range[1]])
    rsv_natural_uc_ind = set([(ii, jj) for jj in range(my_tb_model.lat_dim[1])
                                       for ii in range(my_tb_model.lat_dim[0])])
    rsv_natural_uc_ind -= sys_natural_uc_ind
    rsv_natural_uc_ind = np.array(list(rsv_natural_uc_ind))
    logging.debug(rsv_natural_uc_ind)
    rsv_ind = np.hstack(
        [my_tb_model.get_reduced_index(rsv_natural_uc_ind[:,0], rsv_natural_uc_ind[:,1], k)
         for k in range(my_tb_model.n_basis)])
    H_offset = np.zeros_like(H_bare)
    H_offset[rsv_ind, rsv_ind] = V_rsv_offset

    H_total = H_bare + H_offset
    eigvals, eigvecs = eigh(H_total)

    ### Plots
    # fig_H, ax_H = util.make_simple_axes(fignum = 100)
    # ax_H.matshow(H_total)
    
    fig_E, ax_E = util.make_simple_axes()
    ax_E.scatter(np.arange(len(eigvals)), eigvals)
    ax_E.set_title("{} ({} unit cells), all states".format(lattice_str, lattice_dim))

    fig_lat, ax_lat = util.make_simple_axes()
    my_tb_model.plot_H(ax = ax_lat, H = H_total)
    my_tb_model.plot_state(eigvecs[0], ax_lat)