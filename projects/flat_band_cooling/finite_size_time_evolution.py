import sys
import logging
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from pathlib import Path
import os

# User defined modules
E9path = os.path.join("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
if E9path not in sys.path: sys.path.insert(1, E9path)
from E9_fn import util
import equilibrium_finder as eqfind
import E9_fn.thermodynamics as thmdy
from E9_fn.tight_binding import E9tb
import E9_fn.E9_models as E9M
# (non-interacting / spinless) fermions dynamics in a system + reservoir setup
'''
--- Initialize the system with some pure state with "temperature T" with no offset
    between the reservoir and the system, and fill to somewhere in the flat band
    --- The "temperature" determines the weight of each orbital in the initial state
        , given by the Fermi-Dirac distribution
--- Find (mu) and S for the initial configuration
    --- mu is ill-defined in a canonical ensemble, but maybe we can define something
        like the energy difference between N -and (N+1)-particle states
--- Evolve the state in time with a varying offset between the reservoir and the system
    --- For each step, decompose each eigenstate of the previous Hamiltonian into
        the superposition of the eigenstates of the new Hamiltonian
    --- Evolve each state, which simply accrues a phase factor
        --- This winds up being faster than calculating matrix exponentials, as in the
            singularity paper
--- Calculate the energy, entropy, and temperature of the system at the end
    (or at each time step, if managable)
'''

logpath = '' # '' if not logging to a file
loglevel = logging.INFO
logroot = logging.getLogger()
list(map(logroot.removeHandler, logroot.handlers))
list(map(logroot.removeFilter, logroot.filters))
logging.basicConfig(filename = logpath, level = loglevel)

#%% Inputs
### Lattice geometry and model parameters
# Lattice parameters
lattice_str = "kagome"
lattice_len = 8
lattice_dim = (lattice_len, lattice_len)
overwrite_param = {}
# overwrite_param = {"lat_bc": (1, 1)}  # Periodic boundary conditions
tb_params = E9tb.get_model_params(lattice_str, overwrite_param = overwrite_param)
my_tb_model= E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)

# Define the size of the system
sys_len = 5
sys_range = ((lattice_len - sys_len) // 2, (lattice_len + sys_len) // 2)

# Reservoir parameters
V_rsv_offset_final = -2

# Other inputs
T_init = 0.2                    # Initial temperature of the system
nu_tar_system = 5/12            # Target filling factor in the system
nu_tar_reservoir = 10/12          # Target filling factor in the reservoir

# Time evolution parameters in unit of 1/t_nn
t_ramp = 125                     # Time to ramp up the offset
t_hold = 0                      # Time to hold at the final offset
t_step = 0.05                   # Time step

#################################### Time evolution ####################################
#%% Initialization
### Finding Hamiltonians
H_bare = my_tb_model.H          # Bare Hamiltonian of the system (i.e. without offset)
n_orbs = my_tb_model.n_orbs
# Find the offset Hamiltonian
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
H_offset_final = np.zeros_like(H_bare)
H_offset_final[rsv_ind, rsv_ind] = V_rsv_offset_final

# Time evolution parameters
t_tot = t_ramp + t_hold
n_t_steps_ramp = int(t_ramp / t_step)
n_t_steps_hold = int(t_hold / t_step)
n_t_steps = n_t_steps_ramp + n_t_steps_hold
t_axis = np.arange(0, n_t_steps_ramp + n_t_steps_hold + t_step) * t_step
n_snapshots = n_t_steps + 1
eigvals_all = np.zeros((n_snapshots, n_orbs))
U_to_init_all = np.zeros((n_snapshots, n_orbs, n_orbs), dtype = complex)
# Filling factors of each orbital at each time step (to be considered one column vector at a time)
fill_evolve_orbs_all = np.zeros((n_snapshots, n_orbs, n_orbs))
# Filling factors of the actual system at each time step
fill_evolve_sys_all = np.zeros((n_snapshots, n_orbs, n_orbs))

# Solve the eigenvalue problem of the initial and final Hamiltonian
#   The final Hamiltonian is used in finding the transformation matrix to the final basis
#   on the fly
eigvals_init, eigvecs_init = eigh(H_bare)
eigvals_final, eigvecs_final = eigh(H_bare + H_offset_final)
eigvecs_prev = eigvecs_init
eigvals_all[0, :] = eigvals_init
U_to_init_all[0, :, :] = util.dagger(eigvecs_init) @ eigvecs_prev
fill_evolve_orbs_all[0, :, :] = np.eye(n_orbs)

#%% Time evolution using trotterized Hamiltonian
def get_offset_amplitude(i_t):
    """Returns the offset amplitude at time step i_t."""
    if i_t < n_t_steps_ramp:
        return (i_t + 1) / n_t_steps_ramp
    else:
        return 1.

# Time evolution
for i_t in range(n_t_steps):
    # Find the new Hamiltonian
    H_offset_now = H_offset_final * get_offset_amplitude(i_t)
    H_total = H_bare + H_offset_now
    eigvals_now, eigvecs_now = eigh(H_total)

    # Spectrally decompose all the eigenstates of the previous Hamiltonian, which is equivalent
    # to finding the unitary matrix that transforms from the previous eigenbasis to the new one
    U_to_now = util.dagger(eigvecs_now) @ eigvecs_prev
    overlap_now = np.diag(np.exp(-1j * eigvals_now * t_step)) @ U_to_now
    fill_evolve_now = overlap_now * overlap_now.conj()
    eigvecs_prev = eigvecs_now @ overlap_now
    eigvals_all[i_t + 1, :] = eigvals_now
    U_to_init_all[i_t + 1, :, :] = util.dagger(eigvecs_init) @ eigvecs_prev
    fill_evolve_orbs_all[i_t + 1, :, :] = fill_evolve_now

#%% Post processing
overlap_with_init_all = np.abs(U_to_init_all.diagonal(axis1 = 1, axis2 = 2))**2
lowest_overlap_with_init_all = np.min(overlap_with_init_all, axis = 0)
_arr = np.arange(n_orbs)
logging.info(f"States whose overlap with initial state are always above 0.99: {_arr[lowest_overlap_with_init_all > 0.99]}")
logging.info(f"States whose overlap with initial state are always above 0.95: {_arr[lowest_overlap_with_init_all > 0.95]}")
logging.info(f"States whose overlap with initial state are always above 0.8: {_arr[lowest_overlap_with_init_all > 0.8]}")

#%% Plots (single orbitals)
### Plot some single orbitals if needed
# fig_lat, ax_lat = util.make_simple_axes(fig_kwarg = {"figsize": (12, 6)})
# my_tb_model.plot_H(ax = ax_lat, H = H_total)
# my_tb_model.plot_state(eigvecs_prev[:,62], ax_lat)

def get_curve_style(i, total, cmap = plt.cm.viridis, n_highlight = 10,
                    alpha_high = 1.0, alpha_low = 0.5,
                    lw_high = 2.0, lw_low = 0.5):
    """Returns a dictionary with color, alpha, and linewidth for the i-th curve.

    Parameters:
        i: index of the curve
        total: total number of curves (for colormap scaling)
        cmap: colormap (e.g., plt.cm.viridis)
        n_highlight: Number of curves to highlight evenly across total
        alpha_high / low: alpha values
        lw_high / low: line width values
    """
    color = cmap(i / total)
    highlight_indices = set()
    if n_highlight > 0:
        highlight_indices = set(np.linspace(0, total - 1, n_highlight, dtype = int))
        
    if i in highlight_indices:
        return {'color': color, 'alpha': alpha_high, 'lw': lw_high}
    else:
        return {'color': color, 'alpha': alpha_low, 'lw': lw_low}

fig_adia, ax_adia = plt.subplots()

for i in range(n_orbs):
    style = get_curve_style(i, n_orbs)
    ax_adia.plot(t_axis, overlap_with_init_all[:, i], **style)
ax_adia.set_xlabel("Time (1/t_nn)")
ax_adia.set_ylabel(r"$|\langle \varphi_i(t = 0)|\varphi_i(t) \rangle|^{2}$")
ax_adia.set_ylim(-0.05, 1.05)
ax_adia.set_title("Overlap with initial state for each orbital")
fig_adia.suptitle(f"V_final = {V_rsv_offset_final:.2f}, {n_orbs} orbitals, "
                  + f"t = {t_tot:.2f}, dt = {t_step:.3f}")

################################## Multiparticle system ##################################
#%% Find the initial filling factors
# Find the total number of particles
N_init = ((lattice_len**2 - sys_len**2) * nu_tar_reservoir + sys_len**2 * nu_tar_system) * 3

# Find the chemical potential that gives the target filling factor in the system
mu_init, rrst = eqfind.muVT_from_NVT_solver(N_init, T_init, eigvals_init)
filling_factors_init = util.fermi_stat(eigvals_init, T_init, mu_init)

#%% Find the evolution of the filling factors in the system
fill_evolve_sys_all = np.einsum('abc,c->ab', fill_evolve_orbs_all, filling_factors_init)
N_total_check = np.sum(fill_evolve_sys_all, axis = 1)

# Sanity checks
if not np.allclose(N_total_check, N_init):
    logging.warning("Total number of particles does not match the initial number at some time step.")
    fig_N_check, ax_N_check = plt.subplots()
    ax_N_check.plot(t_axis, N_total_check - N_init, label='N_total')
    ax_N_check.set_title(f"N_init = {N_init}, {n_orbs} orbitals")

    mu_helper = np.linspace(-4, 2, 601)
    N_helper = np.array([sum(util.fermi_stat(eigvals_init, T_init, mu)) for mu in mu_helper])
    fig_N_vs_mu, ax_N_vs_mu = plt.subplots()
    ax_N_vs_mu.plot(mu_helper, N_helper - N_init, label = 'N - N_init')
    ax_N_vs_mu.axhline(0, color='k', linestyle='--', label='N_init')
    ax_N_vs_mu.legend()
if np.any(fill_evolve_sys_all < 0) or np.any(fill_evolve_sys_all > 1):
    logging.warning("Filling factors out of bounds [0, 1] at some time step.")

#%% Post processing
E_tot_all = np.sum(fill_evolve_sys_all * eigvals_all, axis = 1)
S_init = thmdy.find_S(eigvals_init, T_init, N_init, xi = -1,mu = mu_init, E_total = E_tot_all[0])

#%% Plots (full system)
# Plot the changes in filling factors after the whole ramp
# fig_dia, ax_dia = plt.subplots()
# ax_dia.plot(np.arange(n_orbs), filling_factors_all[-1] - filling_factors_all[0])

fig_fill = plt.figure(figsize = (6, 13), tight_layout = True)
gs = GridSpec(nrows = 7, ncols = 1, figure = fig_fill)
ax_fill_by_orb = fig_fill.add_subplot(gs[0, 0])
ax_E_tot = fig_fill.add_subplot(gs[1, 0])
ax_fill_by_E = fig_fill.add_subplot(gs[2:, 0])

# Plot the filling factors of each orbital
for i in range(n_orbs):
    style = get_curve_style(i, n_orbs)
    ax_fill_by_orb.plot(t_axis, fill_evolve_sys_all[:, i], **style)
ax_fill_by_orb.set_ylabel("Filling factor")
ax_fill_by_orb.set_ylim(-0.05, 1.05)
# ax_fill.set_ylim(bottom = -0.05)
ax_fill_by_orb.set_title("Filling factors of each orbital")
fig_fill.suptitle(f"V_final = {V_rsv_offset_final:.2f}, {n_orbs} orbitals, "
                  + f"T = {T_init:.2f}, N = {N_init:.2f}")

# Plot the total energy of the system
ax_E_tot.plot(t_axis[1:], (E_tot_all - eigvals_all @ filling_factors_init)[1:])
ax_E_tot.set_title("Total energy increase of the system from non-adiabatic evolution")
ax_E_tot.set_ylabel("E/t_nn")
ax_E_tot.set_ylim(bottom = 0)
ax_E_tot.legend()

# Plot the filling factors of each orbital as a function of energy
sampling_rate = 5
for i in range(n_orbs):
    alpha_original = fill_evolve_sys_all[:, i]**3
    alpha_limited = np.clip(alpha_original, 0, 1)
    if not np.allclose(alpha_limited - alpha_original, 0):
        logging.warning("transparency clipping occurred, which may affect the visualization.")
    ax_fill_by_E.scatter(t_axis[::sampling_rate], eigvals_all[::sampling_rate, i], s = 1,
                         color = "k", marker = ".", alpha = alpha_limited)
ax_fill_by_E.set_title(r"$\alpha = \nu^3$")
ax_fill_by_E.set_ylabel("E/t_nn")
ax_fill_by_E.set_xlabel("Time (1/t_nn)")

# %%
