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
lattice_len = 10
lattice_dim = (lattice_len, lattice_len)
overwrite_param = {}
# overwrite_param = {"lat_bc": (1, 1)}  # Periodic boundary conditions
tb_params = E9tb.get_model_params(lattice_str, overwrite_param = overwrite_param)
my_tb_model= E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)

# Define the size of the system
sys_len = 6
sys_range = ((lattice_len - sys_len) // 2, (lattice_len + sys_len) // 2)

# Reservoir parameters
V_rsv_offset_final = -2.001

# Time evolution parameters in unit of 1/t_nn
t_ramp = 1                     # Time to ramp up the offset
t_hold = 0                      # Time to hold at the final offset
t_step = 0.01                   # Time step

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
sys_ind = np.array([i for i in np.arange(n_orbs) if i not in rsv_ind])
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
eigvecs_all = np.zeros((n_snapshots, n_orbs, n_orbs), dtype = complex)
W_of_t_all = np.zeros((n_snapshots, n_orbs, n_orbs), dtype = complex) # see eqn(10) for W(t) in my notes
U_to_init_all = np.zeros((n_snapshots, n_orbs, n_orbs), dtype = complex)
# Filling factors of each orbital at each time step (to be considered one column vector at a time)
fill_evolve_orbs_all = np.zeros((n_snapshots, n_orbs, n_orbs))
log_progress = (n_t_steps // 20) * np.arange(1, 21)  # Log progress every 5% of the total time steps

# Solve the eigenvalue problem of the initial and final Hamiltonian
#   The final Hamiltonian is used in finding the transformation matrix to the final basis
#   on the fly
eigvals_init, eigvecs_init = eigh(H_bare)
eigvals_final, eigvecs_final = eigh(H_bare + H_offset_final)
eigvecs_prev = eigvecs_init
eigvals_all[0, :] = eigvals_init
eigvecs_all[0, :, :] = eigvecs_init
W_of_t_all[0, :, :] = eigvecs_init
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
    if i_t in log_progress:
        logging.info(f"Time step {i_t}/{n_t_steps} ({i_t / n_t_steps * 100:.2f}%) ...")
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
    eigvecs_all[i_t + 1, :, :] = eigvecs_now
    W_of_t_all[i_t + 1, :] = eigvecs_prev
    U_to_init_all[i_t + 1, :, :] = util.dagger(eigvecs_init) @ eigvecs_prev
    fill_evolve_orbs_all[i_t + 1, :, :] = fill_evolve_now

#%% Post processing
overlap_with_init_all = np.abs(U_to_init_all.diagonal(axis1 = 1, axis2 = 2))**2
# lowest_overlap_with_init_all = np.min(overlap_with_init_all, axis = 0)
# _arr = np.arange(n_orbs)
# logging.info(f"States whose overlap with initial state are always above 0.99: {_arr[lowest_overlap_with_init_all > 0.99]}")
# logging.info(f"States whose overlap with initial state are always above 0.95: {_arr[lowest_overlap_with_init_all > 0.95]}")
# logging.info(f"States whose overlap with initial state are always above 0.8: {_arr[lowest_overlap_with_init_all > 0.8]}")

#%% overlap with the initial orbitals for matching energy levels
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

# fig_overlap_init, ax_overlap_init = plt.subplots()

# for i in range(n_orbs):
#     style = get_curve_style(i, n_orbs)
#     ax_overlap_init.plot(t_axis, overlap_with_init_all[:, i], **style)
# ax_overlap_init.set_xlabel("Time (1/t_nn)")
# ax_overlap_init.set_ylabel(r"$|\langle \varphi_i(t = 0)|\varphi_i(t) \rangle|^{2}$")
# ax_overlap_init.set_ylim(-0.05, 1.05)
# ax_overlap_init.set_title("Overlap with initial state for each orbital")
# fig_overlap_init.suptitle(f"V_final = {V_rsv_offset_final:.2f}, {n_orbs} orbitals, "
#                   + f"t = {t_tot:.2f}, dt = {t_step:.3f}")
#%% ################################ Quantum random walk #################################
# QRW_pos_ind = my_tb_model.get_reduced_index(5, 5, 1)
# QRW_pos_basis_init = np.eye(n_orbs)[QRW_pos_ind]
# QRW_spec_decom_init = util.dagger(eigvecs_init) @ QRW_pos_basis_init
# QRW_pos_basis_all = np.einsum("abc,c->ab", W_of_t_all, QRW_spec_decom_init)

# #%% Plots
# i_t_plot_QRW = np.linspace(0, n_t_steps, 5, dtype = int)
# N_QRW_plots = len(i_t_plot_QRW)

# fig_QRW = plt.figure(figsize = (N_QRW_plots * 4, 6))
# for j, i_t in enumerate(i_t_plot_QRW):
#     ax_QRW_now = fig_QRW.add_subplot(1, N_QRW_plots, j + 1)
#     my_tb_model.plot_H(ax = ax_QRW_now)
#     my_tb_model.plot_state(abs(QRW_pos_basis_all[i_t, :])**2, ax = ax_QRW_now)
#     ax_QRW_now.set_title(f"t = {t_axis[i_t]:.4f}")

################################## Multiparticle system ##################################
#%% inputs
T_init = 0.2                    # Initial temperature of the system
nu_tar_system = 5/12            # Target filling factor in the system
nu_tar_reservoir = 10/12          # Target filling factor in the reservoir

#%% Find the initial filling factors
# Find the total number of particles
N_init = ((lattice_len**2 - sys_len**2) * nu_tar_reservoir + sys_len**2 * nu_tar_system) * 3

# Find the chemical potential that gives the target filling factor in the system
mu_init, rrst_muVT_from_NVT = eqfind.muVT_from_NVT_solver(N_init, T_init, eigvals_init)
filling_factors_init = util.fermi_stat(eigvals_init, T_init, mu_init)

#%% Find the evolution of the filling factors in the system
# Filling factors of the actual system at each time step
fill_evolve_sys_all = np.einsum('abc,c->ab', fill_evolve_orbs_all, filling_factors_init)

# Sanity checks
N_total_check = np.sum(fill_evolve_sys_all, axis = 1)
if not np.allclose(N_total_check, N_init):
    logging.warning("Total number of particles does not match the initial number at some time step.")
    fig_N_check, ax_N_check = plt.subplots()
    ax_N_check.plot(t_axis, N_total_check - N_init, label='N_total')
    ax_N_check.set_title(f"N_init = {N_init}, {n_orbs} orbitals")
if np.any(fill_evolve_sys_all < 0) or np.any(fill_evolve_sys_all > 1):
    logging.warning("Filling factors out of bounds [0, 1] at some time step.")

#%% Post processing
E_tot_all = np.sum(fill_evolve_sys_all * eigvals_all, axis = 1)
E_tot_adiabatic_all = eigvals_all @ filling_factors_init
pop_in_rsv = np.sum(abs(eigvecs_all[:, rsv_ind, :])**2, axis = 1)

#%% Plots (full system)
# Plot the changes in filling factors after the whole ramp
# fig_dia, ax_dia = plt.subplots()
# ax_dia.plot(np.arange(n_orbs), filling_factors_all[-1] - filling_factors_all[0])

fig_fill = plt.figure(figsize = (10, 13), tight_layout = True)
gs = GridSpec(nrows = 7, ncols = 2, figure = fig_fill)
ax_fill_by_orb = fig_fill.add_subplot(gs[0, :])
ax_E_tot = fig_fill.add_subplot(gs[1, :])
ax_fill_by_E = fig_fill.add_subplot(gs[2:, 0])
ax_fill_by_E_adia = fig_fill.add_subplot(gs[2:, 1])

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
ax_E_tot.plot(t_axis[1:], (E_tot_all - E_tot_adiabatic_all)[1:])
ax_E_tot.set_title("Total energy increase of the system from non-adiabatic evolution")
ax_E_tot.set_xlabel("Time (1/t_nn)")
ax_E_tot.set_ylabel("E/t_nn")
ax_E_tot.set_ylim(bottom = 0)
ax_E_tot.legend()

# Plot the filling factors of each orbital as a function of energy
sampling_rate = 1
alpha_pwr = 1
for i in range(n_orbs):
    alpha_original = fill_evolve_sys_all[:, i]**alpha_pwr
    alpha_limited = np.clip(alpha_original, 0, 1)
    # pop_in_rsv = np.sum(abs(eigvecs_all[::sampling_rate, rsv_ind, i])**2, axis = 1)
    ax_fill_by_E.scatter(t_axis[::sampling_rate], eigvals_all[::sampling_rate, i], s = 1,
                         color = plt.cm.viridis(pop_in_rsv[::sampling_rate, i]), marker = ".", alpha = alpha_limited)
    
    alpha_original_adia = np.ones_like(t_axis[::sampling_rate]) * filling_factors_init[i]**alpha_pwr
    alpha_limited_adia = np.clip(alpha_original_adia, 0, 1)
    ax_fill_by_E_adia.scatter(t_axis[::sampling_rate], eigvals_all[::sampling_rate, i], s = 1,
                         color = plt.cm.viridis(pop_in_rsv[::sampling_rate, i]), marker = ".", alpha = alpha_limited_adia)
ax_fill_by_E.set_title((rf"$\alpha = \nu^{{{alpha_pwr:.1f}}}$" "; more yellow = more weight in the reservoir"))
ax_fill_by_E_adia.set_title("adiabatic case")
ax_fill_by_E.set_ylabel("E/t_nn")

#%% Tracking single orbitals
rsv_flat_orbs = np.array([i for i in range(196, 230)])          # "rsv_flat" below - states in the flat band of the reservoir
orbs_of_interest = rsv_flat_orbs                                # "ooi" below
fig_ooi = plt.figure(figsize = (6, 11), tight_layout = True)
ax_ooi_rsv = fig_ooi.add_subplot(3, 1, 1)
ax_ooi_adia = fig_ooi.add_subplot(3, 1, 2)
ax_ooi_adia_final = fig_ooi.add_subplot(3, 1, 3)

# fill_ooi traces the filling factor in the time-dependent basis
# !!! assuming that at t = 0 we have a single particle in an orbital of interest !!!
# fill_in_rsv_flat sums up the occupation in rsv_flat
fill_in_rsv_flat_final_ooi = np.zeros(len(orbs_of_interest))
for j, i_orb in enumerate(orbs_of_interest):
    color = plt.cm.viridis((i_orb - orbs_of_interest[0]) / (orbs_of_interest[-1] - orbs_of_interest[0]))
    ax_ooi_rsv.plot(t_axis, pop_in_rsv[:, i_orb], color = color)

    fill_ooi = np.einsum('abc,c->ab', fill_evolve_orbs_all, np.eye(n_orbs)[i_orb,:])
    fill_in_rsv_flat_ooi = np.sum(fill_ooi[:,rsv_flat_orbs], axis = 1)
    fill_in_rsv_flat_final_ooi[j] = fill_in_rsv_flat_ooi[-1]
    ax_ooi_adia.plot(t_axis, fill_in_rsv_flat_ooi, color = color)

ax_ooi_adia_final.plot(rsv_flat_orbs, fill_in_rsv_flat_final_ooi)
fig_ooi.suptitle("Time evolution for each of the reservoir flat band states")
ax_ooi_rsv.set_ylabel(r"$\langle \psi_{i}(t) | \hat{\mathrm{P}}_{rsv} |\psi_{i}(t) \rangle$")
ax_ooi_rsv.set_title("Wavefunction weight in the reservoir")

ax_ooi_adia.set_title("Remaining occupation in the reservoir flat band")
ax_ooi_adia.set_ylabel(r"$\langle \psi_{i}(t) | \hat{\mathrm{P}}_{rsv flat}(t) |\psi_{i}(t) \rangle$")
ax_ooi_adia.set_yscale("log")
ax_ooi_adia.set_ylim((1e-3, 1e0))
ax_ooi_adia.set_xlabel("time")

ax_ooi_adia_final.set_title("Remaining occupation at the end")
ax_ooi_adia_final.set_xlabel("state label")
############################# Find matching muVT system #############################
#%% Try to find a muVT system (grand canonical ensemble) that matches the final total energy
down_sample = 10
t_axis_down_sample = t_axis[::down_sample]
n_samples = n_snapshots // down_sample + 1
mu_all = np.zeros(n_samples)
T_all = np.zeros(n_samples)
S_all = np.zeros(n_samples)
mu_all[0] = mu_init
T_all[0] = T_init
S_all[0] = thmdy.find_S(eigvals_init, T_init, N_init, xi = 1, mu = mu_init, E_total = E_tot_all[0])
mu_adia_all = np.zeros(n_samples)
T_adia_all = np.zeros(n_samples)
S_adia_all = np.zeros(n_samples)
mu_adia_all[0] = mu_init
T_adia_all[0] = T_init
S_adia_all[0] = S_all[0]

for i_samp in range(1, n_samples):
    i_t = i_samp * down_sample
    mu_all[i_samp], T_all[i_samp], _ = eqfind.muVT_from_NVE_solver(
        N_init, E_tot_all[i_t], eigvals_all[i_t, :], T_all[i_samp - 1])
    S_all[i_samp] = thmdy.find_S(eigvals_all[i_t, :], T_all[i_samp], N_init, xi = 1, mu = mu_all[i_samp], E_total = E_tot_all[i_t])
    mu_adia_all[i_samp], T_adia_all[i_samp], _ = eqfind.muVT_from_NVE_solver(
        N_init, E_tot_adiabatic_all[i_t], eigvals_all[i_t, :], T_adia_all[i_samp - 1])
    S_adia_all[i_samp] = thmdy.find_S(eigvals_all[i_t, :], T_adia_all[i_samp], N_init, xi = 1,
                                         mu = mu_adia_all[i_samp], E_total = E_tot_adiabatic_all[i_t])
print(f"Final muVT: mu = {mu_all[-1]:.6f}, T = {T_all[-1]:.6f}")

#%% Plots
# Time evolution of matching GCE thermodynamical parameters
fig_muVT = plt.figure(figsize = (6, 10), tight_layout = True)
ax_mu_evolve = fig_muVT.add_subplot(3, 1, 1)
ax_T_evolve = fig_muVT.add_subplot(3, 1, 2)
ax_S_evolve = fig_muVT.add_subplot(3, 1, 3)

ax_mu_evolve.plot(t_axis_down_sample, mu_all, label = r"$\mu^*$")
ax_mu_evolve.plot(t_axis_down_sample, mu_adia_all, label = r"$\mu_{adia}$")
ax_mu_evolve.set_ylabel(r"$\mu$")
ax_mu_evolve.set_title("Chemical potential evolution")
ax_mu_evolve.legend()

ax_T_evolve.plot(t_axis_down_sample, T_all, label = r"$T^*$")
ax_T_evolve.plot(t_axis_down_sample, T_adia_all, label = r"$T_{adia}$")
ax_T_evolve.set_ylabel(r"$T$")
ax_T_evolve.set_title("Temperature evolution")
# ax_T_evolve.legend()

ax_S_evolve.plot(t_axis_down_sample, S_all, label = r"$S^*$")
ax_S_evolve.plot(t_axis_down_sample, S_adia_all, label = r"$S_{adia}$")
ax_S_evolve.set_xlabel("Time")
ax_S_evolve.set_ylabel(r"$S/k_B$")
ax_S_evolve.set_title("Entropy evolution")
# ax_S_evolve.legend()

# Filling distribution plots
fig_fill_hist = plt.figure(figsize = (10, 8), tight_layout = True)
ax_fill_init = fig_fill_hist.add_subplot(2, 2, 1)
ax_fill_init_norm = fig_fill_hist.add_subplot(2, 2, 2)
ax_fill_final = fig_fill_hist.add_subplot(2, 2, 3)
ax_fill_final_norm = fig_fill_hist.add_subplot(2, 2, 4)
color_sys = "#218ddb"
color_adia = "#fb4c2b"

for i_t, ax_fill, ax_fill_norm in zip([0, -1],
                                      [ax_fill_init, ax_fill_init_norm],
                                      [ax_fill_final, ax_fill_final_norm]):
    eigvals_bin_hist = np.histogram(eigvals_all[i_t, :], bins = 100)
    fill_sys_hist = np.histogram(eigvals_all[i_t, :], bins = 100, weights = fill_evolve_sys_all[i_t, :])
    fill_sys_adia_hist = np.histogram(eigvals_all[i_t, :], bins = 100, weights = filling_factors_init)
    eigavls_bin_vals = eigvals_bin_hist[0]
    eigavls_bin_leftlim = eigvals_bin_hist[1]
    ax_fill.stairs(*eigvals_bin_hist, color = "k", label = "All states")
    ax_fill.stairs(*fill_sys_hist, fill = True, label = "time-evolved distribution", color = color_sys)
    ax_fill.stairs(*fill_sys_adia_hist, label = "adiabatic (= initial) distribution", color = color_adia)
    ax_fill.set_xlabel(r"E/t_{nn}")
    ax_fill.set_ylabel("Particle number")
    ax_fill.set_title("filling distribution (not normalized)")
    ax_fill.legend()

    ax_fill_norm.stairs(fill_sys_hist[0] / eigavls_bin_vals, fill_sys_hist[1], fill = True, color = color_sys, alpha = 0.6)
    ax_fill_norm.stairs(fill_sys_adia_hist[0] / eigavls_bin_vals, fill_sys_adia_hist[1], color = color_adia)
    ax_fill_norm.plot(eigavls_bin_leftlim, util.fermi_stat(eigavls_bin_leftlim, T_all[i_t], mu_all[i_t])
                            , color = color_sys, label = rf"T = {T_all[i_t]:.4f}, mu = {mu_all[i_t]:.4f}")
    ax_fill_norm.plot(eigavls_bin_leftlim, util.fermi_stat(eigavls_bin_leftlim, T_adia_all[i_t], mu_adia_all[i_t])
                            , color = color_adia, label = rf"T = {T_adia_all[i_t]:.4f}, mu = {mu_adia_all[i_t]:.4f}")
    ax_fill_norm.set_xlabel(r"E/t_{nn}")
    ax_fill_norm.set_ylabel("Particle number")
    ax_fill_norm.set_title("filling distribution (normalized by each orbital)")
    ax_fill_norm.legend()
fig_fill_hist.suptitle(f"left: t = 0, right: t = {t_ramp}")
#%% density matrix stuff
# rho_pure_final_all = np.einsum('ab,ca->abc', eigvecs_all[-1,:,:], eigvecs_all[-1,:,:].conj())
# rho_sum_final_all = np.einsum("abc,a->bc", rho_pure_final_all, fill_evolve_sys_all[-1,:])
# N_rsv_final = rho_sum_final_all.diagonal()[rsv_ind].sum()
# testsysrho = np.pad(rho_sum_final_all[sys_ind][:,sys_ind], (0, 1), constant_values = 0)
# testsysrho[-1, -1] = N_init - testsysrho.diagonal().sum() # the last index is for vacuum state