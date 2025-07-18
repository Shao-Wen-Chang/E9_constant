import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# User defined modules
E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
if str(E9path) not in sys.path:
    sys.path.insert(1, str(E9path))
from E9_fn import util
import E9_fn.E9_models as E9M
import E9_fn.thermodynamics as thmdy
from projects.flat_band_cooling import equilibrium_finder as eqfind
from projects.flat_band_cooling import helper_fns as hpfn

logpath = '' # '' if not logging to a file
loglevel = logging.INFO
logroot = logging.getLogger()
list(map(logroot.removeHandler, logroot.handlers))
list(map(logroot.removeFilter, logroot.filters))
logging.basicConfig(filename = logpath, level = loglevel)

# fermions in a predefined lattice (+ offset) potential
# I don't distinguish system and reservoir here, since they are already included in
# the tight binding calculation

#%% Experiment initialization
lattice_str = "kagome"
lattice_len = 20
sys_len = 12
S_total = np.linspace(400., 100, 31)    # Working from high to low entropy is easier
S_total = np.hstack((np.linspace(400., 220, 10), np.linspace(200, 100, 21)))    # Working from high to low entropy is easier
V_rsv_offsets = np.linspace(-3.5, -0.02, 88)#np.linspace(-2.5, -1.5, 51)#np.linspace(-2., 2., 11)
nu_sys = 5/12
nu_rsv = 5/6

# initial guesses at the first value of total entropy for each offset
T_guesses = np.array([0.3 for _ in V_rsv_offsets])
mu_guesses = np.array([2. + V for V in V_rsv_offsets])
N_tol = 1e-3            # Tolerable error in resultant N
S_tol = 1e-1            # Tolerable error in resultant S

#%% Load the pre-calculated orbital energies and find the mu and T for all entropies
data_folder = Path(E9path, "projects", "flat_band_cooling", "eigvals_library")
lattice_dim = (lattice_len, lattice_len)

n_orbs_tot = lattice_len**2 * 3
n_orbs_sys = sys_len**2 * 3
n_orbs_rsv = n_orbs_tot - n_orbs_sys
N_tot = int(nu_sys * n_orbs_sys + nu_rsv * n_orbs_rsv)
N_offsets = len(V_rsv_offsets)
N_S = len(S_total)

all_T = np.full((N_offsets, N_S), np.nan)
all_mu = np.full((N_offsets, N_S), np.nan)
all_E = np.full((N_offsets, N_S), np.nan)
all_S = np.full((N_offsets, N_S), np.nan)
all_N = np.full((N_offsets, N_S), np.nan)
all_nu_sys = np.full((N_offsets, N_S), np.nan)
all_fails = np.zeros((N_offsets, N_S))

for i_rsv, V_rsv in enumerate(V_rsv_offsets):
    logging.debug(f"working on offset = {V_rsv:.4f}...")
    folder_name = hpfn.get_model_str(lattice_str, lattice_dim, sys_len, V_rsv)
    with open(Path(data_folder, folder_name, "np_arrays.npz"), 'rb') as f:
        loaded_arrs_dict = np.load(f)
        eigvals = loaded_arrs_dict["eigvals"]
        density_sys = loaded_arrs_dict["density_sys"]
    E_range = (eigvals[0], eigvals[-1])

    sp_name = "fermi1"
    name_sr1 = sp_name
    sr_list = [E9M.muVT_subregion(name_sr1, sp_name, n_orbs_tot, +1, None, E_range, [], eigvals)]

    for i_S, S_now in enumerate(S_total):
        # Update guess values, and find the new equilibrium states
        if i_S == 0:
            T_guess_now = T_guesses[i_rsv]
            mu_guess_now = mu_guesses[i_rsv]
        else:
            T_guess_now, mu_guess_now = all_T[i_rsv, i_S - 1], all_mu[i_rsv, i_S - 1]
        Tmu_out, orst = eqfind.muVT_from_NVS_solver(S_now,
                                                    N_tot,
                                                    sr_list,
                                                    T_guess_now,
                                                    mu_guess_now,
                                                    Tbounds = (0, 2),
                                                    mubounds = (-3, 6),
                                                    options_dict = {"fatol": 1e-8, "xatol": 1e-8})
        if orst.success:
            T_now, mu_now = Tmu_out[0], Tmu_out[1]
            E_now = thmdy.find_E(eigvals, T_now, mu_now, +1)
            N_now = thmdy.find_Np(eigvals, T_now, mu_now, +1)
            all_T[i_rsv, i_S] = T_now
            all_mu[i_rsv, i_S] = mu_now
            all_E[i_rsv, i_S] = E_now
            all_N[i_rsv, i_S] = N_now
            all_S[i_rsv, i_S] = thmdy.find_S(eigvals, T_now, N_now, +1, mu_now, E_now)
            all_nu_sys[i_rsv, i_S] = np.sum(util.fermi_stat(eigvals, T_now, mu_now) * density_sys) / n_orbs_sys
            N_err, S_err = all_N[i_rsv, i_S] - N_tot, all_S[i_rsv, i_S] - S_now
            if abs(N_err) < N_tol and abs(S_err) < S_tol:
                pass
            else:
                logging.warning(f"The solution of V_rsv = {V_rsv:.4f} and S = {S_now:.4f} doesn't satisfy the specified tolerance!")
                logging.warning(f"(N_err = {N_err:.4f}, S_err = {S_err:.4f})")
                all_fails[i_rsv, i_S] = 1.
        else:
            logging.warning(f"The solver failed to converge for V_rsv = {V_rsv:.4f} and S = {S_now:.4f}!")
            all_fails[i_rsv, i_S] = 1.

#%% plot
ind_Sselect = np.array([i for i in range(0, N_S, 4)])

fig = plt.figure(figsize = (10, 8))
ax_T = fig.add_subplot(221)
ax_nu = fig.add_subplot(222)
ax_T_Sselect = fig.add_subplot(223)
cmap = plt.get_cmap('coolwarm')

for ttl, data, ax in zip(["T/J", r"$\nu_{S}$"],
                         [all_T, all_nu_sys],
                         [ax_T, ax_nu]):
    mesh_T = ax.pcolormesh(V_rsv_offsets, S_total, data.T, shading = 'auto', cmap = 'viridis')
    ax.pcolormesh(V_rsv_offsets, S_total, np.ones_like(all_fails.T), shading = 'auto'
                  , color = "red", alpha = all_fails.T * 0.5, edgecolors = 'none')

    # Add colorbar, labels
    fig.colorbar(mesh_T, ax = ax, label = f'{ttl} value')
    ax.set_xlabel('V_rsv_offsets')
    ax.set_ylabel('S_total')
    ax.set_title('T')
    ax.set_title(ttl)
    
    cntr = ax.contour(V_rsv_offsets, S_total, data.T, colors = "white", levels = 10)
    ax.clabel(cntr, inline = True)

for i in ind_Sselect:
    S = S_total[i]
    color = util.get_color(S, S_total, cmap, assignment = "value")
    # failed_this_S = all_fails[:, i].astype(bool)
    ax_T_Sselect.plot(V_rsv_offsets, all_T[:, i], color = color, label = f"S = {S}")
    # ax_T_Sselect.scatter(V_rsv_offsets[failed_this_S], np.nan_to_num(all_T)[failed_this_S, i], color = color, marker = ".")
ax_T_Sselect.legend()

fig.suptitle(f"{lattice_str}, lattice size {lattice_dim}, system size {sys_len}x{sys_len}; N_atoms = {N_tot}")
fig.tight_layout()

# %%
