import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import eigh

# User defined modules
E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
if str(E9path) not in sys.path:
    sys.path.insert(1, str(E9path))
from E9_fn import util
from E9_fn.tight_binding import E9tb
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

#%% --------------- Experiment initialization ---------------
lattice_str = "sawtooth"
lattice_len = 40
sys_len = 20
# Working from high to low temperature is easier
T_init_list = np.linspace(0.02, 1., 50)
# s_avg = np.hstack((np.linspace(0.81, 0.27, 28), np.linspace(0.25, 0.15, 21))) # sawtooth
V_rsv_offsets = np.linspace(-5.5, 2., 76)#np.linspace(-2., 2., 11)
# The two lists below both start from the value closest to 0
V_rsv_offsets_neg0 = V_rsv_offsets[np.where(V_rsv_offsets <= 0)][::-1]
V_rsv_offsets_pos = V_rsv_offsets[np.where(V_rsv_offsets > 0)]
l_res = 0
V_std_random = 0.
N_tot = 36
tnnn = 0.
runnum_to_load = 1

N_V_rsv_neg0 = len(V_rsv_offsets_neg0)
N_V_rsv_pos = len(V_rsv_offsets_pos)
N_tol = 1e-3            # Tolerable error in resultant N
S_tol = 1e-1            # Tolerable error in resultant S

#%% --------------- Load the pre-calculated orbital energies ---------------
parent_folder_name = lattice_str
data_folder = Path(E9path, "projects", "flat_band_cooling", "eigvals_library", parent_folder_name)
param_dict = dict()
if V_std_random != 0:
    param_dict["Vran"] = V_std_random
if tnnn != 0 and lattice_str in {"kagome_nnn", "bilayer_kagome"}:
    param_dict["tnnn"] = tnnn
if l_res != 0:
    param_dict["lres"] = l_res

# Dirty fix to make life easier when looking at sawtooth lattices
lattice_dim = (lattice_len, lattice_len)
N_orbs = lattice_len**2 * 3
if lattice_str.startswith("sawtooth"):
    lattice_dim = (lattice_len, 1)
    N_orbs = lattice_len * 2
N_offsets = len(V_rsv_offsets)
N_Ts = len(T_init_list)

tb_params = E9tb.get_model_params(lattice_str)
my_tb_model= E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)
sys_range = ((lattice_len - sys_len) // 2, (lattice_len + sys_len) // 2)
sys_natural_uc_ind = set([(ii, jj) for jj in range(my_tb_model.lat_dim[1])
                                    for ii in range(my_tb_model.lat_dim[0]) if sys_range[0] <= ii and ii < sys_range[1]])
sys_reduced_uc_ind = [my_tb_model.get_reduced_index(ii, jj, k) for (ii, jj) in sys_natural_uc_ind
                                                                for k in range(my_tb_model.n_basis)]
rsv_natural_uc_ind = set([(ii, jj) for jj in range(my_tb_model.lat_dim[1])
                                    for ii in range(my_tb_model.lat_dim[0])])
rsv_natural_uc_ind -= sys_natural_uc_ind
rsv_reduced_uc_ind = [my_tb_model.get_reduced_index(ii, jj, k) for (ii, jj) in rsv_natural_uc_ind
                                                                for k in range(my_tb_model.n_basis)]

# Collect all the pre-calculated results
all_eigvals = np.full((N_offsets, N_orbs), np.nan)
all_eigvecs = np.full((N_offsets, N_orbs, N_orbs), np.nan)  # of each orbial
all_density_sys = np.full((N_offsets, N_orbs), np.nan)      # of each orbial
eigvals_Vrsv0 = None                                        # No offset

for i_rsv, V_rsv in enumerate(V_rsv_offsets):
    folder_name = hpfn.get_model_str(lattice_str, lattice_dim, sys_len, V_rsv, runnum = runnum_to_load, param_dict = param_dict)
    with open(Path(data_folder, folder_name, "np_arrays.npz"), 'rb') as f:
        loaded_arrs_dict = np.load(f)
        all_eigvals[i_rsv, :] = loaded_arrs_dict["eigvals"]
        all_eigvecs[i_rsv, :, :] = loaded_arrs_dict["eigvecs"]
        all_density_sys[i_rsv, :] = loaded_arrs_dict["density_sys"]
        if np.isclose(V_rsv, 0., atol = 1e-4):
            eigvals_Vrsv0 = loaded_arrs_dict["eigvals"]
            density_sys_Vrsv0 = loaded_arrs_dict["density_sys"]

#%% --------------- Find \Delta E_{min} ---------------
# Initial state parameters
all_S_tot = np.full(N_Ts, np.nan)               # total entropy, determined by T_init
all_mu_init = np.full(N_Ts, np.nan)             # initial chemical potential, determined by T_init
all_fill_init = np.full((N_orbs, N_Ts), np.nan) # filling factors of each energy eigenstates (orbitals)
# By definition, all_fill_init is also the diagonal elements of the Correlation matrix
all_SvN_sys_init = np.full(N_Ts, np.nan)        # In general different from S_tot / 2
all_SvN_rsv_init = np.full(N_Ts, np.nan)        # In general different from S_tot / 2
all_N_sys_init = np.full(N_Ts, np.nan)          # Might be a little bit different from N_tot / 2

# "quantum adiabatic" results (filling factor unchanged)
# experiment (system + reservoir)
all_E_min = np.full((N_offsets, N_Ts), np.nan)      # from \rho_{min}, determined by all_fill_init

# "quantum adiabatic" results - system / reservoir only
all_N_sys = np.full((N_offsets, N_Ts), np.nan)     # of \rho_{min}, summing contribution from all orbitals
all_SvN_sys = np.full((N_offsets, N_Ts), np.nan)    # of \rho_{min}
all_SvN_rsv = np.full((N_offsets, N_Ts), np.nan)    # of \rho_{min}

# "minimum energy" results
all_T_star_G = np.full((N_offsets, N_Ts), np.nan)
all_mu_star_G = np.full((N_offsets, N_Ts), np.nan)
all_E_G = np.full((N_offsets, N_Ts), np.nan)
all_fails = np.zeros((N_offsets, N_Ts))
def get_i_rsv_for_flag(flag, i):
    if flag == 0:   # V_rsv_offsets_pos
        return i + N_V_rsv_neg0
    elif flag == 1:
        return (N_V_rsv_neg0 - 1) - i

# Some required inputs for the thmdy solvers that doesn't really do anything
sp_name = "fermi1"
name_sr1 = sp_name

for i_T, T_init in enumerate(T_init_list):
    # ------------ initial state ------------
    mu_init, rrst = eqfind.muVT_from_NVT_solver(N_tot, T_init, eigvals_Vrsv0)
    if rrst.converged:
        filling_factor_T = util.fermi_stat(eigvals_Vrsv0, T_init, mu_init)
        all_fill_init[:, i_T] = filling_factor_T
        all_S_tot[i_T] = thmdy.find_S(eigvals_Vrsv0, T_init, N_tot, +1, mu_init)
        if not np.allclose(thmdy.find_SvN_fermi(filling_factor_T), all_S_tot[i_T]):
            raise(Exception(f"S error from filling_factor_T = {thmdy.find_SvN_fermi(filling_factor_T) - all_S_tot[i_T]:.4f} (T = {T_init:.4f})"))
        all_mu_init[i_T] = mu_init
        S_tar = all_S_tot[i_T]
        logging.debug(f"S = {S_tar:.4f} for T = {T_init:.4f}")
        
        # --------- things the can be calculated without trimming away the reservoir ---------
        all_E_min[:, i_T] = all_eigvals @ filling_factor_T
        all_N_sys_init[i_T] = density_sys_Vrsv0 @ filling_factor_T
        all_N_sys[:, i_T] = all_density_sys @ filling_factor_T

        # --------- Correlation matrix in the system / reservoir region ---------
        corr_mat_pos_basis = all_eigvecs[i_rsv, :, :] @ np.diag(filling_factor_T) @ util.dagger(all_eigvecs[i_rsv, :, :])
        corr_mat_sys = corr_mat_pos_basis[np.ix_(sys_reduced_uc_ind, sys_reduced_uc_ind)]
        C_sys_eigvals, _ = eigh(corr_mat_sys)
        all_SvN_sys_init[i_T] = thmdy.find_SvN_fermi(C_sys_eigvals)
        corr_mat_rsv = corr_mat_pos_basis[np.ix_(rsv_reduced_uc_ind, rsv_reduced_uc_ind)]
        C_rsv_eigvals, _ = eigh(corr_mat_rsv)
        all_SvN_rsv_init[i_T] = thmdy.find_SvN_fermi(C_rsv_eigvals)
    else:
        logging.warning(f"Couldn't find the initial state for T = {T_init:.4f}!")
        logging.warning("Skipping the Gibbs state calculation")
        continue

    # ------------ Work through different reservoir offsets ------------
    T_guess_now = T_init
    mu_guess_now = mu_init
    # Break it down into V_rsv > 0 and V_rsv < 0 and start with near 0 for both
    for sgn_flag, V_rsv_offsets_sgn in enumerate([V_rsv_offsets_pos, V_rsv_offsets_neg0]):
        bad_rsv_log_fn = logging.warning  # All following rsv's also seems to fail, so surpress the logging level
        
        # i_rsv_sgn is for V_rsv_offsets_sgn only; i_rsv is for everything else (with length N_offsets)
        for i_rsv_sgn, V_rsv in enumerate(V_rsv_offsets_sgn):
            TVS_msg = f"T = {T_init:.4f}, V_rsv = {V_rsv:.4f}, S = {S_tar:.4f}"
            # --------- "quantum adiabatic" case ---------
            # ------ Find the Correlation matrix in the position basis ------
            corr_mat_pos_basis = all_eigvecs[i_rsv, :, :] @ np.diag(filling_factor_T) @ util.dagger(all_eigvecs[i_rsv, :, :])
            
            # ------ Trim away the reservoir, and find parameters of interest ------
            corr_mat_sys = corr_mat_pos_basis[np.ix_(sys_reduced_uc_ind, sys_reduced_uc_ind)]
            if not util.IsHermitian(corr_mat_sys):
                raise(Exception(f"corr_mat_sys is not hermitian ({TVS_msg})"))
            corr_mat_rsv = corr_mat_pos_basis[np.ix_(rsv_reduced_uc_ind, rsv_reduced_uc_ind)]
            if not util.IsHermitian(corr_mat_rsv):
                raise(Exception(f"corr_mat_rsv is not hermitian ({TVS_msg})"))
            
            C_sys_eigvals, _ = eigh(corr_mat_sys)
            if not np.allclose(C_sys_eigvals.sum(), all_N_sys[i_rsv, i_T]):
                raise(Exception(f"N_sys error from corr_mat = {C_sys_eigvals.sum() - all_N_sys[i_rsv, i_T]:.4f} ({TVS_msg})"))
            C_rsv_eigvals, _ = eigh(corr_mat_rsv)
            
            all_SvN_sys[i_rsv, i_T] = thmdy.find_SvN_fermi(C_sys_eigvals)
            all_SvN_rsv[i_rsv, i_T] = thmdy.find_SvN_fermi(C_rsv_eigvals)
            # --------- "minimum energy" case, i.e. Gibbs state with the same entropy ---------
            i_rsv = get_i_rsv_for_flag(sgn_flag, i_rsv_sgn)
            eigvals = all_eigvals[i_rsv, :]
            E_range = (eigvals[0], eigvals[-1])
            sr_list = [E9M.muVT_subregion(name_sr1, sp_name, N_orbs, +1, None, E_range, [], eigvals)]

            Tmu_out, orst = eqfind.muVT_from_NVS_solver(S_tar,
                                                        N_tot,
                                                        sr_list,
                                                        T_guess_now,
                                                        mu_guess_now,
                                                        Tbounds = (0, 2),
                                                        mubounds = (-6, 6),
                                                        options_dict = {"fatol": 1e-8, "xatol": 1e-8},
                                                        logging_fn = bad_rsv_log_fn)
            T_star, mu_star = Tmu_out[0], Tmu_out[1]

            # Double check that the solution is within the specified tolerance
            E_now = thmdy.find_E(eigvals, T_star, mu_star, +1)
            N_now = thmdy.find_Np(eigvals, T_star, mu_star, +1)
            S_now = thmdy.find_S(eigvals, T_star, N_now, +1, mu_star, E_now)
            N_err, S_err = N_now - N_tot, S_now - S_tar
            if orst.success:
                if abs(N_err) < N_tol and abs(S_err) < S_tol:
                    all_T_star_G[i_rsv, i_T] = T_star
                    all_mu_star_G[i_rsv, i_T] = mu_star
                    all_E_G[i_rsv, i_T] = E_now
                    # all_nu_sys[i_rsv, i_T] = np.sum(util.fermi_stat(eigvals, T_now, mu_now) * density_sys) / n_orbs_sys
                    T_guess_now = T_star
                    mu_guess_now = mu_star
                else:
                    bad_rsv_log_fn(f"The solution doesn't satisfy the specified tolerance! ({TVS_msg})")
                    bad_rsv_log_fn(f"(N_err = {N_err:.4f}, S_err = {S_err:.4f})")
                    bad_rsv_log_fn(f"Supressing the logging level to debug for the following offsets for T = {T_init:.4f}")
                    all_fails[i_rsv, i_T] = 1.
                    bad_rsv_log_fn = logging.debug
            else:
                bad_rsv_log_fn(f"The solver failed to converge for {TVS_msg}!")
                all_fails[i_rsv, i_T] = 1.
                bad_rsv_log_fn = logging.debug

all_Delta_E = all_E_min - all_E_G
all_Delta_s_sys = all_SvN_sys / all_N_sys - all_SvN_sys_init[np.newaxis, :] / all_N_sys_init[np.newaxis, :]

#%% --------------- T-x plots (initial parameters x) ---------------
util.set_custom_plot_style(overwrite = {"font.size": 10})
fig_title_str = (f"{lattice_str}, lattice size {lattice_dim}, system size {sys_len}"
                 f"\nN_atoms = {N_tot}, V_std_random = {V_std_random:.3f}")

fig_Tx = plt.figure()
ax_s_init = fig_Tx.add_subplot(111)
ax_s_init.plot(T_init_list, all_S_tot / N_tot, label = r'$s_{exp}$')
ax_s_init.plot(T_init_list, all_SvN_sys_init / all_N_sys_init, label = r'$s_{sys}$')
ax_s_init.plot(T_init_list, all_SvN_rsv_init / (N_tot - all_N_sys_init), label = r'$s_{rsv}$')
ax_s_init.plot(T_init_list, (all_SvN_sys_init + all_SvN_rsv_init - all_S_tot) / N_tot, label = (r'$s_{mutual}$' '?'))
ax_s_init.set_xlabel(r'$T_{init}$')
ax_s_init.set_ylabel(r'$s_{init}$')
ax_s_init.set_ylim([0, 1])
ax_s_init.legend()
fig_Tx.suptitle(fig_title_str)

#%% --------------- Vrsv-T plots (V_rsv scans) ---------------
fig_VrsvT = plt.figure(figsize = (9, 8))
ax_s_sys = fig_VrsvT.add_subplot(221)
ax_N_sys = fig_VrsvT.add_subplot(222)
ax_E = fig_VrsvT.add_subplot(223)
# ax_s_init = fig_VrsvT.add_subplot(224)

for ttl, data, ax in zip([r"$\Delta s_{sys}$",  r"$N_{S}$", r'$\Delta E_{tot} / N_{tot}$'],
                         [all_Delta_s_sys,      all_N_sys,  all_Delta_E / N_tot],
                         [ax_s_sys,             ax_N_sys,   ax_E]):
    mesh_T = ax.pcolormesh(V_rsv_offsets, T_init_list, data.T, shading = 'auto', cmap = 'viridis')
    ax.pcolormesh(V_rsv_offsets, T_init_list, np.ones_like(all_fails.T), shading = 'auto'
                  , color = "red", alpha = all_fails.T * 0.5, edgecolors = 'none')

    # Add colorbar, labels
    # fig_VS.colorbar(mesh_T, ax = ax, label = f'{ttl} value')
    ax.set_xlabel('V_rsv_offsets')
    ax.set_ylabel(r'$T_{init}$')
    ax.set_title(ttl)
    cntr = ax.contour(V_rsv_offsets, T_init_list, data.T, colors = "white", linestyles = "solid", levels = 10)
    ax.clabel(cntr, inline = True)

fig_VrsvT.suptitle(fig_title_str)