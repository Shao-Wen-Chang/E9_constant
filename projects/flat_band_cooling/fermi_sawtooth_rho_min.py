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
lattice_str = "sawtooth"
lattice_len = 40
sys_len = 20
# Working from high to low entropy is easier
T_init_list = np.linspace(0., 1., 51)
# s_avg = np.hstack((np.linspace(0.81, 0.27, 28), np.linspace(0.25, 0.15, 21))) # sawtooth
V_rsv_offsets = np.linspace(-5.5, 2., 76)#np.linspace(-2., 2., 11)
# The two lists below both start from the value closest to 0
V_rsv_offsets_neg0 = V_rsv_offsets[np.where(V_rsv_offsets <= 0)][::-1]
V_rsv_offsets_pos = V_rsv_offsets[np.where(V_rsv_offsets > 0)]
l_res = 0
V_std_random = 0.
N_tot = 42
tnnn = 0.
runnum_to_load = 1

N_V_rsv_neg0 = len(V_rsv_offsets_neg0)
N_V_rsv_pos = len(V_rsv_offsets_pos)
N_tol = 1e-3            # Tolerable error in resultant N
S_tol = 1e-1            # Tolerable error in resultant S

#%% Load the pre-calculated orbital energies
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

# Collect all the pre-calculated results
all_eigvals = np.full((N_offsets, N_orbs), np.nan)
all_density_sys = np.full((N_offsets, N_orbs), np.nan)  # of each orbial
eigvals_Vrsv0 = None                                    # No offset

for i_rsv, V_rsv in enumerate(V_rsv_offsets):
    folder_name = hpfn.get_model_str(lattice_str, lattice_dim, sys_len, V_rsv, runnum = runnum_to_load, param_dict = param_dict)
    with open(Path(data_folder, folder_name, "np_arrays.npz"), 'rb') as f:
        loaded_arrs_dict = np.load(f)
        all_eigvals[i_rsv, :] = loaded_arrs_dict["eigvals"]
        all_density_sys[i_rsv, :] = loaded_arrs_dict["density_sys"]
        if np.isclose(V_rsv, 0., atol = 1e-4):
            eigvals_Vrsv0 = loaded_arrs_dict["eigvals"]

#%% Find \Delta E_{min}
all_S_tot = np.full(N_Ts, np.nan)               # determined by T_init

# "quantum adiabatic" results
all_mu_QAA = np.full(N_Ts, np.nan)              # determined by T_init
all_E_min = np.full((N_offsets, N_Ts), np.nan)  # from \rho_{min}
# all_nu_sys = np.full((N_offsets, N_Ts), np.nan) # of \rho_{min}, summing contribution from all orbitals

# "minimum energy" results
all_T_G = np.full((N_offsets, N_Ts), np.nan)
all_mu_G = np.full((N_offsets, N_Ts), np.nan)
all_E_G = np.full((N_offsets, N_Ts), np.nan)
all_fails = np.zeros((N_offsets, N_Ts))
def get_i_rsv_for_flag(flag, i):
    if flag == 0:   # V_rsv_offsets_pos
        return i + N_V_rsv_neg0
    elif flag == 1:
        return (N_V_rsv_neg0 - 1) - i

for i_T, T_init in enumerate(T_init_list):
    # "quantum adiabatic" case - filling factor unchanged
    mu_init, _ = eqfind.muVT_from_NVT_solver(N_tot, T_init, eigvals_Vrsv0)
    filling_factors_init = util.fermi_stat(eigvals_Vrsv0, T_init, mu_init)
    all_S_tot[i_T] = thmdy.find_S(eigvals_Vrsv0, T_init, N_tot, +1, mu_init)
    all_mu_QAA[i_T] = mu_init
    all_E_min[:, i_T] = all_eigvals @ filling_factors_init

    # "minimum energy" case, i.e. Gibbs state with the same entropy
    S_tar = all_S_tot[i_T]
    if np.isnan(S_tar):
        logging.warning(f"S = {np.nan} for T = {T_init}; skipping the Gibbs state calculation")
        continue
    else:
        logging.debug(f"S = {np.nan} for T = {T_init}")
    sp_name = "fermi1"
    name_sr1 = sp_name

    # Break it down into V_rsv > 0 and V_rsv < 0 and start with near 0 for both
    T_guess_now = T_init
    mu_guess_now = mu_init
    for sgn_flag, V_rsv_offsets_sgn in enumerate([V_rsv_offsets_pos, V_rsv_offsets_neg0]):
        # i_rsv_sgn is for V_rsv_offsets_sgn only; i_rsv is for everything else (with length N_offsets)
        bad_rsv_log_fn = logging.warning  # All following rsv's also seems to fail, so surpress the logging level
        for i_rsv_sgn, V_rsv in enumerate(V_rsv_offsets_sgn):
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
                                                        options_dict = {"fatol": 1e-8, "xatol": 1e-8})
            T_now, mu_now = Tmu_out[0], Tmu_out[1]

            # Double check that the solution is within the specified tolerance
            E_now = thmdy.find_E(eigvals, T_now, mu_now, +1)
            N_now = thmdy.find_Np(eigvals, T_now, mu_now, +1)
            S_now = thmdy.find_S(eigvals, T_now, N_now, +1, mu_now, E_now)
            N_err, S_err = N_now - N_tot, S_now - S_tar
            if orst.success:
                if abs(N_err) < N_tol and abs(S_err) < S_tol:
                    all_T_G[i_rsv, i_T] = T_now
                    all_mu_G[i_rsv, i_T] = mu_now
                    all_E_G[i_rsv, i_T] = E_now
                    # all_nu_sys[i_rsv, i_T] = np.sum(util.fermi_stat(eigvals, T_now, mu_now) * density_sys) / n_orbs_sys
                    T_guess_now = T_now
                    mu_guess_now = mu_now
                else:
                    bad_rsv_log_fn(f"The solution of V_rsv = {V_rsv:.4f} and S = {S_tar:.4f} doesn't satisfy the specified tolerance!")
                    bad_rsv_log_fn(f"(N_err = {N_err:.4f}, S_err = {S_err:.4f})")
                    all_fails[i_rsv, i_T] = 1.
                    bad_rsv_log_fn = logging.debug
            else:
                bad_rsv_log_fn(f"The solver failed to converge for V_rsv = {V_rsv:.4f} and S = {S_tar:.4f}!")
                all_fails[i_rsv, i_T] = 1.
                bad_rsv_log_fn = logging.debug

all_Delta_E = all_E_min - all_E_G

#%% Vrsv-S plots (energy)
util.set_custom_plot_style(overwrite = {"font.size": 10})

fig_Delta_E = plt.figure()
ax_Delta_E = fig_Delta_E.add_subplot(111)
mesh_Delta_E = ax_Delta_E.pcolormesh(V_rsv_offsets, T_init_list, all_Delta_E.T / N_tot, cmap = 'viridis')
fig_Delta_E.colorbar(mesh_Delta_E, ax = ax_Delta_E, label = r'$\Delta E_{tot} / N_{tot}$')
ax_Delta_E.set_xlabel('V_rsv_offsets')
ax_Delta_E.set_ylabel('T_init')
fig_Delta_E.suptitle((f"{lattice_str}, lattice size {lattice_dim}, system size {sys_len}"
                    f"\nN_atoms = {N_tot}, V_std_random = {V_std_random:.3f}"))

cntr = ax_Delta_E.contour(V_rsv_offsets, T_init_list, all_Delta_E.T / N_tot, colors = "white", linestyles = "solid", levels = 10)
ax_Delta_E.clabel(cntr, inline = True)

#%% Vrsv-S plots (other thermaldynamical parameters)
fig_VS = plt.figure(figsize = (9, 8))
ax_T = fig_VS.add_subplot(221)
ax_nu = fig_VS.add_subplot(222)
ax_TTF = fig_VS.add_subplot(223)
ax_mu = fig_VS.add_subplot(224)

for ttl, data, ax in zip([r"$T/J$", r"$\nu_{S}$",   r"$T/T_F$", r"$\mu/J$"],
                         [all_T,    all_nu_sys,     all_TTF,    all_mu_QAA],
                         [ax_T,     ax_nu,          ax_TTF,     ax_mu]):
    mesh_T = ax.pcolormesh(V_rsv_offsets, S_total, data.T, shading = 'auto', cmap = 'viridis')
    ax.pcolormesh(V_rsv_offsets, S_total, np.ones_like(all_fails.T), shading = 'auto'
                  , color = "red", alpha = all_fails.T * 0.5, edgecolors = 'none')

    # Add colorbar, labels
    # fig_VS.colorbar(mesh_T, ax = ax, label = f'{ttl} value')
    ax.set_xlabel('V_rsv_offsets')
    ax.set_ylabel('S_total')
    ax.set_title(ttl)
    cntr = ax.contour(V_rsv_offsets, S_total, data.T, colors = "white", linestyles = "solid", levels = 10)
    ax.clabel(cntr, inline = True)

    # right y-axis
    ryax = ax.secondary_yaxis('right', functions = (lambda x: x / N_tot, lambda x: x * N_tot))
    ryax.set_ylabel(r"$s$")
fig_VS.suptitle((f"{lattice_str}, lattice size {lattice_dim}, system size {sys_len}x{sys_len}"
                 f"; N_atoms = {N_tot}, V_std_random = {V_std_random:.3f}"))

#%% fig2a plots
util.set_custom_plot_style(overwrite = {"font.size": 10})
mag_fig = 2.5
bool_save_fig = True

fig_2b = plt.figure(figsize = (1.7 * mag_fig, 1.3 * mag_fig))
ax_2b = fig_2b.add_subplot(111)

mesh_T_2b = ax_2b.pcolormesh(V_rsv_offsets, s_avg, all_T.T, shading = 'auto', cmap = 'coolwarm')
ax_2b.pcolormesh(V_rsv_offsets, s_avg, np.ones_like(all_fails.T), shading = 'auto'
                , color = "red", alpha = all_fails.T * 0.5, edgecolors = 'none')

# contours (filling factor in the system)
cntr_levels_ref = np.arange(0.3, 1.0001, 0.1)
cntr_levels_all = util.arr_insert_sorted(cntr_levels_ref, nu_sys)
cntr_lw = np.ones_like(cntr_levels_all)
cntr_lw[cntr_levels_all == nu_sys] = 2
cntr_colors = np.full_like(cntr_levels_all, "w", dtype = str)
cntr_colors[cntr_levels_all == nu_sys] = "k"

cntr_2b = ax_2b.contour(V_rsv_offsets, s_avg, all_nu_sys.T, levels = cntr_levels_all
                        , colors = cntr_colors, linestyles = "solid", linewidths = cntr_lw)
cntr_labels = ax_2b.clabel(cntr_2b, inline = True, fmt = "%.2f")
util.fix_clabel_orientation(cntr_labels)

# Add colorbar, labels
fig_2b.colorbar(mesh_T_2b, ax = ax_2b, label = r"$T/J$")
ax_2b.set_xlabel(r'$V_R$')
ax_2b.set_ylabel(r'$s^\circ$')

fig2apath = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "Projects",
                "2023 Optical potential engineering", "paper", "fig2"
                , f"fig2a_nsys{nu_sys:.2f}_v1.svg")
if bool_save_fig:
    fig_2b.savefig(fig2apath, format = "svg")

#%% Other plots
ind_Sselect = np.array([i for i in range(0, N_S, 4)])
ind_ref = np.where(np.round(V_rsv_offsets, 4) == 0.)[0]

fig_S = plt.figure(figsize = (15, 5))
ax_T_Sselect = fig_S.add_subplot(131)
ax_TTF_Sselect = fig_S.add_subplot(132)
ax_cool_Sselect = fig_S.add_subplot(133)
cmap = plt.get_cmap('coolwarm')

for i in ind_Sselect:
    s_plt = s_avg[i]
    color = util.get_color(s_plt, s_avg, cmap, assignment = "value")
    # failed_this_S = all_fails[:, i].astype(bool)
    ax_T_Sselect.plot(V_rsv_offsets, all_T[:, i], color = color, label = f"s = {s_plt:.3f}")
    ax_TTF_Sselect.plot(V_rsv_offsets, all_TTF[:, i], color = color)
    ax_cool_Sselect.plot(V_rsv_offsets, all_TTF[:, i] / all_TTF[ind_ref, i], color = color)
    # ax_T_Sselect.scatter(V_rsv_offsets[failed_this_S], np.nan_to_num(all_T)[failed_this_S, i], color = color, marker = ".")
ax_T_Sselect.legend()
ax_T_Sselect.set_title((r"$T$" " for selected " r"$S$"))
ax_TTF_Sselect.set_title((r"$T/T_F$" " for selected " r"$S$"))
ax_cool_Sselect.hlines(1., V_rsv_offsets[0], V_rsv_offsets[-1], colors = "k", linestyles = "--")
ax_cool_Sselect.scatter(V_rsv_offsets[ind_ref], 1., s = 100, color = "k", label = "reference")
ax_cool_Sselect.set_title(("Change in " r"$T/T_F$" " for selected " r"$S$"))
ax_cool_Sselect.legend()

# %%
