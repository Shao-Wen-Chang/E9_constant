import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# User defined modules
E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
import equilibrium_finder as eqfind # For some reason removing this line gives an error (ModuleNotFoundError: No module named 'E9_fn')
sys.path.insert(1, E9path)
from E9_fn import util
import E9_fn.E9_models as E9M
# fermions in a predefined lattice (+ offset) potential
# I don't distinguish system and reservoir here, since they are already included in
# the tight binding calculation

#%% Experiment initialization
# Load the pre-calculated orbital energies, and get some basic values from it
file_name = "kagome_nnn-0p1_lat10x10_sys6x6_Vrsv-2.npz"
file_path = Path(E9path, "projects", "flat_band_cooling", "eigvals_library", file_name)
loaded_file = np.load(file_path)
E_orbs_exp = loaded_file["eigvals"]
d_sys_exp = loaded_file["density_sys"]
n_orbs = len(E_orbs_exp)
n_sys = loaded_file["n_sys"]
n_rsv = n_orbs - n_sys
E_range = (E_orbs_exp[0], E_orbs_exp[1])

sp_name = "fermi1"
name_sr1 = sp_name
sr_list = [E9M.muVT_subregion(name_sr1, sp_name, n_orbs, +1, None, E_range, [], E_orbs_exp)]
#%% Calculation
mu_scan = np.linspace(-0.3, 0.3, 61)
T_scan = np.linspace(0.01, 0.5, 50)
exp_list = [[None for _ in mu_scan] for _ in T_scan]
s_list = np.zeros_like(exp_list, dtype = float)
N_list = np.zeros_like(exp_list, dtype = float)
nu_sys_list = np.zeros_like(exp_list, dtype = float)
for i, T in enumerate(T_scan):
    for j, mu in enumerate(mu_scan):
        exp_ij = E9M.muVT_exp(T, sr_list, {sp_name: mu})
        exp_list[i][j] = exp_ij
        s_list[i, j] = exp_ij.S / n_orbs
        N_list[i, j] = exp_ij.N_dict[sp_name]
        nu_sys_list[i, j] = sum(util.fermi_stat(E_orbs_exp, T, mu) * d_sys_exp) / n_sys / 3

#%% plot
fig = plt.figure(figsize = (9, 9))
fig.suptitle(file_name)
ax_S = fig.add_subplot(221)
ax_N = fig.add_subplot(222)
ax_nsys = fig.add_subplot(223)
n_cntr = 10
for ttl, data, ax in zip(["s", "N", r"$\n_{sys}$"],
                         [s_list, N_list, nu_sys_list],
                         [ax_S, ax_N, ax_nsys]):
    img = ax.imshow(data, aspect = (mu_scan[-1] - mu_scan[0]) / (T_scan[-1] - T_scan[0]),
              extent = [mu_scan[0], mu_scan[-1], T_scan[0], T_scan[-1]], origin = "lower")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$T$")
    ax.set_title(ttl)
    cntr = ax.contour(mu_scan, T_scan, data, colors = "white", levels = n_cntr)
    ax.clabel(cntr, inline = True)
    plt.colorbar(mappable = img, ax = ax)
fig.tight_layout()
# %%
