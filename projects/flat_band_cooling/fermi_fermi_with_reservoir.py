# Recommended import call: import fermi_fermi_with_reservoir as ffwr
import sys
from copy import deepcopy
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
# User defined modules
import equilibrium_finder as eqfind
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util
import E9_fn.E9_models as E9M
# spin-1/2 fermions in a step potential with flat band dispersion

#%% Experiment initialization 
#   "system": 1/10 of the number of sites, half-filled
#   "reservoir": 9/10 of the number of sites with lower on-site potential
V = 3 * 100**2      # size of the experiment (imagine a kagome lattice with n*n unit cells)
T = 0.4             # (fundamental) temperature (k_B * T)
bandwidth = 6       # bandwidth (of band structure; 6 for tight-binding kagome lattice)

# Shared by systems
r_s = 0.1            # ratio of the system size
V_s = int(V * r_s) # size of the system
E_range_s = (0, bandwidth + 1)    # energies considered in calculation
    # Defining DoS
simple_2D_DoS = lambda x: np.tanh(E_range_s[1] - x) * util.rect_fn(x, 0, E_range_s[1])
norm_factor = quad(simple_2D_DoS, E_range_s[0], E_range_s[1])[0]
simple_flatband_DoS = lambda x: (2/3) * simple_2D_DoS(x) \
    + (1/3) * norm_factor * util.Gaussian_1D(x, s = 0.1, mu = 4)
DoS_sys = simple_flatband_DoS

# Shared by reservoirs
offset = 1.5           # energy offset in the DoS of reservoir
V_rsv = int(V - V_s) # size of the reservoir
E_range_rsv = (offset, offset + bandwidth + 1) # energies considered in calculation
DoS_rsv = lambda x: simple_flatband_DoS(x - offset)

# system_fermi1 specific
nu_sf1 = 3/6               # filling factor (1 is fully filled)
Np_sf1 = int(V * r_s * nu_sf1)        # number of particles in the system

# system_fermi2 specific
nu_sf2 = 5/6               # filling factor (1 is fully filled)
Np_sf2 = int(V * r_s * nu_sf2)        # number of particles in the system

exp_ffwr = E9M.DoS_exp(T, [
    {"name": "fermi1", "V": V_s, "Np": Np_sf1, "stat": +1, "DoS": DoS_sys,
        "E_range": E_range_s, "reservoir": "", "comment": {}},
    {"name": "fermi1_rsv", "V": V_rsv, "Np": 0, "stat": +1, "DoS": DoS_rsv,
        "E_range": E_range_rsv, "reservoir": "fermi1", "comment": {}},
    {"name": "fermi2", "V": V_s, "Np": Np_sf2, "stat": +1, "DoS": DoS_sys,
        "E_range": E_range_s, "reservoir": "", "comment": {}},
    {"name": "fermi2_rsv", "V": V_rsv, "Np": 0, "stat": +1, "DoS": DoS_rsv,
        "E_range": E_range_rsv, "reservoir": "fermi2", "comment": {}},
])

#%% simulation
def calc_simple():
    """Find the thermodynamical values at fixed T."""
    exp_ffwr.find_outputs()
    fig_exp = plt.figure(1, figsize = (5, 8))
    ax_DoS = fig_exp.add_subplot(211)
    ax_table = fig_exp.add_axes(212)
    ax_table.set_axis_off()
    exp_ffwr.plot_DoSs(ax_DoS, offset_traces = True)
    exp_ffwr.tabulate_params(ax_table)
    plt.tight_layout()

def calc_isentropic():
    """For a range of total entropy, find the equilibrium conditions."""
    # Additional inputs
    isen_S_list = np.linspace(10000, 20000, 11)
    # Additional initialization
    isen_T_list = np.zeros_like(isen_S_list)
    isen_rrst_list = [None for _ in isen_S_list]
    isen_exp_list = [None for _ in isen_S_list]

    for i, S_now in enumerate(isen_S_list):
        logging.debug("working on loop #{}, S = {}...".format(i, S_now))
        isen_T_list[i], isen_rrst_list[i] = eqfind.isentropic_solver(S_now, exp_ffwr)
        
        exp_now = deepcopy(exp_ffwr)
        exp_now.T = isen_T_list[i]
        exp_now.find_outputs()
        isen_exp_list[i] = deepcopy(exp_now)
    
    # Plots
    # Entropy scan
    fig_exp = plt.figure(1)
    ax_SvsT = fig_exp.add_subplot(221)
    ax_SvsT.plot(isen_S_list, isen_T_list)
    ax_SvsT.set_xlabel(r"$S_{tot}$")
    ax_SvsT.set_ylabel("T")

    S_sys = np.array([x.species_list[0]["S"] + x.species_list[2]["S"] for x in isen_exp_list])
    ax_S_sys = fig_exp.add_subplot(222)
    ax_S_sys.plot(isen_S_list, S_sys / r_s / isen_S_list)
    ax_S_sys.set_xlabel(r"$S_{tot}$")
    ax_S_sys.set_ylabel(r"$S_{sys}/(r_s S_{tot})$")
    ax_S_sys.set_title("Effective reduction in the entropy of system")

    S_fermi1 = np.array([x.species_list[0]["S"] for x in isen_exp_list])
    ax_Srel = fig_exp.add_subplot(223)
    ax_Srel.plot(isen_S_list, S_fermi1 / S_sys)
    ax_Srel.set_xlabel(r"$S_{tot}$")
    ax_Srel.set_ylabel(r"$S_{sys}^{fermi1}/S_{tot}$")

    fig_exp.suptitle("Fermi-fermi with reservoirs")
    fig_exp.tight_layout()

    # Sample run
    fig_sam = plt.figure(2, figsize = (5,8))
    sampled_run = 2

    ax_DoS = fig_sam.add_subplot(211)
    isen_exp_list[sampled_run].plot_DoSs(ax_DoS, offset_traces = True)

    ax_tab = fig_sam.add_subplot(212)
    ax_tab.set_axis_off()
    isen_exp_list[sampled_run].tabulate_params(ax_tab)

    fig_sam.suptitle("Showing S = {}".format(isen_S_list[sampled_run]))

def main(**kwargs):
    calculation_mode = kwargs["calculation_mode"]

    if calculation_mode == "simple":
        calc_simple()    
    elif calculation_mode == "isentropic":
        calc_isentropic()

if __name__ == "__main__":
    main()