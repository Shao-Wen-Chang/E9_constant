# Recommended import call: import fermi_kagome_scan_reservoir as fksr
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
# spin-1/2 fermions in a step potential with kagome lattice dispersion

#%% Experiment initialization 
#   "system": 1/10 of the number of sites, half-filled
#   "reservoir": 9/10 of the number of sites with lower on-site potential
V = 3 * 100**2      # size of the experiment (imagine a kagome lattice with n*n unit cells)
bandwidth = 6       # bandwidth (of band structure; 6 for tight-binding kagome lattice)

# System
r_s = 0.1            # ratio of the system size
V_s = int(V * r_s) # size of the system
E_range_s = (0, bandwidth + 1)    # energies considered in calculation
nu_sf1 = 5/12               # filling factor (1 is fully filled)
Np_sf1 = int(V * r_s * nu_sf1)        # number of particles in the system
    # Defining DoS
# DoS_sys = lambda x: util.kagome_DoS(x, hbw = bandwidth / 2) + (1/3) * util.dirac_delta(x, x0 = bandwidth, hw = 0.1)
DoS_sys = lambda x: util.kagome_DoS(x, hbw = bandwidth / 2, fhbw = 0.1)

# Reservoirs
offset_list = np.arange(-2, 1, 0.1, dtype = float)     # energy offsets in the DoS of reservoir
V_rsv = int(V - V_s) # size of the reservoir

# Combined
S_tot = 10000

def make_rsv_fn(offset):
    def fn_out(x):
        return DoS_sys(x - offset)
    return fn_out

#%% Available simulations
def calc_isentropic():
    """For a range of total entropy and fixed filling in system species, find the
    equilibrium conditions."""
    # Initialization
    isen_T_list = np.zeros_like(offset_list)
    isen_rrst_list = [None for _ in offset_list]
    isen_exp_list = [None for _ in offset_list]
    for i, offset in enumerate(offset_list):
        E_range_rsv = (offset, offset + bandwidth + 1) # energies considered in calculation
        DoS_rsv = make_rsv_fn(offset)
        exp_fksr = E9M.NVT_exp(1, [
        {"name": "fermi1", "V": V_s, "Np": Np_sf1, "stat": +1, "DoS": DoS_sys,
            "E_range": E_range_s, "reservoir": "", "comment": {}},
        {"name": "fermi1_rsv", "V": V_rsv, "Np": 0, "stat": +1, "DoS": DoS_rsv,
            "E_range": E_range_rsv, "reservoir": "fermi1", "comment": {}},
            ])

        logging.debug("working on loop #{}, offset = {}...".format(i, offset))
        isen_T_list[i], isen_rrst_list[i] = eqfind.NVT_from_NVS_solver(S_tot, exp_fksr)
        
        exp_now = deepcopy(exp_fksr)
        exp_now.T = isen_T_list[i]
        exp_now.find_outputs()
        isen_exp_list[i] = deepcopy(exp_now)
    
    # Plots
    # Entropy scan
    fig_exp = plt.figure(1)
    ax_SvsT = fig_exp.add_subplot(221)
    ax_SvsT.plot(offset_list, isen_T_list)
    ax_SvsT.set_xlabel("offset")
    ax_SvsT.set_ylabel("T")

    S_sys = np.array([x.subregion_list[0]["S"] for x in isen_exp_list])
    ax_S_sys = fig_exp.add_subplot(222)
    ax_S_sys.plot(offset_list, S_sys / r_s / S_tot)
    ax_S_sys.set_xlabel("offset")
    ax_S_sys.set_ylabel(r"$S_{sys}/(r_s S_{tot})$")
    ax_S_sys.set_title("Effective reduction in {}".format(r"$S_{sys}$"))

    S_fermi1 = np.array([x.subregion_list[0]["S"] for x in isen_exp_list])
    ax_Srel = fig_exp.add_subplot(223)
    ax_Srel.plot(offset_list, S_fermi1 / S_sys)
    ax_Srel.set_xlabel("offset")
    ax_Srel.set_ylabel(r"$S_{sys}^{fermi1}/S_{tot}$")

    N_fermi1 = np.array([x.subregion_list[0]["Np"] + x.subregion_list[1]["Np"] for x in isen_exp_list])
    N_fermi1_sys = np.array([x.subregion_list[0]["Np"] for x in isen_exp_list])
    ax_Np = fig_exp.add_subplot(224)
    ax_Np.plot(offset_list, N_fermi1, label = "tot")
    ax_Np.plot(offset_list, N_fermi1_sys, color = "black", ls = "--", label = "sys")
    ax_Np.set_xlabel("offset")
    ax_Np.set_ylabel(r"$N_p$")
    ax_Np.set_ylim(bottom = 0)
    ax_Np.set_title("# of fermi1 particles")
    ax_Np.legend()

    fig_exp.suptitle("Fermi-fermi with reservoirs in a kagome lattice")
    fig_exp.tight_layout()

    # Sample run
    fig_sam = plt.figure(2, figsize = (5,8))
    sampled_run = 5

    ax_DoS = fig_sam.add_subplot(211)
    isen_exp_list[sampled_run].plot_DoSs(ax_DoS, offset_traces = True)

    ax_tab = fig_sam.add_subplot(212)
    ax_tab.set_axis_off()
    isen_exp_list[sampled_run].tabulate_params(ax_tab)

    fig_sam.suptitle("Showing offset = {:.2f}".format(offset_list[sampled_run]))

#%% Code execution
available_modes = {
    "isentropic": calc_isentropic,
}

def main(**kwargs):
    logging.debug("kwargs: {}".format(kwargs))
    calculation_mode = kwargs["calculation_mode"]

    if calculation_mode not in available_modes.keys():
        raise Exception("{} is not defined".format(calculation_mode))
    else:
        available_modes[calculation_mode]()

if __name__ == "__main__":
    main(calculation_mode = "isentropic")