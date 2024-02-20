# Recommended import call: import simple_fermi_fermi as sff
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
# non-interacting spin-1/2 fermions in a uniform potential
# (Spins are labelled with numbers for ease of generalizing to >2 spin species)

#%% Experiment initialization 
V = 3 * 100**2      # size of the experiment (imagine a kagome lattice with n*n unit cells)
T = 0.05             # (fundamental) temperature (k_B * T)
bandwidth = 6       # bandwidth (of band structure; 6 for tight-binding kagome lattice)
E_range_exp = (0, bandwidth + 1)    # energies considered in calculation
    # Defining DoS
simple_2D_DoS = lambda x: np.tanh(E_range_exp[1] - x) * util.rect_fn(x, 0, E_range_exp[1])
norm_factor = quad(simple_2D_DoS, E_range_exp[0], E_range_exp[1])[0]
simple_flatband_DoS = lambda x: (2/3) * simple_2D_DoS(x) \
    + (1/3) * norm_factor * util.Gaussian_1D(x, s = 0.1, mu = 4)
DoS_exp = simple_flatband_DoS

# spin 1 specific
nu_1 = 3/6               # filling factor (1 is fully filled)
Np_1 = int(V * nu_1)        # number of fermions

# spin 2 specific
nu_2 = 3/6               # filling factor (1 is one particle per site)
Np_2 = int(V * nu_2)        # number of bosons

exp_fb = E9M.DoS_exp(T, [
    {"name": "fermion 1", "V": V, "Np": Np_1, "stat": 1, "DoS": DoS_exp,
        "E_range": E_range_exp, "reservoir": "", "comment": {}},
    {"name": "fermion 2", "V": V, "Np": Np_2, "stat": 1, "DoS": DoS_exp,
        "E_range": E_range_exp, "reservoir": "", "comment": {}},
])

#%% simulation
def main(**kwargs):
    # What to calculate: (inputs specific for each calculation mode are defined below)
    #   simple - Find basic thermodynamic parameters of the experiment defined above
    #   isentropic - Find equilibirum conditions for total entropies in isen_S_list
    #                T is used as the initial guess of the first entry
    calculation_mode = kwargs["calculation_mode"]

    if calculation_mode == "simple":
        exp_fb.find_outputs()

        fig_exp = plt.figure(1, figsize = (5, 8))
        ax_DoS = fig_exp.add_subplot(211)
        ax_table = fig_exp.add_axes(212)
        ax_table.set_axis_off()
        exp_fb.plot_DoSs(ax_DoS)
        exp_fb.tabulate_params(ax_table)
        plt.tight_layout()
    
    elif calculation_mode == "isentropic":
        # Additional inputs
        isen_S_list = np.linspace(10000, 20000, 11)
        # Additional initialization
        isen_T_list = np.zeros_like(isen_S_list)
        isen_rrst_list = [None for _ in isen_S_list]
        isen_exp_list = [None for _ in isen_S_list]

        for i, S_now in enumerate(isen_S_list):
            logging.info("working on loop #{}, S = {}...".format(i, S_now))
            isen_T_list[i], isen_rrst_list[i] = eqfind.isentropic_fix_filling_solver(S_now, exp_fb)
            
            exp_now = deepcopy(exp_fb)
            exp_now.T = isen_T_list[i]
            exp_now.find_outputs()
            isen_exp_list[i] = deepcopy(exp_now)
        
        # Plots
        fig_exp = plt.figure(1, figsize = (5,8))
        ax_SvsT = fig_exp.add_subplot(221)
        ax_SvsT.plot(isen_S_list, isen_T_list)
        ax_SvsT.set_xlabel(r"$S_{tot}$")
        ax_SvsT.set_ylabel("T")

        fermi_S = np.array([x.species_list[0]["S"] for x in isen_exp_list])
        ax_Srel = fig_exp.add_subplot(222)
        ax_Srel.plot(isen_S_list, fermi_S / isen_S_list)
        ax_Srel.set_xlabel(r"$S_{tot}$")
        ax_Srel.set_ylabel(r"$S_f1/S_{tot}$")

        tabulated_run = 2
        ax_tab = fig_exp.add_subplot(313)
        ax_tab.set_axis_off()
        isen_exp_list[tabulated_run].tabulate_params(ax_tab)
        ax_tab.set_title("Showing S = {}".format(isen_S_list[tabulated_run]))
        
        fig_exp.suptitle("Simple fermi-fermi experiment")
        fig_exp.tight_layout()

if __name__ == "__main__":
    main(calculation_mode = "isentropic")