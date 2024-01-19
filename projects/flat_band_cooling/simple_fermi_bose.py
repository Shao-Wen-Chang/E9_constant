import sys
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
# User defined modules
import equilibrium_finder as eqfind
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util
import E9_fn.E9_models as E9M
# mixture of spinless fermions and spinless bosons in a uniform potential

#%% Experiment initialization 
V = 3 * 100**2      # size of the experiment (imagine a kagome lattice with n*n unit cells)
T = 0.4             # (fundamental) temperature (k_B * T)
bandwidth = 6       # bandwidth (of band structure; 6 for tight-binding kagome lattice)
E_range_exp = (0, bandwidth + 1)    # energies considered in calculation
    # Defining DoS
simple_2D_DoS = lambda x: np.tanh(E_range_exp[1] - x) * util.rect_fn(x, 0, E_range_exp[1])
norm_factor = quad(simple_2D_DoS, E_range_exp[0], E_range_exp[1])[0]
simple_flatband_DoS = lambda x: (2/3) * simple_2D_DoS(x) \
    + (1/3) * norm_factor * util.Gaussian_1D(x, s = 0.1, mu = 4)
DoS_exp = simple_flatband_DoS

# fermion specific
nu_f = 4/6               # filling factor (1 is fully filled)
Np_f = int(V * nu_f)        # number of fermions

# boson specific
nu_b = 2/6               # filling factor (1 is one particle per site)
Np_b = int(V * nu_b)        # number of bosons

exp_fb = E9M.DoS_exp(T, [
    {"name": "fermion", "V": V, "Np": Np_f, "stat": 1, "DoS": DoS_exp,
        "E_range": E_range_exp, "reservoir": "", "comment": {}},
    {"name": "boson", "V": V, "Np": Np_b, "stat": -1, "DoS": DoS_exp,
        "E_range": E_range_exp, "reservoir": "", "comment": {}},
])

#%% simulation
if __name__ == "__main__":
    # What to calculate: (inputs specific for each calculation mode are defined below)
    #   simple - Find basic thermodynamic parameters of the experiment defined above
    #   isentropic - Find equilibirum conditions for total entropies in isen_S_list
    #                T is used as the initial guess of the first entry
    calculation_mode = "isentropic"

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
            print("working on loop #{}, S = {}...".format(i, S_now))
            isen_T_list[i], isen_rrst_list[i] = eqfind.isentropic_solver(S_now, exp_fb)
            
            exp_now = deepcopy(exp_fb)
            exp_now.T = isen_T_list[i]
            exp_now.find_outputs()
            isen_exp_list[i] = deepcopy(exp_now)
        
        # Plots
        fig_exp = plt.figure(1, figsize = (5,8))
        ax_SvsT = fig_exp.add_subplot(421)
        ax_SvsT.plot(isen_S_list, isen_T_list)
        ax_SvsT.set_xlabel(r"$S_{tot}$")
        ax_SvsT.set_ylabel("T")

        fermi_S = np.array([x.species_list[0]["S"] for x in isen_exp_list])
        ax_Srel = fig_exp.add_subplot(422)
        ax_Srel.plot(isen_S_list, fermi_S / isen_S_list)
        ax_Srel.set_xlabel(r"$S_{tot}$")
        ax_Srel.set_ylabel(r"$S_f/S_{tot}$")

        bose_BEC = np.array([x.species_list[1]["N_BEC"] for x in isen_exp_list])
        box_width = isen_S_list[1] - isen_S_list[0]
        ax_BEC = fig_exp.add_subplot(423)
        ax_BEC.bar(x = isen_S_list, height = bose_BEC, width = box_width)
        ax_BEC.set_xlabel(r"$S_{tot}$")
        ax_BEC.set_ylabel(r"$N_{BEC}^{b}$")

        tabulated_run = 2
        ax_tab = fig_exp.add_subplot(212)
        ax_tab.set_axis_off()
        isen_exp_list[tabulated_run].tabulate_params(ax_tab)
        ax_tab.set_title("Showing S = {}".format(isen_S_list[tabulated_run]))
        
        fig_exp.suptitle("Simple fermi-bose experiment")
        fig_exp.tight_layout()