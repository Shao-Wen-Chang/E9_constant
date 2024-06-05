# Recommended import call: import fermi_kagome_scan_reservoir as fksr
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
# User defined modules
from scipy.integrate import quad
import equilibrium_finder as eqfind
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util
import E9_fn.E9_models as E9M
# spin-1/2 fermions in a step potential with kagome lattice dispersion

#%% Experiment initialization 
# values with _scan are used if it is being scanned in the calculation; otherwise, the value without _scan is used

#   "system": 1/10 of the number of sites, half-filled
#   "reservoir": 9/10 of the number of sites with lower on-site potential
V = 3 * 500**2      # size of the experiment (imagine a kagome lattice with n*n unit cells)
bandwidth = 6       # bandwidth (of band structure; 6 for tight-binding kagome lattice)

# System
sys_name = "system"
r_s = 0.1           # ratio of the system size
fhbw = 0.01         # half band width of the flat band
E_range_s = (0, bandwidth + fhbw)  # energies considered in calculation
    # Defining DoS
DoS_sys = lambda x: util.kagome_DoS(x, hbw = bandwidth / 2, fhbw = fhbw)
# print("DoS integrates to {}".format(quad(DoS_sys, E_range_s[0], E_range_s[1])))

V_s = int(V * r_s)  # size of the system
rsv_rel_size = 1 / r_s - 1      # relative size of the reservoir, V_rsv / V_s

# Reservoirs
res_name = "reservoir"
offset = -2       # energy offsets in the DoS of reservoir
offset_scan = np.arange(-2.1, -1.5, 0.1)
V_rsv = int(V - V_s)            # size of the reservoir

# Whole experiment
sp_name = "fermi1"
T = 0.2             # Temperature
T_scan = np.linspace(0.025, 1, 40)
mu = 4              # chemical potential
mu_scan = np.linspace(3.8, 4.2, 5)

# Utility variables
calculation_mode = "offset_scan"
colormap_traces = "Blues"       # Currently using matplotlib built-in colormaps
nu_tar = 5/12       # Target filling factor
    # Plot parameters
# xlim_DoS = None
xlim_DoS = [0, 3]
ylim_Srel = [0., 1.1]
Tmu_sample = []
# Tmu_sample = [(0, 0), (0, 4), (20, 0), (20, 4)]
# Tmu_solve_sample = []
Tmu_solve_sample = [0, 3, 6]

# Initial conditions in terms of N, V and S (, which should set mu and T of the system)
V_actual = 3 * 50**2    # For me to come up with some reasonable guesses in the thermodynamic limit
nu_rsv = 5/6
s_target = 0.3          # Entropy per particle
s_scan = np.arange(3, 7) / 10.

thmdy_size_factor = V / V_actual
N_target = thmdy_size_factor * (V_actual * r_s * nu_tar + V_actual * (1 - r_s) * nu_rsv)
S_target = N_target * s_target
print("N_target = {}, S_target = {}".format(N_target, S_target))

# Initial guesses for the muVT solver
T_guess0 = 0.04
mu_guess0 = 6 + offset

def make_rsv_fn(offset):
    """Generator of reservoir DoS.
    
    My old code that uses lambda functions has some reference issues."""
    def fn_out(x):
        return DoS_sys(x - offset)
    return fn_out

#%% Available simulations
def calc_simple():
    """Find the thermodynamical values at some fixed mu, T, and offset."""
    E_range_rsv = (offset, offset + bandwidth + fhbw)
    DoS_rsv = make_rsv_fn(offset)
    exp_fksr = E9M.muVT_exp(T,
                            [E9M.muVT_subregion(sys_name, sp_name, V_s, +1, DoS_sys, E_range_s),
                             E9M.muVT_subregion(res_name, sp_name, V_rsv, +1, DoS_rsv, E_range_rsv)],
                             {sp_name: mu})
    fig_exp = plt.figure(1, figsize = (5, 8))
    ax_DoS = fig_exp.add_subplot(211)
    ax_table = fig_exp.add_axes(212)
    ax_table.set_axis_off()
    exp_fksr.plot_DoSs(ax_DoS)
    exp_fksr.tabulate_params(ax_table)
    plt.tight_layout()
    if xlim_DoS is not None: ax_DoS.set_xlim(xlim_DoS)

    # exp_fksr.plot_E_orb(2.1 * fhbw)

def calc_T_scan():
    """Find the thermodynamical values at some fixed mu and offset while scanning T."""
    # Initialization
    E_range_rsv = (offset, offset + bandwidth + fhbw)
    DoS_rsv = make_rsv_fn(offset)
    exp_list = [None for _ in T_scan]

    # Calculation (done when an muVT_exp is instantiated)
    for i, T in enumerate(T_scan):
        logging.debug("working on loop #{}, T = {}...".format(i, T))
        exp_list[i] = E9M.muVT_exp(T,
                                [E9M.muVT_subregion(sys_name, sp_name, V_s, +1, DoS_sys, E_range_s),
                                 E9M.muVT_subregion(res_name, sp_name, V_rsv, +1, DoS_rsv, E_range_rsv)],
                                 {sp_name: mu})
    S_tot_arr = np.array([ex.S for ex in exp_list])
    S_sys_arr = np.array([ex.results["system"]["S"] for ex in exp_list])
    S_rsv_arr = np.array([ex.results["reservoir"]["S"] for ex in exp_list])
    N_tot_arr = np.array([ex.N_dict[sp_name] for ex in exp_list])
    nu_sys_arr = np.array([ex.results["system"]["nu"] for ex in exp_list])
    nu_rsv_arr = np.array([ex.results["reservoir"]["nu"] for ex in exp_list])

    # Plots
    fig_exp = plt.figure(1)
    ax_TvsS = fig_exp.add_subplot(221)
    ax_TvsS.plot(T_scan, S_tot_arr)
    ax_TvsS.set_xlabel(r"$k_B T / t$")
    ax_TvsS.set_ylabel(r"$S_{tot}$")

    ax_TvsN = fig_exp.add_subplot(222)
    ax_TvsN.plot(T_scan, N_tot_arr)
    ax_TvsN.set_xlabel(r"$k_B T / t$")
    ax_TvsN.set_ylabel(r"$N_{tot}$")

    ax_Srel = fig_exp.add_subplot(223)
    ax_Srel.plot(T_scan, S_sys_arr / (S_rsv_arr / rsv_rel_size))
    ax_Srel.set_xlabel(r"$k_B T / t$")
    ax_Srel.set_ylabel(r"$s_{sys}/s_{rsv}$")

    ax_nu = fig_exp.add_subplot(224)
    ax_nu.plot(T_scan, nu_sys_arr)
    ax_nu.set_xlabel(r"$k_B T / t$")
    ax_nu.set_ylabel(r"$\nu_{sys}$")

    fig_exp.suptitle("mu = {}, offset = {}".format(mu, offset))
    fig_exp.tight_layout()

    # Sample run
    fig_sam = plt.figure(2, figsize = (5,8))
    sampled_run = 5

    ax_DoS = fig_sam.add_subplot(211)
    exp_list[sampled_run].plot_DoSs(ax_DoS)
    if xlim_DoS is not None: ax_DoS.set_xlim(xlim_DoS)

    ax_tab = fig_sam.add_subplot(212)
    ax_tab.set_axis_off()
    exp_list[sampled_run].tabulate_params(ax_tab)

    fig_sam.suptitle("Showing T = {:.2f}".format(T_scan[sampled_run]))

def calc_T_mu_scan():
    """Find the thermodynamical values at some fixed offset while scanning T and mu."""
    # Initialize plots
    colormap_mu = colormaps[colormap_traces].resampled(int(len(mu_scan) * 1.5))
    color_offset = int(len(mu_scan) * 0.5)
    fig_exp = plt.figure(1)
    ax_TvsS = fig_exp.add_subplot(221)
    ax_TvsS.set_xlabel(r"$k_B T / t$")
    ax_TvsS.set_ylabel(r"$_{tot}$")

    ax_TvsN = fig_exp.add_subplot(222)
    ax_TvsN.set_xlabel(r"$k_B T / t$")
    ax_TvsN.set_ylabel(r"$N_{tot}$")

    ax_Srel = fig_exp.add_subplot(223)
    ax_Srel.set_xlabel(r"$k_B T / t$")
    ax_Srel.set_ylabel(r"$s_{sys}/s_{rsv}$")

    ax_nu = fig_exp.add_subplot(224)
    ax_nu.set_xlabel(r"$k_B T / t$")
    ax_nu.set_ylabel(r"$\nu_{sys}$")

    fig_exp.suptitle("V = {}, offset = {}, flat band bandwidth = {}".format(V, offset, 2 * fhbw))
    fig_exp.tight_layout()

    # Calculation (done when an muVT_exp is instantiated)
    for j, mu in enumerate(mu_scan):
        # Initialization
        E_range_rsv = (offset, offset + bandwidth + fhbw)
        DoS_rsv = make_rsv_fn(offset)
        exp_list = [None for _ in T_scan]
        c_mu = colormap_mu(color_offset + j)
        for i, T in enumerate(T_scan):
            logging.debug("working on mu = {}, T = {}...".format(mu, T))
            exp_list[i] = E9M.muVT_exp(T,
                                    [E9M.muVT_subregion(sys_name, sp_name, V_s, +1, DoS_sys, E_range_s),
                                    E9M.muVT_subregion(res_name, sp_name, V_rsv, +1, DoS_rsv, E_range_rsv)],
                                    {sp_name: mu})
            # Plot samples
            if (i, j) in Tmu_sample:
                fig_sam = plt.figure(figsize = (8, 5))

                ax_DoS = fig_sam.add_subplot(211)
                exp_list[i].plot_DoSs(ax_DoS)
                if xlim_DoS is not None: ax_DoS.set_xlim(xlim_DoS)

                ax_tab = fig_sam.add_subplot(212)
                ax_tab.set_axis_off()
                exp_list[i].tabulate_params(ax_tab)

                fig_sam.suptitle("Showing T = {:.2f}, mu = {:.2f}".format(T, mu))
        
        # Plot traces for this mu
        S_tot_arr = np.array([ex.S for ex in exp_list])
        S_sys_arr = np.array([ex.results["system"]["S"] for ex in exp_list])
        S_rsv_arr = np.array([ex.results["reservoir"]["S"] for ex in exp_list])
        N_tot_arr = np.array([ex.N_dict[sp_name] for ex in exp_list])
        nu_sys_arr = np.array([ex.results["system"]["nu"] for ex in exp_list])
        nu_rsv_arr = np.array([ex.results["reservoir"]["nu"] for ex in exp_list])

        ax_TvsS.plot(T_scan, S_tot_arr, color = c_mu, label = "{}{:.2f}".format(r"$\mu=$", mu))
        ax_TvsN.plot(T_scan, N_tot_arr, color = c_mu, label = "{}{:.2f}".format(r"$\mu=$", mu))
        ax_Srel.plot(T_scan, S_sys_arr / (S_rsv_arr / rsv_rel_size), color = c_mu, label = "{}{:.2f}".format(r"$\mu=$", mu))
        ax_nu.plot(T_scan, nu_sys_arr, color = c_mu, label = "{}{:.2f}".format(r"$\mu=$", mu))
        ax_nu.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Additional plotting
    ax_Srel.hlines(1, 0, 1, colors = "k", linestyles = "--")    # cooling when below 1
    ax_Srel.set_ylim(ylim_Srel[0], min(ylim_Srel[1], ax_Srel.get_ylim()[1]))
    ax_nu.hlines(nu_tar, 0, 1, colors = "k", linestyles = "--") # at the 2nd band vHS

def calc_offset_scan():
    """Given some S and N, determine what are the resultant mu and T for each offset considered.
    
    This mainly shows that when the particle number allows, you can dump a lot of particles (and entropies) in
    the flat band, and the system region is always cold. See 2024/06/04 personal notes."""
    # Initialization
    all_T = np.zeros_like(offset_scan)
    all_mu = np.zeros_like(offset_scan)
    all_S = np.zeros_like(offset_scan)
    all_N = np.zeros_like(offset_scan)
    all_nu_sys = np.zeros_like(offset_scan)
    all_success = [False for _ in offset_scan]
    all_exp = [None for _ in offset_scan]

    for i, offset in enumerate(offset_scan):
        print("working on offset = {}...".format(offset))
        T_guess, mu_guess = all_T[i - 1], all_mu[i - 1]
        if T_guess == 0: T_guess = T_guess0
        if mu_guess == 0: mu_guess = mu_guess0
        E_range_rsv = (offset, offset + bandwidth + fhbw)
        DoS_rsv = make_rsv_fn(offset)
        subregion_list = [E9M.muVT_subregion(sys_name, sp_name, V_s, +1, DoS_sys, E_range_s),
                        E9M.muVT_subregion(res_name, sp_name, V_rsv, +1, DoS_rsv, E_range_rsv)]
        Tmu_out, orst = eqfind.muVT_from_NVS_solver(S_target,
                                                    N_target,
                                                    subregion_list,
                                                    T_guess,
                                                    mu_guess,
                                                    Tbounds = (0, 2),
                                                    mubounds = (3, bandwidth + fhbw),
                                                    options_dict = {"fatol": 1e-8, "xatol": 1e-8})
        if orst.success:
            exp_fksr = E9M.muVT_exp(Tmu_out[0], subregion_list, {sp_name: Tmu_out[1]})
            all_T[i] = Tmu_out[0]
            all_mu[i] = Tmu_out[1]
            all_S[i] = exp_fksr.S
            all_N[i] = exp_fksr.N_dict[sp_name]
            all_nu_sys[i] = exp_fksr.results[sys_name]["Np"] / V_s
            all_exp[i] = exp_fksr
            if all_N[i] - N_target < 100 and all_S[i] - S_target < 100: # pretty arbitrary
                all_success[i] = True
            else:
                logging.warning("offset = {} didn't converge, but the minimizer thought it did!".format(offset))
            if i in Tmu_solve_sample:
                fig_exp = plt.figure(figsize = (5, 8))
                ax_DoS = fig_exp.add_subplot(211)
                ax_table = fig_exp.add_axes(212)
                ax_table.set_axis_off()
                exp_fksr.plot_DoSs(ax_DoS)
                exp_fksr.tabulate_params(ax_table)
                plt.tight_layout()
                if xlim_DoS is not None: ax_DoS.set_xlim(xlim_DoS)
    
    fig_exp = plt.figure(figsize = (7, 8))
    ax_offsetvsT = fig_exp.add_subplot(321)
    ax_offsetvsT.plot(offset_scan[all_success], all_T[all_success])
    ax_offsetvsT.set_xlabel(r"$V_{offset} / t$")
    ax_offsetvsT.set_ylabel(r"$T$")

    ax_offsetvsmu = fig_exp.add_subplot(322)
    ax_offsetvsmu.plot(offset_scan[all_success], all_mu[all_success])
    ax_offsetvsmu.set_xlabel(r"$V_{offset} / t$")
    ax_offsetvsmu.set_ylabel(r"$\mu$")

    ax_Nerr = fig_exp.add_subplot(323)
    ax_Nerr.plot(offset_scan[all_success], all_N[all_success] - N_target)
    ax_Nerr.set_xlabel(r"$V_{offset} / t$")
    ax_Nerr.set_ylabel(r"$N - N_{tar}$")

    ax_Serr = fig_exp.add_subplot(324)
    ax_Serr.plot(offset_scan[all_success], all_S[all_success] - S_target)
    ax_Serr.set_xlabel(r"$V_{offset} / t$")
    ax_Serr.set_ylabel(r"$S - S_{tar}$")

    ax_nu_sys = fig_exp.add_subplot(325)
    ax_nu_sys.plot(offset_scan[all_success], all_nu_sys[all_success])
    ax_nu_sys.set_xlabel(r"$V_{offset} / t$")
    ax_nu_sys.set_ylabel(r"$\nu_{sys}$")
    fig_exp.suptitle("S = {}, N = {}, flat band bandwidth = {}".format(S_target, N_target, 2 * fhbw))

def calc_S_scan():
    """Given some N and offset, determine what are the resultant mu and T for each S considered.
    
    This shows that the cooling effect of the flat band only works if the temperature is very low."""
    # Initialization
    E_range_rsv = (offset, offset + bandwidth + fhbw)
    DoS_rsv = make_rsv_fn(offset)
    subregion_list = [E9M.muVT_subregion(sys_name, sp_name, V_s, +1, DoS_sys, E_range_s),
                      E9M.muVT_subregion(res_name, sp_name, V_rsv, +1, DoS_rsv, E_range_rsv)]
    all_T = np.zeros_like(s_scan)
    all_mu = np.zeros_like(s_scan)
    all_S = np.zeros_like(s_scan)
    all_N = np.zeros_like(s_scan)
    all_nu_sys = np.zeros_like(s_scan)
    all_success = [False for _ in s_scan]
    all_exp = [None for _ in s_scan]

    for i, s in enumerate(s_scan):
        print("working on s = {}...".format(s))
        T_guess, mu_guess = all_T[i - 1], all_mu[i - 1]
        if T_guess == 0: T_guess = T_guess0
        if mu_guess == 0: mu_guess = mu_guess0
        S_target = N_target * s

        Tmu_out, orst = eqfind.muVT_from_NVS_solver(S_target,
                                                    N_target,
                                                    subregion_list,
                                                    T_guess,
                                                    mu_guess,
                                                    Tbounds = (0, 20),
                                                    mubounds = (0, 10),
                                                    options_dict = {"fatol": 1e-8, "xatol": 1e-8})
        if orst.success:
            exp_fksr = E9M.muVT_exp(Tmu_out[0], subregion_list, {sp_name: Tmu_out[1]})
            all_T[i] = Tmu_out[0]
            all_mu[i] = Tmu_out[1]
            all_S[i] = exp_fksr.S
            all_N[i] = exp_fksr.N_dict[sp_name]
            all_nu_sys[i] = exp_fksr.results[sys_name]["Np"] / V_s
            all_exp[i] = exp_fksr
            if abs(all_N[i] - N_target) < 100 and abs(all_S[i] - S_target) < 100: # pretty arbitrary
                all_success[i] = True
            else:
                logging.warning("s = {} didn't converge, but the minimizer thought it did!".format(s))
                raise Exception("I want a perfect curve")
            if i in Tmu_solve_sample:
                fig_exp = plt.figure(figsize = (5, 8))
                ax_DoS = fig_exp.add_subplot(211)
                ax_table = fig_exp.add_axes(212)
                ax_table.set_axis_off()
                exp_fksr.plot_DoSs(ax_DoS)
                exp_fksr.tabulate_params(ax_table)
                plt.tight_layout()
                if xlim_DoS is not None: ax_DoS.set_xlim(xlim_DoS)
    
    fig_exp = plt.figure(figsize = (7, 8))
    ax_offsetvsT = fig_exp.add_subplot(321)
    ax_offsetvsT.plot(s_scan[all_success], all_T[all_success])
    ax_offsetvsT.set_xlabel(r"$s/k_B$")
    ax_offsetvsT.set_ylabel(r"$T$")

    ax_offsetvsmu = fig_exp.add_subplot(322)
    ax_offsetvsmu.plot(s_scan[all_success], all_mu[all_success])
    ax_offsetvsmu.set_xlabel(r"$s/k_B$")
    ax_offsetvsmu.set_ylabel(r"$\mu$")

    ax_Nerr = fig_exp.add_subplot(323)
    ax_Nerr.plot(s_scan[all_success], all_N[all_success] - N_target)
    ax_Nerr.set_xlabel(r"$s/k_B$")
    ax_Nerr.set_ylabel(r"$N - N_{tar}$")

    ax_Serr = fig_exp.add_subplot(324)
    ax_Serr.plot(s_scan[all_success], all_S[all_success] - N_target * s_scan[all_success])
    ax_Serr.set_xlabel(r"$s/k_B$")
    ax_Serr.set_ylabel(r"$S - S_{tar}$")

    ax_nu_sys = fig_exp.add_subplot(325)
    ax_nu_sys.plot(s_scan[all_success], all_nu_sys[all_success])
    ax_nu_sys.set_xlabel(r"$s/k_B$")
    ax_nu_sys.set_ylabel(r"$\nu_{sys}$")
    fig_exp.suptitle("offset = {}, N = {}, flat band bandwidth = {}".format(offset, N_target, 2 * fhbw))

#%% Code execution
available_modes = {
    "simple": calc_simple,
    "T_scan": calc_T_scan,
    "T_mu_scan": calc_T_mu_scan,
    "offset_scan": calc_offset_scan,
    "S_scan": calc_S_scan,
}

def main(**kwargs):
    logging.debug("kwargs: {}".format(kwargs))
    calculation_mode = kwargs["calculation_mode"]

    if calculation_mode not in available_modes.keys():
        raise Exception("{} is not defined".format(calculation_mode))
    else:
        available_modes[calculation_mode]()

if __name__ == "__main__":
    main(calculation_mode = calculation_mode)    