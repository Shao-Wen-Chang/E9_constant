import numpy as np
import matplotlib.pyplot as plt
import util
import E9_models as E9M
from scipy.integrate import quad
# spinless fermion in a step potential with flat band dispersion

#%% Experiment initialization 
#   "system": 1/10 of the number of sites, half-filled
#   "reservoir": 9/10 of the number of sites with lower on-site potential
V = 3 * 100**2      # size of the experiment (imagine a kagome lattice with n*n unit cells)
T = 0.4             # (fundamental) temperature (k_B * T)
bandwidth = 6       # bandwidth (of band structure; 6 for tight-binding kagome lattice)

# system specific
r_sys = 0.1            # ratio of the system size
nu = 5/6               # filling factor (1 is fully filled)
V_sys = int(V * r_sys) # size of the system
Np_sys = int(V * r_sys * nu)        # number of particles in the system
E_range_sys = (0, bandwidth + 1)    # energies considered in calculation
    # Defining DoS
simple_2D_DoS = lambda x: np.tanh(E_range_sys[1] - x) * util.rect_fn(x, 0, E_range_sys[1])
norm_factor = quad(simple_2D_DoS, E_range_sys[0], E_range_sys[1])[0]
simple_flatband_DoS = lambda x: (2/3) * simple_2D_DoS(x) \
    + (1/3) * norm_factor * util.Gaussian_1D(x, s = 0.1, mu = 4)
DoS_sys = simple_flatband_DoS

# reservoir specific
offset = 1.5           # energy offset in the DoS of reservoir
V_rsv = int(V - V_sys) # size of the reservoir
E_range_rsv = (offset, offset + bandwidth + 1) # energies considered in calculation
DoS_rsv = lambda x: simple_flatband_DoS(x - offset)

step_pot_exp = E9M.DoS_exp(T, [
    {"name": "spin_up", "V": V_sys, "Np": Np_sys, "stat": "fermi", "DoS": DoS_sys,
        "E_range": E_range_sys, "reservoir": "", "comment": {}},
    {"name": "spin_up_rsv", "V": V_rsv, "Np": 0, "stat": "fermi", "DoS": DoS_rsv,
        "E_range": E_range_rsv, "reservoir": "spin_up", "comment": {}},
])

#%% simulation
if __name__ == "__main__":
    step_pot_exp.find_outputs()
    fig_exp = plt.figure(1, figsize = (5, 8))
    ax_DoS = fig_exp.add_subplot(211)
    ax_table = fig_exp.add_axes(212)
    ax_table.set_axis_off()
    step_pot_exp.plot_DoSs(ax_DoS)
    step_pot_exp.tabulate_params(ax_table)
    plt.tight_layout()