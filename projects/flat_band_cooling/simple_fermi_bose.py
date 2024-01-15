import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util
import E9_fn.E9_models as E9M
# mixture of spinless fermions and spinless bosons in a uniform potential

#%% Experiment initialization 
V = 3 * 100**2      # size of the experiment (imagine a kagome lattice with n*n unit cells)
T = 0.2             # (fundamental) temperature (k_B * T)
bandwidth = 6       # bandwidth (of band structure; 6 for tight-binding kagome lattice)
E_range_exp = (0, bandwidth + 1)    # energies considered in calculation
    # Defining DoS
simple_2D_DoS = lambda x: np.tanh(E_range_exp[1] - x) * util.rect_fn(x, 0, E_range_exp[1])
norm_factor = quad(simple_2D_DoS, E_range_exp[0], E_range_exp[1])[0]
simple_flatband_DoS = lambda x: (2/3) * simple_2D_DoS(x) \
    + (1/3) * norm_factor * util.Gaussian_1D(x, s = 0.1, mu = 4)
DoS_exp = simple_flatband_DoS

# fermion specific
nu_f = 5/6               # filling factor (1 is fully filled)
Np_f = int(V * nu_f)        # number of fermions

# boson specific
nu_b = 1/6               # filling factor (1 is one particle per site)
Np_b = int(V * nu_b)        # number of bosons

step_pot_exp = E9M.DoS_exp(T, [
    {"name": "fermion", "V": V, "Np": Np_f, "stat": 1, "DoS": DoS_exp,
        "E_range": E_range_exp, "reservoir": "", "comment": {}},
    {"name": "boson", "V": V, "Np": Np_b, "stat": -1, "DoS": DoS_exp,
        "E_range": E_range_exp, "reservoir": "", "comment": {}},
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