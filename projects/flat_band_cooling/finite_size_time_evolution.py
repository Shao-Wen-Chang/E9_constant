import sys
import logging
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from pathlib import Path

E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
import equilibrium_finder as eqfind # For some reason removing this line gives an error (ModuleNotFoundError: No module named 'E9_fn')
sys.path.insert(1, E9path)
from E9_fn import util
import E9_fn.thermodynamics as thmdy
from E9_fn.tight_binding import E9tb

# User defined modules
E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
import equilibrium_finder as eqfind # For some reason removing this line gives an error (ModuleNotFoundError: No module named 'E9_fn')
sys.path.insert(1, E9path)
from E9_fn import util
import E9_fn.E9_models as E9M
# (non-interacting / spinless) fermions dynamics in a system + reservoir setup
'''
 - Initialize the system with some thermal state with temperature T with no offset between
   reservoir and system, and fill to somewhere in the flat band
 - Find mu and S for the initial configuration
'''
logpath = '' # '' if not logging to a file
loglevel = logging.INFO
logroot = logging.getLogger()
list(map(logroot.removeHandler, logroot.handlers))
list(map(logroot.removeFilter, logroot.filters))
logging.basicConfig(filename = logpath, level = loglevel)

#%% Initialization
lattice_str = "kagome"
lattice_len = 10
lattice_dim = (lattice_len, lattice_len)
tb_params = E9tb.get_model_params(lattice_str)
my_tb_model= E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)
H_bare = my_tb_model.H

sys_len = 6
sys_range = ((lattice_len - sys_len) // 2, (lattice_len + sys_len) // 2)
n_sys = sys_len**2

T_init = 0.2
nu_tar_system = 1/3     # Target filling factor in the system
nu_tar_reservoir = 5/6  # Target filling factor in the reservoir
N_init = (lattice_len - sys_len)**2 * nu_tar_reservoir + sys_len**2 * nu_tar_system

t_tot = 5
t_step = 0.05
n_steps = int(t_tot // t_step)
V_rsv_offset_final = -2
all_mus = np.zeros(n_steps)
all_Ss = np.zeros(n_steps)
all_Ts = np.zeros(n_steps)
all_snapshots = [None for _ in range(n_steps)]
