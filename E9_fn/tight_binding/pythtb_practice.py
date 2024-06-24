import logging
import numpy as np
import matplotlib.pyplot as plt
import pythtb as tb
import sys
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
import E9_fn.E9_constants as E9c
import E9_fn.E9_numbers as E9n

# Initialization
vzero = np.array([0, 0])
a1 = np.array([0, 1])
a2 = np.array([-np.sqrt(3)/2, 1/2])
Mp = np.array([1/2, 0])
Kp = np.array([1/3, -1/3])
primitive_vectors = [2 * a1, 2 * a2]
basis_vectors = [vzero, [0.5, 0], [0, 0.5]] # In reduced coordinates (i.e. units in primitive vectors)!
kagome_inf = tb.tb_model(2, 2, primitive_vectors, basis_vectors)
t_nn = -1

kagome_inf.set_onsite([0., 0., 0.])
# (amplitude, i, j, [lattice vector to cell containing j])
# In reduced coordinates (i.e. units in primitive vectors)!
kagome_inf.set_hop(t_nn, 0, 1, [0, 0])
kagome_inf.set_hop(t_nn, 0, 2, [0, 0])
kagome_inf.set_hop(t_nn, 1, 2, [0, 0])
kagome_inf.set_hop(t_nn, 0, 1, [-1, 0])
kagome_inf.set_hop(t_nn, 0, 2, [0, -1])
kagome_inf.set_hop(t_nn, 1, 2, [1, -1])

# print tight-binding model
kagome_inf.display()
kagome_inf.visualize(0, 1)

# generate k-point path and labels
# In reduced coordinates (i.e. units in reciprocal vectors)!
path = [vzero, Mp, Kp, vzero]
label = (r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$')
(k_vec, k_dist, k_node) = kagome_inf.k_path(path, 301)
evals = kagome_inf.solve_all(k_vec)

fig, ax = plt.subplots()
ax.set_xlim(k_node[0],k_node[-1])
ax.set_xticks(k_node)
ax.set_xticklabels(label)
for n in range(len(k_node)):
  ax.axvline(x=k_node[n], linewidth=0.5, color='k')
for n in range(3):
  ax.plot(k_dist,evals[n])
ax.set_title("kagome band structure")
ax.set_xlabel("Path in k-space")
ax.set_ylabel("Band energy")
fig.tight_layout()