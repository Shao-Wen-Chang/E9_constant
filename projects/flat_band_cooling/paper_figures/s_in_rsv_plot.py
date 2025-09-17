import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# User defined modules
E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
if str(E9path) not in sys.path:
    sys.path.insert(1, str(E9path))
from E9_fn import util

def S_flatband(nu):
    return -(1 - nu) * np.log(1 - nu) - nu * np.log(nu)

mag_fig = 3
bool_save_fig = True
cmap = plt.get_cmap('coolwarm')

fig_s, ax_s = plt.subplots(figsize = (1.6 * mag_fig, 1 * mag_fig))
nus = np.linspace(0.01, 0.99, 99)
all_B = [B for B in range(3)]
for B in all_B:
    B_color = util.get_color(B, all_B, cmap, assignment = "index", crange = (0.3, 1))
    s_rsv = S_flatband(nus) / (nus + B)
    ax_s.plot(nus, s_rsv, label = f"B = {B}", color = B_color)
    inflection_pts = util.find_sign_change(util.find_derivative(s_rsv))
    ax_s.scatter(nus[inflection_pts], s_rsv[inflection_pts]
                 , marker = 'o', facecolors = "None", edgecolors = B_color)

ax_s.hlines([0.5], 0, 1, color = 'k', ls = "--")
ax_s.legend(loc = "upper right", fancybox = False, edgecolor = "black")
ax_s.set_xlim((0, 1))
ax_s.set_xticks(np.linspace(0, 1, 6))
ax_s.set_ylim((0, 2))
ax_s.set_yticks(np.linspace(0, 2, 5))
ax_s.set_xlabel(r"$n_R$")
ax_s.set_ylabel(r"$s_R/k_B$")

fig1bpath = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "Projects",
                "2023 Optical potential engineering", "paper", "fig1", "fig1b_v1.png")
if bool_save_fig:
    fig_s.savefig(fig1bpath, format = "png")