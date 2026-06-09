import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
import sys
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util
from pathlib import Path

#%% main
def main():
    num_pts = 501
    mu = 0.6
    V_sys = 0.5
    V_res = 0.35
    V_box = 0.8
    Use_box = True
    smoothness = 15
    N_half_sys = 50

    # Generate potential shapes
    x_axis = np.linspace(-1, 1, num_pts)
    V_lat_har = np.linspace(-1, 1, num_pts)**2
    ind_mid = int((num_pts - 1) / 2)
    V_target = V_lat_har.copy()
    V_target[ind_mid - N_half_sys: ind_mid + N_half_sys + 1] = V_sys
    V_target[V_target < V_res] = V_res
    if Use_box:
        V_target[(V_target < V_box) & (V_target > V_res) & (V_target != V_sys)] = V_box
    if smoothness > 1:
        V_target_cs = PchipInterpolator(x_axis[::smoothness], V_target[::smoothness])
        V_target = V_target_cs(x_axis)
    V_DMD = V_target - V_lat_har

    # Plot
    fig, ax = plt.subplots(figsize = (7, 1.5))
    ax.plot(x_axis, V_lat_har, color = "#f16951", linewidth = 3, label = r"$V_{\mathrm{ext}}$")
    ax.plot(x_axis, V_target, color = "k", linewidth = 1, label = r"$V_{\mathrm{tar}}$")
    ax.plot(x_axis, V_DMD, label = r"$V_{\mathrm{DMD}}$")
    ax.hlines(mu, -1, 1, colors = ["k"], linestyles = ["--"], linewidth = 1, label = r"$E_F$")
    ax.fill_between(x_axis, V_target, mu, V_target < mu, alpha = 0.5)
    ax.fill_between(x_axis, 0, 1, np.isclose(V_target, V_sys), color = "#CCCCCC",
                    alpha = 0.5, transform = ax.get_xaxis_transform())
    ax.fill_between(x_axis, 0, 1, np.isclose(V_target, V_res), color = "#888888",
                    edgecolor = None, alpha = 0.5, transform = ax.get_xaxis_transform())
    ax.text(0, 0.8, r"$\mathcal{S}$", color = "black")
    ax.text(0.38, 0.8, r"$\mathcal{R}$", color = "black")
    ax.text(-0.44, 0.8, r"$\mathcal{R}$", color = "black")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_axis_off()

    thesis_fig_path_app = Path(r"C:\Users\ken92\Documents\Studies\E5\Theses\figures\outlook")
    file_name = "DMD_comic.pdf"
    file_full_path = Path(thesis_fig_path_app, file_name)
    fig.savefig(file_full_path, facecolor = "none", dpi = 300, bbox_inches='tight')

if __name__ == "__main__":
    main()
# %%
