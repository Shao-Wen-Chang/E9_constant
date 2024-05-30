import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline
import sys
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util

#%% main
def main():
    num_pts = 501
    mu = 0.6
    V_sys = 0.5
    V_res = 0.35
    V_box = 0.8
    Use_box = True
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
    V_DMD = V_target - V_lat_har

    # Plot
    fig, ax = util.make_simple_axes(fig_kwarg = {"figsize": (8, 3)})
    ax.plot(x_axis, V_lat_har, color = "#f16951", linewidth = 4, label = r"$V_{lat}$")
    ax.plot(x_axis, V_target, color = "k", linewidth = 1, label = r"$V_{target}$")
    ax.plot(x_axis, V_DMD, label = r"$V_{DMD}$")
    ax.hlines(mu, -1, 1, colors = ["k"], linestyles = ["--"])
    ax.fill_between(x_axis, V_target, mu, V_target < mu, alpha = 0.5)
    ax.legend()
    ax.set_axis_off()

if __name__ == "__main__":
    main()