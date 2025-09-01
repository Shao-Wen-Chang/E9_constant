import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Units:
#   a_lat = 1       (lattice constant)
#   V_lat = 1       (lattice depth)
xlim_plt = 15
x_sys = 4.5
x_rsv = 10.5
N_x_pt = 2000
V_rsv_offset = -0.3
a_har = 0.01        # Vx = a_har * x**2

V_har_at_edge = a_har * x_rsv**2 - 0.4
xrange = np.linspace(-xlim_plt, xlim_plt, N_x_pt)
Vx_lat = np.cos(xrange * 2 * np.pi)
Vx_har_and_box = np.where(abs(xrange) < x_rsv, V_har_at_edge, a_har * xrange**2)
Vx_rsv_offset = np.where(np.logical_and(abs(xrange) < x_rsv, abs(xrange) > x_sys), V_rsv_offset, 0)
Vx_tot_init = Vx_har_and_box 
Vx_tot_offset = Vx_har_and_box + Vx_rsv_offset
Vx_tot_lat = Vx_lat + Vx_har_and_box + Vx_rsv_offset

#%% Plots
mag_fig = 1
bool_save_fig = True and (mag_fig == 1)
fig = plt.figure(figsize = (1.6 * mag_fig, 2 * mag_fig))
mu_init = 0.9
ax_init = fig.add_subplot(311)
ax_init.plot(xrange, Vx_tot_init, color = "black", lw = 0.5 * mag_fig)
ax_init.fill_between(xrange, Vx_tot_init, mu_init, where = mu_init > Vx_tot_init
                     , color = "red", alpha = 0.5, ls = "None")
ax_init.set_axis_off()

mu_offset_on = 0.8
ax_offset = fig.add_subplot(312)
ax_offset.plot(xrange, Vx_tot_offset, color = "black", lw = 0.5 * mag_fig)
ax_offset.fill_between(xrange, Vx_tot_offset, mu_offset_on, where = mu_offset_on > Vx_tot_offset
                       , color = "red", alpha = 0.5, ls = "None")
ax_offset.set_axis_off()

mu_lat_on = 0.1
ax_lat = fig.add_subplot(313)
ax_lat.plot(xrange, Vx_tot_lat, color = "black", lw = 0.5 * mag_fig)
ax_lat.fill_between(xrange, Vx_tot_lat, mu_lat_on, where = mu_lat_on > Vx_tot_lat
                    , color = "red", alpha = 0.5, ls = "None")
ax_lat.set_axis_off()

fig1apath = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "Projects",
                "2023 Optical potential engineering", "paper", "fig1", "fig1a_v1.svg")
if bool_save_fig:
    fig.savefig(fig1apath, format = "svg")