import sys
from pathlib import Path
import scienceplots

plt.style.use(['nature'])

# User defined modules
E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
if str(E9path) not in sys.path:
    sys.path.insert(1, str(E9path))
import E9_fn.E9_constants as E9c

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

hex_vertices = [E9c.Mp, E9c.Mp2, E9c.Mp3, E9c.Mp4, E9c.Mp5, E9c.Mp6, E9c.Mp]
legs = [v * 1.5 for v in hex_vertices[:-1]]

_arrow_dir = [-E9c.K1, -E9c.K2, -E9c.K3]
arrow_tips = [-1.05 * v for v in _arrow_dir]
arrow_ends = [-1.6 * v for v in _arrow_dir]

str_journal = "BIGGUS"  # for screen display
str_journal = "nature"

mm2inch = 1 / 25.4
plt_sizes_all_journals = {
    "nature": {
        "fig_figsize":      np.array([20, 30]) * mm2inch,

        "AB_site_size":     16,
        "AB_site_lw":       0.2,

        "honeycomb_lw":     1.5,
        "a_hc_H_lw":        0.5,
        "coor_lw":          0.5,
        "lat_beams_lw":     0.5,

        "lat_arrow_style":  '-|>,head_width=0.01,head_length=0.02',

        "a_hc_fontsize":    7,
        "AB_fontsize":      7,
        "coor_fontsize":    7,
        "delta_fontsize":   6,
    },
    "BIGGUS": {
        "fig_figsize":      np.array([4, 6]),

        "AB_site_size":     400,
        "AB_site_lw":       1,
        
        "honeycomb_lw":     5,
        "a_hc_H_lw":        3,
        "coor_lw":          3,
        "lat_beams_lw":     4,

        "lat_arrow_style":  '-|>,head_width=0.05,head_length=0.12',

        "a_hc_fontsize":    35,
        "AB_fontsize":      38,
        "coor_fontsize":    35,
        "delta_fontsize":   30,
    }
}

plt_sizes = plt_sizes_all_journals[str_journal]

#%% Plot
print(f"For {str_journal} submission")

fig, ax1 = plt.subplots(figsize = plt_sizes["fig_figsize"])
ax1.set_aspect("equal")
ax1.axis("off")
fig.tight_layout(pad = 0)

# honeycomb lattice stuff
idx_label_a0 = 3
# "H" is the thing that indicates what a0 is
H_a0_close = np.array([np.sqrt(3), 1]) * 0.1 * E9c.k_lw
H_a0_far = np.array([np.sqrt(3), 1]) * 0.2 * E9c.k_lw
for i_pt, (p1, p2, pl) in enumerate(zip(hex_vertices[:-1], hex_vertices[1:], legs)):
    fc = np.full(3, fill_value = i_pt % 2)
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color = "k", lw = plt_sizes["honeycomb_lw"])   # hexagon
    ax1.plot([p1[0], pl[0]], [p1[1], pl[1]], color = "k", lw = plt_sizes["honeycomb_lw"])   # legs
    ax1.scatter(*p1, s = plt_sizes["AB_site_size"], facecolor = fc, edgecolor = "grey",
                lw = plt_sizes["AB_site_lw"], zorder = 100)
    # label a_hc
    if i_pt == idx_label_a0:
        H1_close = p1 + H_a0_close
        H2_close = p2 + H_a0_close
        H1_far = p1 + H_a0_far
        H2_far = p2 + H_a0_far
        Hmid1 = (H1_close + H1_far) / 2
        Hmid2 = (H2_close + H2_far) / 2
        ax1.plot([H1_close[0], H1_far[0]], [H1_close[1], H1_far[1]], color = "k", lw = plt_sizes["a_hc_H_lw"])
        ax1.plot([H2_close[0], H2_far[0]], [H2_close[1], H2_far[1]], color = "k", lw = plt_sizes["a_hc_H_lw"])
        ax1.plot([Hmid1[0], Hmid2[0]], [Hmid1[1], Hmid2[1]], color = "k", lw = plt_sizes["a_hc_H_lw"])
ax1.text(*hex_vertices[5] + np.array([0., -0.2]) * E9c.k_lw, r"$A$",
         fontsize = plt_sizes["AB_fontsize"], ha = "left", va = "top")
ax1.text(*hex_vertices[4] + np.array([0., -0.2]) * E9c.k_lw, r"$B$",
         fontsize = plt_sizes["AB_fontsize"], ha = "right", va = "top")
ax1.text(*hex_vertices[idx_label_a0] + np.array([0.05, 0.45]) * E9c.k_lw, r"$a_{\text{hc}}$",
         fontsize = plt_sizes["a_hc_fontsize"], ha = "left", va = "bottom")

# orientation
coor_pos = np.array([-1.25, -2.3]) * E9c.k_lw
x_axis_tip = np.array([0.8, 0]) * E9c.k_lw + coor_pos
y_axis_tip = np.array([0, 0.8]) * E9c.k_lw + coor_pos
for p1 in (x_axis_tip, y_axis_tip):
    arrow = FancyArrowPatch(
        coor_pos, p1,
        edgecolor = "black",
        facecolor = "black",
        linewidth = plt_sizes["coor_lw"],
        arrowstyle = plt_sizes["lat_arrow_style"],
        mutation_scale = 100,
        shrinkA = 0,
        shrinkB = 0
    )
    ax1.add_patch(arrow)
ax1.text(*x_axis_tip + np.array([0., 0.1]) * E9c.k_lw, r"$x$", fontsize = plt_sizes["coor_fontsize"])
ax1.text(*y_axis_tip + np.array([0.2, -0.2]) * E9c.k_lw, r"$y$", fontsize = plt_sizes["coor_fontsize"])

# lattice beam stuff
for p1, p2 in zip(arrow_ends, arrow_tips):
    arrow = FancyArrowPatch(
        p1, p2,
        edgecolor = "red",
        facecolor = "red",
        linewidth = plt_sizes["lat_beams_lw"],
        arrowstyle = plt_sizes["lat_arrow_style"],
        mutation_scale = 100,
        shrinkA = 0,
        shrinkB = 0
    )
    ax1.add_patch(arrow)
ax1.text(*arrow_ends[2] + np.array([-0.08, -0.35]) * E9c.k_lw, r"$\delta_1(t)$", fontsize = plt_sizes["delta_fontsize"])
ax1.text(*arrow_ends[0] + np.array([-0.63, -0.35]) * E9c.k_lw, r"$\delta_2(t)$", fontsize = plt_sizes["delta_fontsize"])

fig1apath = Path(r"C:\Users\ken92\Documents\Studies\E5\simulation\E9_simulations\projects\band_spectroscopy\paper_figures\fig1",
                 f"fig1c_{str_journal}.svg")
fig.savefig(fig1apath, format = "svg", facecolor = "none")
# %%
