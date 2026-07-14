import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scienceplots

plt.style.use(['nature'])

# Time axis
t = np.linspace(0, 1, 1001)
t_ramp_up = 0.4
t_hold_end = 0.55
t_shake_end = 0.9
t_BM_end = 0.98
tticks = [0.0, t_hold_end, t_shake_end]
tticklabels = ['', '', '']

# Colors
c_int = "#005CEF"
c_pos = "#FF6A00"

str_journal = "BIGGUS"
# str_journal = "nature"

mm2inch = 1 / 25.4
plt_sizes_all_journals = {
    "nature": {
        "fig_figsize":      np.array([35, 30]) * mm2inch,
        "ax1_frame_lw":     0.1,

        "plot_lw":          0.7,

        "t_PM_height":      0.01,
        "font_size":        7,
        
        "tick_width":       1,
        "xlabel_fontsize":  7,
        "ylabel_fontsize":  6,
    },
    "BIGGUS": {
        "fig_figsize":      np.array([4., 3.5]),
        "ax1_frame_lw":     1,

        "plot_lw":          2,

        "t_PM_height":      0.02,
        "font_size":        26,
        
        "tick_width":       2,
        "xlabel_fontsize":  24,
        "ylabel_fontsize":  24,
    }
}

plt_sizes = plt_sizes_all_journals[str_journal]

#%% Intensity curve (blue, left y-axis)
I = np.zeros_like(t)
I_max = 0.7
y_lim_plt = I_max + 0.05

# 0 to t_ramp_up: ramp from 0 to t_hold_end (linear for illustration)
mask_1 = (t >= 0.0) & (t < t_ramp_up)
I[mask_1] = I_max * (t[mask_1] / t_ramp_up)

# t_ramp_up to t_hold_end: constant t_hold_end
mask_2 = (t >= t_ramp_up) & (t < t_shake_end)
I[mask_2] = I_max

# t_hold_end to 1.0: back to 0 (already zero by initialization)

# Band mapping curve
t_BM = np.linspace(t_shake_end, t_BM_end, 201)
I_BM = I_max * np.exp(-(t_BM - t_shake_end) * 50)

#%% Position curve (red, right y-axis)
pos_offset = I_max / 2
pos = np.full_like(t, pos_offset)

# Oscillating piece: t_hold_end <= t <= t_shake_end
n_periods = 6
A_osc = 0.15
t_osc_ramp = 0.06
t_start = t_hold_end
t_rampend = t_hold_end + t_osc_ramp
t_end = t_shake_end
delta_t = t_end - t_start

# Angular frequency omega such that we get 4 periods in [t_hold_end, t_shake_end]
omega = n_periods * 2 * np.pi / delta_t  # this is your "f" in sin(f * T)

ramp_mask = (t >= t_start) & (t <= t_rampend)
pos[ramp_mask] = pos_offset +   A_osc * np.sin(np.pi * (t[ramp_mask] - t_start) / t_osc_ramp / 2) *\
                                np.sin(omega * (t[ramp_mask] - t_start))
osc_mask = (t >= t_rampend) & (t <= t_end)
pos[osc_mask] = pos_offset + A_osc * np.sin(omega * (t[osc_mask] - t_start))

#%% Plot
print(f"For {str_journal} submission")

fig, ax1 = plt.subplots(figsize = plt_sizes["fig_figsize"])
for spine in ax1.spines.values():
    spine.set_linewidth(plt_sizes["ax1_frame_lw"])

# Left axis: intensity
ax1.plot(t, I, color=c_int, lw = plt_sizes["plot_lw"], label='Intensity')
# ax1.plot(t_BM, I_BM, color=c_int, lw=2, ls = "--")      # band mapping
ax1.set_xlim(0, 1)
ax1.set_ylim(0, y_lim_plt)
ax1.set_xlabel('Time', fontsize = plt_sizes["xlabel_fontsize"], labelpad = 0)
ax1.set_ylabel(r'$V_0(t)$ [arb.]', color=c_int, fontsize = plt_sizes["ylabel_fontsize"], labelpad = 1)
ax1.set_yticks([])

# Custom x-ticks including t_M between 0.7 and t_shake_end
ax1.set_xticks(tticks, tticklabels)
ax1.tick_params(axis = 'x', labelcolor=c_int, direction = "in", width = plt_sizes["tick_width"])
# ax1.set_xticklabels(tticklabels)
ax1.text(
    (t_hold_end + t_shake_end) / 2, plt_sizes["t_PM_height"],
    r'$t_{\text{PM}}$',
    ha='center', va='bottom', fontsize = plt_sizes["font_size"]
)

# Right axis: position
ax2 = ax1.twinx()
ax2.set_ylim(0, y_lim_plt)
ax2.set_ylabel(r'$\vec{r}(t)$ [arb.]', color=c_pos, fontsize = plt_sizes["xlabel_fontsize"], labelpad = 2)
ax2.set_yticks([])

# Plot position in segments to handle dashed part
# Actually decided to just use a solid line
# mask_pos1 = (t >= 0.0) & (t <= 0.2)
# ax2.plot(t[mask_pos1], pos[mask_pos1], color=c_pos, lw=2, ls='--')
# mask_pos2 = (t > 0.2)# & (t < t_hold_end)
mask_pos2 = (t > 0.)# & (t < t_hold_end)
ax2.plot(t[mask_pos2], pos[mask_pos2], color=c_pos, lw = plt_sizes["plot_lw"], ls='-')

# Add a few step indicators (texts and shades)
# step_text_height = 0.67
# ax1.text(t_hold_end / 2, step_text_height, '①',
#          ha='center', va='top', fontsize = plt_sizes["font_size"])
# ax1.text((t_hold_end + t_shake_end) / 2, step_text_height, '②',
#          ha='center', va='top', fontsize = plt_sizes["font_size"])
# ax1.text((t_shake_end + 1) / 2, step_text_height, '③',
#          ha='center', va='top', fontsize = plt_sizes["font_size"])

# fill_color = (0, 0, 0, 0.2)
# ax1.fill_betweenx([0, y_lim_plt], [0, 0], [t_hold_end, t_hold_end],
#                   color = fill_color, lw = 0)
# ax1.fill_betweenx([0, y_lim_plt], [t_shake_end, t_shake_end], [1, 1],
#                   color = fill_color, lw = 0)

fig.tight_layout(pad = 0)
plt.show()

fig1bpath = Path(r"C:\Users\ken92\Documents\Studies\E5\simulation\E9_simulations\projects\band_spectroscopy\paper_figures\fig1",
                 f"fig1b_{str_journal}.svg")
fig.savefig(fig1bpath, format = "svg", facecolor = "none")
# %%
